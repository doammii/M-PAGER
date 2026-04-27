"""
Stage 3: Condition Augmentation

Attaches structured conditions to UMLS layer triples by comparing each triple
against its source CREST guideline recommendation(s).

Matching strategy:
  Triple head/tail names are UMLS preferred names while entity candidates use
  LLM-normalized forms. Direct string matching fails, so we use CUI-based
  matching via stage2_umls_matched.json:
    triple.head_cui → match_results → entity.source_guidelines → recommendations

  Bidirectional filter: a triple is sent to the LLM only when head AND tail
  both resolve to a shared recommendation. Triples without such a rec are
  passed through with empty conditions (has_conditions=False).

Cost optimization:
  - Pair cache for find_relevant_recommendations (head_cui, tail_id)
  - Global dedup by (head_cui, tail_id, rec_set): conditions depend on rec
    text + endpoints, not relation. One LLM call covers all triples sharing
    that group.
  - response_format=json_object: model closes JSON cleanly so we never need
    a hard max_tokens cap to control cost.

Condition types (4):
  numeric_threshold, categorical_state, medication_history, temporal_condition
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import config

logger = logging.getLogger(__name__)

_openai_client = None

# UMLS CUI: literal "C" + 7 digits. Tighter than startswith("C") so we don't
# misclassify source codes that happen to start with "C" (e.g. HCPCS "C1300").
_CUI_RE = re.compile(r"^C\d{7}$")


def _get_openai_client(api_key: str = None):
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=api_key or config.OPENAI_API_KEY)
    return _openai_client


# ──────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────

CONDITION_SYSTEM_PROMPT = """Extract conditions from guideline sentences for medical triples. 4 types: numeric_threshold (age>=65), categorical_state (diagnosis=diabetes), medication_history (drug status), temporal_condition (within 72h). Only include conditions constraining the triple. Keep evidence_text ≤50 chars. Return a JSON object: {"results":[{"triple_index":<int>, ...}]}."""

FEW_SHOT_USER = """Triple: ("Lung Neoplasms","screened_by","Low-dose CT")
Rec: "screening recommended for adults aged 55-74, 30+ pack-years, quit within 15 years" [id: g42]"""

FEW_SHOT_ASSISTANT = """{"results":[{"triple_index":0,"conditions":[{"type":"numeric_threshold","variable":"age","comparator":">=","value":55,"unit":"years","evidence_text":"aged 55-74"},{"type":"numeric_threshold","variable":"age","comparator":"<=","value":74,"unit":"years"},{"type":"numeric_threshold","variable":"smoking_pack_year","comparator":">=","value":30,"unit":"pack-years","evidence_text":"30+ pack-years"}],"condition_logic":"AND","condition_source":{"guideline_id":"g42","evidence_level":"sentence_aligned","evidence_texts":["aged 55-74","30+ pack-years"]}}]}"""


# ──────────────────────────────────────────────────────────────────
# CUI-based Recommendation Matching
# ──────────────────────────────────────────────────────────────────

def build_recommendation_index(
    recommendations: list[dict],
    entities: list[dict],
    match_results: list[dict],
) -> dict:
    """Build CUI→rec-indices and name→rec-indices lookup tables."""
    guideline_to_rec_idx: dict[str, list[int]] = {}
    for i, rec in enumerate(recommendations):
        gid = rec.get("guideline_id", "")
        if gid:
            guideline_to_rec_idx.setdefault(gid, []).append(i)

    cui_to_recs: dict[str, set] = {}
    for mr in match_results:
        if not mr.get("matched"):
            continue

        entity = mr.get("entity", {})
        rec_indices: set = set()
        for gid in entity.get("source_guidelines", []):
            for idx in guideline_to_rec_idx.get(gid, []):
                rec_indices.add(idx)

        if not rec_indices:
            continue

        for m in mr.get("matches", []):
            cui = m.get("cui", "")
            if _CUI_RE.match(cui):
                cui_to_recs.setdefault(cui, set()).update(rec_indices)

    # Name-based fallback for triples whose tail_id is not a CUI.
    unique_names: set[str] = set()
    for ent in entities:
        for k in ("normalized_form", "surface_form"):
            n = ent.get(k, "").lower().strip()
            if n and len(n) > 3:
                unique_names.add(n)

    rec_texts_lower = [rec.get("text", "").lower() for rec in recommendations]
    name_to_recs: dict[str, set] = {}
    for name in unique_names:
        hits = {i for i, t in enumerate(rec_texts_lower) if name in t}
        if hits:
            name_to_recs[name] = hits

    logger.info(
        f"Built recommendation index: {len(cui_to_recs)} CUIs mapped, "
        f"{len(name_to_recs)} name-based entries (fallback)"
    )

    return {"cui_to_recs": cui_to_recs, "name_to_recs": name_to_recs}


def find_relevant_recommendations(
    triple: dict,
    recommendations: list[dict],
    rec_index: dict,
    max_recs: int = None,
) -> list[dict]:
    """Recommendations mentioning BOTH endpoints. Empty list = skip LLM."""
    max_recs = max_recs or config.STAGE3_MAX_RECS_PER_TRIPLE

    cui_index = rec_index.get("cui_to_recs", {})
    name_index = rec_index.get("name_to_recs", {})

    head_cui = triple.get("head_cui", "")
    tail_id = triple.get("tail_id", "")
    head_name = triple.get("head_name", "").lower().strip()
    tail_name = triple.get("tail_name", "").lower().strip()

    head_recs = cui_index.get(head_cui, set())
    tail_recs = cui_index.get(tail_id, set()) if _CUI_RE.match(tail_id) else set()

    if not head_recs:
        head_recs = name_index.get(head_name, set())
    if not tail_recs:
        tail_recs = name_index.get(tail_name, set())

    both = head_recs & tail_recs
    if not both:
        return []

    # Sort for deterministic recs[0] (used for recommendation_strength).
    selected = sorted(both)[:max_recs]
    return [recommendations[i] for i in selected]


# ──────────────────────────────────────────────────────────────────
# LLM Call
# ──────────────────────────────────────────────────────────────────

def _build_user_message(
    triples_batch: list[dict],
    batch_recommendations: list[list[dict]],
) -> str:
    parts = ["[TRIPLES]"]
    for i, triple in enumerate(triples_batch):
        parts.append(
            f'Triple {i}:\n'
            f'  ("{triple["head_name"]}", "{triple["relation"]}", "{triple["tail_name"]}")'
        )

    parts.append("\n[RECOMMENDATION SENTENCES]")
    for i, recs in enumerate(batch_recommendations):
        for rec in recs:
            text = rec.get("text", "")
            if len(text) > config.STAGE3_REC_TEXT_MAX_CHARS:
                text = text[:config.STAGE3_REC_TEXT_MAX_CHARS].rsplit(" ", 1)[0] + "..."
            parts.append(
                f"Rec for Triple {i}:\n"
                f'  "{text}"\n'
                f"  [guideline_id: {rec.get('guideline_id', '')}, "
                f"strength: {rec.get('strength', '')}]"
            )

    return "\n".join(parts)


def _parse_condition_response(response_text: str) -> list[dict]:
    """Parse `{"results": [...]}` shape with a small truncation safety net."""
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Tier-1 salvage: trim to last `}` and synthesize closers. With
        # response_format=json_object we rarely reach this path, but if we do
        # the model has likely closed at least one object cleanly.
        last_brace = text.rfind("}")
        if last_brace != -1:
            for closer in ("]}", "}"):
                try:
                    data = json.loads(text[: last_brace + 1] + closer)
                    logger.warning("Recovered truncated JSON via brace-trim salvage")
                    break
                except json.JSONDecodeError:
                    continue

    if data is None:
        logger.warning(f"Failed to parse condition response: {text[:200]}")
        return []

    if isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, list):
            return results
        if isinstance(results, dict):
            return [results]
        return [data]
    if isinstance(data, list):
        return data
    return []


def extract_conditions_batch(
    triples_batch: list[dict],
    batch_recommendations: list[list[dict]],
    api_key: str = None,
    model: str = None,
    _retry_depth: int = 0,
) -> list[dict]:
    """Call LLM to extract conditions. Recursively retries only the missing
    triple_indices in small chunks so a partial failure doesn't trigger a full
    re-call."""
    client = _get_openai_client(api_key)
    model = model or config.LLM_MODEL

    messages = [
        {"role": "system", "content": CONDITION_SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_USER},
        {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
        {"role": "user", "content": _build_user_message(triples_batch, batch_recommendations)},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            reasoning_effort=config.LLM_REASONING_EFFORT,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.error(f"Condition extraction LLM call failed: {e}")
        return []

    data = _parse_condition_response(response.choices[0].message.content)

    expected = set(range(len(triples_batch)))
    seen = {r.get("triple_index", -1) for r in data if isinstance(r, dict)}
    missing = sorted(expected - seen)
    if not missing:
        return data

    if _retry_depth >= config.STAGE3_MAX_RETRY_DEPTH:
        logger.warning(
            f"Missing {len(missing)} result(s) after max retries; keeping partial"
        )
        return data

    logger.warning(
        f"Retrying {len(missing)}/{len(triples_batch)} missing triple(s) "
        f"at depth {_retry_depth + 1}"
    )

    for chunk_start in range(0, len(missing), config.STAGE3_RETRY_CHUNK_SIZE):
        chunk = missing[chunk_start: chunk_start + config.STAGE3_RETRY_CHUNK_SIZE]
        sub_triples = [triples_batch[i] for i in chunk]
        sub_recs = [batch_recommendations[i] for i in chunk]
        retry_data = extract_conditions_batch(
            sub_triples, sub_recs,
            api_key=api_key, model=model,
            _retry_depth=_retry_depth + 1,
        )
        for r in retry_data:
            if not isinstance(r, dict):
                continue
            local_idx = r.get("triple_index", -1)
            if isinstance(local_idx, int) and 0 <= local_idx < len(chunk):
                r["triple_index"] = chunk[local_idx]
                data.append(r)

    # Latest result per triple_index wins.
    merged_by_idx: dict = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        idx = item.get("triple_index", -1)
        if isinstance(idx, int):
            merged_by_idx[idx] = item
    return list(merged_by_idx.values())


# ──────────────────────────────────────────────────────────────────
# Condition Validation
# ──────────────────────────────────────────────────────────────────

VALID_CONDITION_TYPES = {
    "numeric_threshold", "categorical_state",
    "medication_history", "temporal_condition",
}

REQUIRED_FIELDS = {
    "numeric_threshold": {"variable", "comparator", "value"},
    "categorical_state": {"variable", "value"},
    "medication_history": {"drug", "status"},
    "temporal_condition": {"event", "anchor", "comparator"},
}


def _validate_condition(cond: dict) -> bool:
    ctype = cond.get("type", "")
    if ctype not in VALID_CONDITION_TYPES:
        return False
    return all(cond.get(f) is not None for f in REQUIRED_FIELDS.get(ctype, set()))


def _normalize_conditions(conditions: list) -> list[dict]:
    if not isinstance(conditions, list):
        return []
    valid = []
    for cond in conditions:
        if isinstance(cond, dict) and _validate_condition(cond):
            cond.setdefault("evidence_text", "")
            valid.append(cond)
    return valid


# ──────────────────────────────────────────────────────────────────
# Triple Augmentation Helpers
# ──────────────────────────────────────────────────────────────────

_EMPTY_COND_SOURCE = {"guideline_id": "", "evidence_level": "", "evidence_texts": []}


def _apply_no_conditions(triple: dict) -> None:
    """Empty-condition stamp for triples that didn't pass head/tail filter."""
    triple["conditions"] = []
    triple["condition_logic"] = None
    triple["condition_source"] = dict(_EMPTY_COND_SOURCE)
    triple["recommendation_strength"] = None
    triple["conditions_json"] = "[]"
    triple["has_conditions"] = False
    triple["parse_failed"] = False


def _apply_parse_failed(triple: dict) -> None:
    """LLM result missing after retry — Stage 4 excludes these from the graph."""
    _apply_no_conditions(triple)
    triple["parse_failed"] = True


def _apply_cr(triple: dict, cr: dict, recs: list[dict]) -> None:
    """Merge an LLM condition result into a triple."""
    valid_conditions = _normalize_conditions(cr.get("conditions", []))
    source = cr.get("condition_source", {})
    strength = recs[0].get("strength") if recs else None

    triple["conditions"] = valid_conditions
    triple["condition_logic"] = cr.get("condition_logic", "AND") if valid_conditions else None
    triple["condition_source"] = {
        "guideline_id": source.get("guideline_id", ""),
        "evidence_level": source.get("evidence_level", ""),
        "evidence_texts": source.get("evidence_texts", []),
    }
    triple["recommendation_strength"] = strength
    triple["conditions_json"] = (
        json.dumps(valid_conditions, ensure_ascii=False) if valid_conditions else "[]"
    )
    triple["has_conditions"] = bool(valid_conditions)
    triple["parse_failed"] = False


# ──────────────────────────────────────────────────────────────────
# Main Stage 3 Runner
# ──────────────────────────────────────────────────────────────────

def run_stage3(
    triples: list[dict],
    recommendations: list[dict],
    entities: list[dict],
    match_results: list[dict],
    batch_size: int = None,
    max_workers: int = None,
) -> list[dict]:
    """Stage 3: Condition Augmentation.

    Args:
        triples: from stage2_umls_layer_triples.json
        recommendations: from stage0_recommendations.json
        entities: from stage1_entity_candidates.json
        match_results: from stage2_umls_matched.json (for CUI-based matching)
        batch_size: triples per LLM call (overrides STAGE3_LLM_CHUNK_SIZE)
        max_workers: parallel LLM workers (defaults to config.LLM_MAX_WORKERS)
    """
    llm_chunk_size = batch_size or config.STAGE3_LLM_CHUNK_SIZE
    max_workers = max_workers or config.LLM_MAX_WORKERS

    logger.info("=" * 60)
    logger.info("STAGE 3: Condition Augmentation")
    logger.info("=" * 60)
    logger.info(f"  Triples: {len(triples)}")
    logger.info(f"  Recommendations: {len(recommendations)}")
    logger.info(f"  Match results: {len(match_results)}")
    logger.info(f"  LLM chunk size: {llm_chunk_size}, workers: {max_workers}")

    rec_index = build_recommendation_index(recommendations, entities, match_results)

    # Step 1: Group triples by endpoint pair. Same (head, tail) → same recs →
    # one LLM call (conditions depend on rec text + endpoints, not relation).
    # Triples without a head+tail rec match get empty conditions and skip LLM.
    pair_cache: dict = {}
    groups: dict = {}  # pair_key → list of triple indices needing LLM
    no_rec_count = 0
    for i, t in enumerate(triples):
        key = (t.get("head_cui", ""), t.get("tail_id", ""),
               t.get("head_name", "").lower().strip(),
               t.get("tail_name", "").lower().strip())
        if key not in pair_cache:
            pair_cache[key] = find_relevant_recommendations(t, recommendations, rec_index)
        if pair_cache[key]:
            groups.setdefault(key, []).append(i)
        else:
            _apply_no_conditions(t)
            no_rec_count += 1

    group_keys = list(groups.keys())
    group_originals = [groups[k] for k in group_keys]
    rep_triples = [triples[oi[0]] for oi in group_originals]
    rep_recs = [pair_cache[k] for k in group_keys]

    candidates = sum(len(oi) for oi in group_originals)
    logger.info(f"  Pair cache: {len(pair_cache)} unique endpoint pairs")
    logger.info(
        f"  LLM dedup: {candidates} candidate triples → "
        f"{len(rep_triples)} unique groups (saved {candidates - len(rep_triples)})"
    )

    # Step 2: Submit one LLM call per group, parallelized.
    rep_results: list[Optional[dict]] = [None] * len(rep_triples)
    if rep_triples:
        chunks = [
            (start, rep_triples[start:start + llm_chunk_size],
             rep_recs[start:start + llm_chunk_size])
            for start in range(0, len(rep_triples), llm_chunk_size)
        ]
        progress_every = max(1, len(chunks) // 50)
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(extract_conditions_batch, ct, cr): cs
                for cs, ct, cr in chunks
            }
            for fut in as_completed(futures):
                chunk_start = futures[fut]
                for r in fut.result():
                    if not isinstance(r, dict):
                        continue
                    local = r.get("triple_index", -1)
                    if isinstance(local, int):
                        abs_idx = chunk_start + local
                        if 0 <= abs_idx < len(rep_results):
                            rep_results[abs_idx] = r
                completed += 1
                if completed % progress_every == 0 or completed == len(chunks):
                    logger.info(f"  LLM progress: {completed}/{len(chunks)} chunks")

    # Step 3: Apply each group's LLM result to all triples sharing that pair.
    parse_failed = 0
    with_cond = 0
    total_cond = 0
    for cr, key, orig_indices in zip(rep_results, group_keys, group_originals):
        if cr is None:
            for i in orig_indices:
                _apply_parse_failed(triples[i])
            parse_failed += len(orig_indices)
            continue
        recs = pair_cache[key]
        for i in orig_indices:
            _apply_cr(triples[i], cr, recs)
            if triples[i]["has_conditions"]:
                with_cond += 1
                total_cond += len(triples[i]["conditions"])

    logger.info("=" * 60)
    logger.info("STAGE 3 COMPLETE")
    logger.info(
        f"  Triples with conditions: {with_cond}/{len(triples)} "
        f"({with_cond / max(len(triples), 1) * 100:.1f}%)"
    )
    logger.info(f"  Total conditions: {total_cond}")
    logger.info(f"  Triples skipped LLM (no head+tail rec match): {no_rec_count}")
    logger.info(f"  LLM calls saved by dedup: {candidates - len(rep_triples)} triples")
    if parse_failed:
        logger.warning(
            f"  Triples with parse failure (excluded from Neo4j): {parse_failed}"
        )
    logger.info("=" * 60)

    return triples
