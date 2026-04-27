"""
Stage 3: Condition Augmentation

Attaches structured conditions to UMLS layer triples by comparing
each triple against its source CREST guideline recommendation(s).

Design rationale (from research proposal §2.4 Step 3):
  "구축된 UMLS layer의 triple을 CREST guideline 문맥과 재비교해
   적용 가능한 condition을 metadata(list 형태)로 추가."

Matching strategy:
  Triple head/tail names are UMLS concept preferred names (e.g. "Carcinoma,
  Non-Small-Cell Lung"), while entity candidates use LLM-normalized forms
  (e.g. "Non-Small Cell Lung Carcinoma"). Direct string matching fails.
  
  Solution: CUI-based matching via stage2_umls_matched.json.
    triple.head_cui → match_results → entity.source_guidelines → recommendations

Condition types (4 only):
  - numeric_threshold, categorical_state, medication_history, temporal_condition

Output is Neo4j-compatible: conditions serialized as JSON string property.
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import config

logger = logging.getLogger(__name__)

_openai_client = None


def _get_openai_client(api_key: str = None):
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=api_key or config.OPENAI_API_KEY)
    return _openai_client


# ──────────────────────────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────────────────────────

CONDITION_SYSTEM_PROMPT = """You extract structured conditions from clinical guideline sentences and attach them to medical knowledge graph triples.

[CONDITION TYPES — use ONLY these 4]

1. numeric_threshold
   {"type":"numeric_threshold","variable":"<str>","comparator":"<str>","value":<num>,"unit":"<str>","evidence_text":"<str>"}
   Examples: age >= 65, eGFR < 30, HbA1c > 7%, BMI >= 30

2. categorical_state
   {"type":"categorical_state","variable":"<str>","value":"<str>","evidence_text":"<str>"}
   Examples: diagnosis=type_2_diabetes, risk=high, pregnancy=true, NYHA_class=III-IV

3. medication_history
   {"type":"medication_history","drug":"<str>","status":"<str>","dose":"<str or empty>","evidence_text":"<str>"}
   status: current | prior | failed | discontinued | naive

4. temporal_condition
   {"type":"temporal_condition","event":"<str>","anchor":"<str>","interval":<num or str>,"interval_unit":"<str>","comparator":"<str>","evidence_text":"<str>"}
   Examples: within 72h of onset, stable for >= 2 years

[RULES]
- Only attach conditions that CONSTRAIN the specific triple relation.
- Do NOT attach conditions irrelevant to the triple.
- Age ranges (e.g. "55–74 years") → two numeric_threshold conditions (>= and <=).
- Disease/drug/procedure/gene concepts are GRAPH NODES, not conditions.
- Patient states, thresholds, time windows, medication status are CONDITIONS.
- If no relevant condition exists, return empty conditions list.
- Specify condition_logic: "AND" (default), "OR", or "NOT".

[OUTPUT FORMAT]
Return a JSON array. For each input triple, output one object:
{
  "triple_index": <int>,
  "conditions": [ ... structured conditions ... ],
  "condition_logic": "AND"|"OR"|"NOT",
  "condition_source": {
    "guideline_id": "<str>",
    "evidence_level": "sentence_aligned"|"guideline_cooccurrence"|"inferred",
    "evidence_texts": ["<str>", ...]
  }
}

Respond ONLY with the JSON array. No markdown, no explanation."""


FEW_SHOT_USER = """[TRIPLES]
Triple 0:
  ("Lung Neoplasms", "screened_by", "Low-dose CT")

[RECOMMENDATION SENTENCES]
Rec for Triple 0:
  "Low-dose CT screening is recommended for adults aged 55 to 74 years with 30 pack-years or more smoking history who quit within the past 15 years."
  [guideline_id: guideline_042, strength: A]"""

FEW_SHOT_ASSISTANT = """[
  {
    "triple_index": 0,
    "conditions": [
      {"type":"numeric_threshold","variable":"age","comparator":">=","value":55,"unit":"years","evidence_text":"aged 55 to 74 years"},
      {"type":"numeric_threshold","variable":"age","comparator":"<=","value":74,"unit":"years","evidence_text":"aged 55 to 74 years"},
      {"type":"numeric_threshold","variable":"smoking_pack_year","comparator":">=","value":30,"unit":"pack-years","evidence_text":"30 pack-years or more"},
      {"type":"temporal_condition","event":"smoking_cessation","anchor":"presentation","interval":15,"interval_unit":"years","comparator":"<=","evidence_text":"quit within the past 15 years"}
    ],
    "condition_logic": "AND",
    "condition_source": {
      "guideline_id": "guideline_042",
      "evidence_level": "sentence_aligned",
      "evidence_texts": ["aged 55 to 74 years","30 pack-years or more","quit within the past 15 years"]
    }
  }
]"""


# ──────────────────────────────────────────────────────────────────
# CUI-based Recommendation Matching
# ──────────────────────────────────────────────────────────────────

def build_recommendation_index(
    recommendations: list[dict],
    entities: list[dict],
    match_results: list[dict],
) -> dict:
    """
    Build CUI → recommendation indices mapping.

    Chain: CUI → match_results → entity.source_guidelines → guideline → rec indices

    Also builds text-search fallback for triples whose tail_id is not a CUI.

    Returns dict with two sub-indices:
      {
        "cui_to_recs": { "C0012345": {0, 3, 7}, ... },
        "name_to_recs": { "metformin": {0, 3}, ... },
      }
    """
    # guideline_id → list of recommendation indices
    guideline_to_rec_idx = {}
    for i, rec in enumerate(recommendations):
        gid = rec.get("guideline_id", "")
        if gid:
            guideline_to_rec_idx.setdefault(gid, []).append(i)

    # CUI → set of recommendation indices (via match_results → entity → guidelines)
    cui_to_recs = {}
    for mr in match_results:
        if not mr.get("matched"):
            continue

        entity = mr.get("entity", {})
        source_guidelines = entity.get("source_guidelines", [])

        # Collect rec indices for this entity's guidelines
        rec_indices = set()
        for gid in source_guidelines:
            for idx in guideline_to_rec_idx.get(gid, []):
                rec_indices.add(idx)

        if not rec_indices:
            continue

        # Map each matched CUI to these rec indices
        for m in mr.get("matches", []):
            cui = m.get("cui", "")
            if cui.startswith("C"):
                existing = cui_to_recs.get(cui, set())
                cui_to_recs[cui] = existing | rec_indices

    # Name-based fallback: scan rec text for entity names.
    # Deduplicate names first — entities often share normalized_form, and the
    # substring check is the same regardless of which entity it came from.
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
        f"Built recommendation index: "
        f"{len(cui_to_recs)} CUIs mapped, "
        f"{len(name_to_recs)} name-based entries (fallback)"
    )

    return {
        "cui_to_recs": cui_to_recs,
        "name_to_recs": name_to_recs,
    }


def find_relevant_recommendations(
    triple: dict,
    recommendations: list[dict],
    rec_index: dict,
    max_recs: int = None,
) -> list[dict]:
    """
    Find CREST recommendations relevant to a triple using CUI-based matching.

    Lookup order:
      1) head_cui in cui_to_recs
      2) tail_id in cui_to_recs (if tail_id is a CUI)
      3) head_name / tail_name in name_to_recs (fallback)

    Prioritizes recommendations where BOTH head and tail CUIs map to the same rec.
    """
    max_recs = max_recs or config.STAGE3_MAX_RECS_PER_TRIPLE

    cui_index = rec_index.get("cui_to_recs", {})
    name_index = rec_index.get("name_to_recs", {})

    head_cui = triple.get("head_cui", "")
    tail_id = triple.get("tail_id", "")
    head_name = triple.get("head_name", "").lower().strip()
    tail_name = triple.get("tail_name", "").lower().strip()

    # CUI-based lookup
    head_recs = cui_index.get(head_cui, set())
    tail_recs = cui_index.get(tail_id, set()) if tail_id.startswith("C") else set()

    # Name-based fallback (for non-CUI tails or when CUI lookup yields nothing)
    if not head_recs:
        head_recs = name_index.get(head_name, set())
    if not tail_recs:
        tail_recs = name_index.get(tail_name, set())

    # Priority 1: recs matching BOTH head and tail
    both = head_recs & tail_recs
    # Priority 2: recs matching either
    either = head_recs | tail_recs

    selected = list(both)[:max_recs]
    if len(selected) < max_recs:
        remaining = [i for i in either if i not in both]
        selected.extend(remaining[: max_recs - len(selected)])

    return [recommendations[i] for i in selected]


# ──────────────────────────────────────────────────────────────────
# LLM Call
# ──────────────────────────────────────────────────────────────────

def _build_user_message(
    triples_batch: list[dict],
    batch_recommendations: list[list[dict]],
) -> str:
    """Build user message for a batch of triples with their recommendations."""
    parts = ["[TRIPLES]"]

    for i, triple in enumerate(triples_batch):
        parts.append(
            f'Triple {i}:\n'
            f'  ("{triple["head_name"]}", "{triple["relation"]}", "{triple["tail_name"]}")'
        )

    parts.append("\n[RECOMMENDATION SENTENCES]")

    for i, recs in enumerate(batch_recommendations):
        if recs:
            for rec in recs:
                gid = rec.get("guideline_id", "")
                strength = rec.get("strength", "")
                text = rec.get("text", "")
                parts.append(
                    f"Rec for Triple {i}:\n"
                    f'  "{text}"\n'
                    f"  [guideline_id: {gid}, strength: {strength}]"
                )
        else:
            parts.append(f"Rec for Triple {i}:\n  (no matching recommendation found)")

    return "\n".join(parts)


def _parse_condition_response(response_text: str) -> list[dict]:
    """Parse LLM JSON array response."""
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse condition response: {text[:200]}")
                return []
        else:
            logger.warning(f"No JSON array in condition response: {text[:200]}")
            return []

    if isinstance(data, dict):
        data = [data]

    return data if isinstance(data, list) else []


def extract_conditions_batch(
    triples_batch: list[dict],
    batch_recommendations: list[list[dict]],
    api_key: str = None,
    model: str = None,
) -> list[dict]:
    """Call LLM to extract conditions for a batch of triples."""
    client = _get_openai_client(api_key)
    model = model or config.LLM_MODEL

    user_message = _build_user_message(triples_batch, batch_recommendations)

    input_messages = [
        {"role": "user", "content": FEW_SHOT_USER},
        {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
        {"role": "user", "content": user_message},
    ]

    try:
        response = client.responses.create(
            model=model,
            instructions=CONDITION_SYSTEM_PROMPT,
            input=input_messages,
            temperature=0.0,
            max_output_tokens=config.LLM_MAX_TOKENS,
            store=False,
        )
        return _parse_condition_response(response.output_text)

    except Exception as e:
        logger.error(f"Condition extraction LLM call failed: {e}")
        return []


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
    required = REQUIRED_FIELDS.get(ctype, set())
    return all(cond.get(f) is not None for f in required)


def _normalize_conditions(conditions: list) -> list[dict]:
    if not isinstance(conditions, list):
        return []
    valid = []
    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        if _validate_condition(cond):
            cond.setdefault("evidence_text", "")
            valid.append(cond)
        else:
            logger.debug(f"Skipping invalid condition: {cond}")
    return valid


# ──────────────────────────────────────────────────────────────────
# Triple Augmentation
# ──────────────────────────────────────────────────────────────────

def apply_conditions_to_triples(
    triples_batch: list[dict],
    condition_results: list[dict],
    batch_recommendations: list[list[dict]],
) -> list[dict]:
    """
    Merge LLM-extracted conditions into original triples.

    Adds Neo4j-compatible fields:
      - conditions: list of validated condition dicts
      - conditions_json: JSON string (for Neo4j relationship property)
      - condition_logic: AND|OR|NOT
      - condition_source: {guideline_id, evidence_level, evidence_texts}
      - recommendation_strength: from matched recommendation
      - has_conditions: boolean
    """
    result_map = {}
    for cr in condition_results:
        idx = cr.get("triple_index", -1)
        result_map[idx] = cr

    augmented = []
    for i, triple in enumerate(triples_batch):
        cr = result_map.get(i, {})

        valid_conditions = _normalize_conditions(cr.get("conditions", []))
        source = cr.get("condition_source", {})

        # Get recommendation_strength from the matched recommendation
        recs = batch_recommendations[i] if i < len(batch_recommendations) else []
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
            json.dumps(valid_conditions, ensure_ascii=False)
            if valid_conditions else "[]"
        )
        triple["has_conditions"] = len(valid_conditions) > 0

        augmented.append(triple)

    return augmented


# ──────────────────────────────────────────────────────────────────
# Main Stage 3 Runner
# ──────────────────────────────────────────────────────────────────

_EMPTY_COND_SOURCE = {"guideline_id": "", "evidence_level": "", "evidence_texts": []}


def _mark_no_conditions(batch: list[dict]) -> list[dict]:
    """Set empty-condition fields on a batch of triples in place."""
    for triple in batch:
        triple["conditions"] = []
        triple["condition_logic"] = None
        triple["condition_source"] = dict(_EMPTY_COND_SOURCE)
        triple["recommendation_strength"] = None
        triple["conditions_json"] = "[]"
        triple["has_conditions"] = False
    return batch


def _process_batch(batch: list[dict], batch_recs: list[list[dict]]) -> tuple[list[dict], bool]:
    """Worker: extract + apply conditions for one batch. Returns (augmented, was_no_recs)."""
    if not any(batch_recs):
        return _mark_no_conditions(batch), True
    condition_results = extract_conditions_batch(batch, batch_recs)
    return apply_conditions_to_triples(batch, condition_results, batch_recs), False


def run_stage3(
    triples: list[dict],
    recommendations: list[dict],
    entities: list[dict],
    match_results: list[dict],
    batch_size: int = None,
    progress_interval: int = None,
    max_workers: int = None,
) -> list[dict]:
    """
    Stage 3: Condition Augmentation

    Args:
        triples: from stage2_umls_layer_triples.json
        recommendations: from stage0_recommendations.json
        entities: from stage1_entity_candidates.json
        match_results: from stage2_umls_matched.json (for CUI-based matching)
        batch_size: triples per LLM call
        progress_interval: log every N triples
        max_workers: parallel LLM batch workers (defaults to config.LLM_MAX_WORKERS)
    """
    batch_size = batch_size or config.STAGE3_BATCH_SIZE
    progress_interval = progress_interval or config.STAGE3_PROGRESS_INTERVAL
    max_workers = max_workers or config.LLM_MAX_WORKERS

    logger.info("=" * 60)
    logger.info("STAGE 3: Condition Augmentation")
    logger.info("=" * 60)
    logger.info(f"  Triples: {len(triples)}")
    logger.info(f"  Recommendations: {len(recommendations)}")
    logger.info(f"  Match results: {len(match_results)}")
    logger.info(f"  Batch size: {batch_size}, workers: {max_workers}")

    # Step 1: Build CUI-based recommendation index
    rec_index = build_recommendation_index(
        recommendations, entities, match_results,
    )

    # Step 2: Pre-slice batches and resolve recommendations.
    # Recommendation lookup is pure CPU and cheap, so we do it serially
    # before fanning out the LLM calls.
    batches: list[tuple[list[dict], list[list[dict]]]] = []
    for start in range(0, len(triples), batch_size):
        end = min(start + batch_size, len(triples))
        batch = triples[start:end]
        batch_recs = [
            find_relevant_recommendations(t, recommendations, rec_index)
            for t in batch
        ]
        batches.append((batch, batch_recs))

    # Step 3: Run LLM batch calls in parallel; preserve original order on output.
    augmented_per_batch: list[Optional[list[dict]]] = [None] * len(batches)
    no_recs_batches = 0
    total_no_recs = 0
    completed_triples = 0
    total_with_cond = 0
    total_cond = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_process_batch, b, br): bi
            for bi, (b, br) in enumerate(batches)
        }
        for fut in as_completed(futures):
            bi = futures[fut]
            augmented, was_no_recs = fut.result()
            augmented_per_batch[bi] = augmented
            completed_triples += len(augmented)
            if was_no_recs:
                no_recs_batches += 1
                total_no_recs += len(augmented)
            for t in augmented:
                if t.get("has_conditions"):
                    total_with_cond += 1
                    total_cond += len(t.get("conditions", []))
            if (completed_triples % progress_interval) < batch_size or completed_triples == len(triples):
                logger.info(
                    f"  Progress: {completed_triples}/{len(triples)} triples, "
                    f"{total_with_cond} with conditions, "
                    f"{total_cond} total conditions"
                )

    all_augmented: list[dict] = []
    for ab in augmented_per_batch:
        if ab:
            all_augmented.extend(ab)

    logger.info("=" * 60)
    logger.info("STAGE 3 COMPLETE")
    logger.info(f"  Triples with conditions: {total_with_cond}/{len(all_augmented)} "
                f"({total_with_cond / max(len(all_augmented), 1) * 100:.1f}%)")
    logger.info(f"  Total conditions: {total_cond}")
    logger.info(f"  Triples with no matching rec: {total_no_recs} "
                f"({no_recs_batches} batches skipped LLM)")
    logger.info("=" * 60)

    return all_augmented