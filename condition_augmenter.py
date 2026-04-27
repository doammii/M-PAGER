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

  Bidirectional filter: a triple is sent to the LLM only when head AND tail
  both resolve to a shared recommendation. Triples without such a rec are
  passed through with empty conditions (has_conditions=False) — kept in the
  output for completeness but never consume LLM tokens.

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

# UMLS CUI pattern: literal "C" + 7 digits. Tighter than startswith("C") to
# avoid mis-classifying source codes that happen to start with "C" (e.g.
# HCPCS "C1300") as CUIs.
_CUI_RE = re.compile(r"^C\d{7}$")


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

[BREVITY RULES — CRITICAL FOR TOKEN BUDGET]
- evidence_text MUST be a short key phrase (≤ 50 chars), NOT a full sentence quote.
- evidence_texts array: AT MOST 2 short phrases; deduplicate.
- Do NOT repeat the full guideline text or restate the triple.
- Output compact JSON: no extra whitespace, no trailing commentary.

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
    Find CREST recommendations that mention BOTH endpoints of the triple.

    Bidirectional filter: a rec only qualifies when head AND tail both
    resolve to it. Triples without any rec satisfying both endpoints get
    an empty list and are skipped by the LLM downstream.

    Lookup order per endpoint:
      - CUI-based: head_cui / tail_id (when tail_id starts with "C") in cui_to_recs
      - name-based fallback: head_name / tail_name in name_to_recs
    """
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

    # Sort for deterministic selection — set iteration order is hash-dependent
    # and would otherwise make recs[0] (used for recommendation_strength) flaky
    # across runs.
    selected = sorted(both)[:max_recs]
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
    """Parse LLM JSON array response.

    Handles truncated output (max_output_tokens hit) by salvaging the
    completed objects up to the last closed `}` and synthesizing the
    closing `]`.
    """
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
                data = _salvage_truncated_array(text)
                if data is None:
                    logger.warning(f"Failed to parse condition response: {text[:200]}")
                    return []
        else:
            data = _salvage_truncated_array(text)
            if data is None:
                logger.warning(f"No JSON array in condition response: {text[:200]}")
                return []

    if isinstance(data, dict):
        data = [data]

    return data if isinstance(data, list) else []


def _salvage_truncated_array(text: str) -> Optional[list]:
    """Recover whatever objects parse from a truncated JSON array.

    Two-tier strategy:
      1) Fast path — trim at the last `}` and synthesize `]`. Works when
         at least one top-level object closed cleanly before the cutoff.
      2) Bracket-balance — when no `}` exists or the fast path produces
         invalid JSON, walk the text with a JSON state machine, rewind
         to the last position where a value successfully completed, and
         synthesize all still-open `}`/`]`. Recovers partially-built
         objects whose closing braces never made it on the wire.
    """
    if not text or "[" not in text:
        return None

    # ── Tier 1: simple brace truncation ─────────────────────────────
    last_obj_end = text.rfind("}")
    if last_obj_end != -1:
        candidate = text[: last_obj_end + 1] + "]"
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                logger.warning(
                    f"Recovered {len(data)} completed object(s) from "
                    f"truncated response (likely hit max_output_tokens)"
                )
                return data
        except json.JSONDecodeError:
            pass

    # ── Tier 2: bracket-balance state machine ───────────────────────
    return _salvage_partial_json(text)


def _salvage_partial_json(text: str) -> Optional[list]:
    """State-machine salvage for deeply-truncated JSON arrays.

    Walks `text` char by char tracking:
      - bracket stack (`{` / `[`)
      - per-level JSON expectation: 'key' | 'colon' | 'val' | 'comma'
      - the position of the last fully-completed value

    Then trims to that position and synthesizes the brackets needed to
    close every still-open structure. Mid-key / mid-primitive truncations
    are correctly discarded because they never advance `last_safe`.
    """
    start = text.find("[")
    if start < 0:
        return None
    text = text[start:]

    stack: list[list] = []   # entries: [bracket_char, expectation]
    in_str = False
    escape = False
    last_safe = -1

    i, n = 0, len(text)
    while i < n:
        c = text[i]

        # Inside a string: just look for the closing quote (with escape).
        if in_str:
            if escape:
                escape = False
            elif c == '\\':
                escape = True
            elif c == '"':
                in_str = False
                if stack:
                    if stack[-1][1] == 'val':
                        stack[-1][1] = 'comma'
                        last_safe = i
                    elif stack[-1][1] == 'key':
                        stack[-1][1] = 'colon'
            i += 1
            continue

        if c.isspace():
            i += 1
            continue

        if c == '"':
            in_str = True
            i += 1
            continue

        if c == '{':
            stack.append(['{', 'key'])
            i += 1
            continue

        if c == '[':
            stack.append(['[', 'val'])
            i += 1
            continue

        if c == '}':
            if not stack or stack[-1][0] != '{':
                break
            stack.pop()
            if stack:
                stack[-1][1] = 'comma'
            last_safe = i
            i += 1
            continue

        if c == ']':
            if not stack or stack[-1][0] != '[':
                break
            stack.pop()
            if stack:
                stack[-1][1] = 'comma'
            last_safe = i
            i += 1
            if not stack:
                break
            continue

        if c == ':':
            if stack and stack[-1][1] == 'colon':
                stack[-1][1] = 'val'
            i += 1
            continue

        if c == ',':
            if stack and stack[-1][1] == 'comma':
                stack[-1][1] = 'key' if stack[-1][0] == '{' else 'val'
            i += 1
            continue

        # Primitive (number / true / false / null)
        if c.isdigit() or c in '-tfn':
            sv = i
            while i < n and text[i] not in ',}]' and not text[i].isspace():
                i += 1
            if i == n:
                # Ran off the end mid-primitive — value is unreliable.
                break
            primitive = text[sv:i]
            try:
                json.loads(primitive)
                if stack and stack[-1][1] == 'val':
                    stack[-1][1] = 'comma'
                    last_safe = i - 1
            except json.JSONDecodeError:
                pass
            continue

        # Unknown char — skip defensively rather than abort.
        i += 1

    if last_safe < 0:
        return None

    trimmed = text[: last_safe + 1]

    # Recompute open-bracket pending list against the trimmed text.
    pending: list[str] = []
    in_str = False
    escape = False
    for ch in trimmed:
        if escape:
            escape = False
            continue
        if in_str:
            if ch == '\\':
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in '{[':
            pending.append(ch)
        elif ch in '}]':
            if pending:
                pending.pop()

    closing = {'{': '}', '[': ']'}
    salvaged = trimmed + ''.join(closing[ch] for ch in reversed(pending))

    try:
        data = json.loads(salvaged)
    except json.JSONDecodeError:
        return None

    if isinstance(data, list):
        logger.warning(
            f"Aggressive recovery: salvaged {len(data)} object(s) from "
            f"deeply-truncated response (no closing brace existed)"
        )
        return data
    if isinstance(data, dict):
        return [data]
    return None


_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def extract_conditions_batch(
    triples_batch: list[dict],
    batch_recommendations: list[list[dict]],
    api_key: str = None,
    model: str = None,
    _retry_depth: int = 0,
) -> list[dict]:
    """Call LLM to extract conditions for a batch of triples.

    Cost-optimized truncation handling: when the response is cut off
    (max_output_tokens hit), parse what came through, identify which
    triple_indices are missing, and re-call with ONLY those triples
    (not the whole batch). One retry max — bounded extra cost.
    """
    client = _get_openai_client(api_key)
    model = model or config.LLM_MODEL

    user_message = _build_user_message(triples_batch, batch_recommendations)

    input_messages = [
        {"role": "user", "content": FEW_SHOT_USER},
        {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
        {"role": "user", "content": user_message},
    ]

    kwargs = dict(
        model=model,
        instructions=CONDITION_SYSTEM_PROMPT,
        input=input_messages,
        temperature=0.0,
        max_output_tokens=config.LLM_MAX_TOKENS,
        store=False,
    )
    # Reasoning models burn tokens on internal CoT before emitting output.
    # "minimal" effort is enough for this structured-extraction task and
    # leaves the bulk of the budget for the JSON body.
    if any(model.startswith(p) for p in _REASONING_MODEL_PREFIXES):
        kwargs["reasoning"] = {"effort": "minimal"}

    try:
        response = client.responses.create(**kwargs)
    except Exception as e:
        logger.error(f"Condition extraction LLM call failed: {e}")
        return []

    data = _parse_condition_response(response.output_text)

    if _retry_depth >= 1:
        return data

    # Identify triples that didn't make it into the parsed result
    # (truncated mid-output, or skipped by the model).
    expected = set(range(len(triples_batch)))
    seen = {r.get("triple_index", -1) for r in data if isinstance(r, dict)}
    missing = sorted(expected - seen)
    if not missing:
        return data

    logger.warning(
        f"Retrying {len(missing)}/{len(triples_batch)} missing triple(s)"
    )
    sub_triples = [triples_batch[i] for i in missing]
    sub_recs = [batch_recommendations[i] for i in missing]
    retry_data = extract_conditions_batch(
        sub_triples, sub_recs,
        api_key=api_key, model=model, _retry_depth=1,
    )
    # Map local indices in retry response back to parent batch positions.
    for r in retry_data:
        if not isinstance(r, dict):
            continue
        local_idx = r.get("triple_index", -1)
        if isinstance(local_idx, int) and 0 <= local_idx < len(missing):
            r["triple_index"] = missing[local_idx]
            data.append(r)

    return data


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
      - parse_failed: True when the LLM result for this triple was lost
        (truncation/parse error after retry). Stage 4 (Neo4j) excludes
        these so the graph never contains uncertain edges.
    """
    result_map = {}
    for cr in condition_results:
        idx = cr.get("triple_index", -1)
        result_map[idx] = cr

    augmented = []
    for i, triple in enumerate(triples_batch):
        cr = result_map.get(i)

        if cr is None:
            # LLM result missing for this triple — parse failed and the
            # one allowed retry also did not return it. Distinguish from
            # legitimate "no conditions" (empty cr.conditions) by flagging.
            triple["conditions"] = []
            triple["condition_logic"] = None
            triple["condition_source"] = dict(_EMPTY_COND_SOURCE)
            triple["recommendation_strength"] = None
            triple["conditions_json"] = "[]"
            triple["has_conditions"] = False
            triple["parse_failed"] = True
            augmented.append(triple)
            continue

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
        triple["parse_failed"] = False

        augmented.append(triple)

    return augmented


# ──────────────────────────────────────────────────────────────────
# Main Stage 3 Runner
# ──────────────────────────────────────────────────────────────────

_EMPTY_COND_SOURCE = {"guideline_id": "", "evidence_level": "", "evidence_texts": []}


def _mark_no_conditions(batch: list[dict]) -> list[dict]:
    """Set empty-condition fields on a batch of triples in place.

    Used for triples that didn't pass the head-tail bidirectional filter.
    These are intentional empty-condition triples (NOT parse failures),
    so parse_failed is set False — Stage 4 keeps them in the graph.
    """
    for triple in batch:
        triple["conditions"] = []
        triple["condition_logic"] = None
        triple["condition_source"] = dict(_EMPTY_COND_SOURCE)
        triple["recommendation_strength"] = None
        triple["conditions_json"] = "[]"
        triple["has_conditions"] = False
        triple["parse_failed"] = False
    return batch


def _process_batch(batch: list[dict], batch_recs: list[list[dict]]) -> tuple[list[dict], int]:
    """
    Worker: extract + apply conditions for one batch.

    With bidirectional matching, batches commonly mix triples-with-recs and
    triples-without. Only the former are sent to the LLM; the rest are
    marked no_conditions in place to save tokens.

    Returns (augmented_batch, no_rec_count_in_batch).
    """
    sub_batch = [t for t, recs in zip(batch, batch_recs) if recs]
    sub_recs = [recs for recs in batch_recs if recs]
    no_rec_triples = [t for t, recs in zip(batch, batch_recs) if not recs]

    _mark_no_conditions(no_rec_triples)

    if sub_batch:
        condition_results = extract_conditions_batch(sub_batch, sub_recs)
        apply_conditions_to_triples(sub_batch, condition_results, sub_recs)

    return batch, len(no_rec_triples)


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
    total_parse_failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_process_batch, b, br): bi
            for bi, (b, br) in enumerate(batches)
        }
        for fut in as_completed(futures):
            bi = futures[fut]
            augmented, batch_no_recs = fut.result()
            augmented_per_batch[bi] = augmented
            completed_triples += len(augmented)
            total_no_recs += batch_no_recs
            if batch_no_recs == len(augmented):
                no_recs_batches += 1
            for t in augmented:
                if t.get("has_conditions"):
                    total_with_cond += 1
                    total_cond += len(t.get("conditions", []))
                if t.get("parse_failed"):
                    total_parse_failed += 1
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
    logger.info(f"  Triples skipped LLM (no head+tail rec match): "
                f"{total_no_recs}/{len(all_augmented)} "
                f"({no_recs_batches} batches fully skipped)")
    if total_parse_failed:
        logger.warning(
            f"  Triples with parse failure (will be excluded from Neo4j): "
            f"{total_parse_failed}/{len(all_augmented)}"
        )
    logger.info("=" * 60)

    return all_augmented