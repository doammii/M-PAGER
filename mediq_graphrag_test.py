"""
MediQ GraphRAG Retrieval Test (with full-flow trace log)

Pipeline (per sampled MediQ row):

  1. Load deterministic 10-row sample from output/mediq_sample.json
     (run sample_mediq.py once to produce this file).
  2. Build input_query = "[CONTEXT]\\n{joined_context}\\n\\n[QUESTION]\\n{question}"
     (no extra prompt; the model only sees context above + question below).
  3. Sentence-split input_query, then per sentence:
        - extract entities  →  entity_extractor.call_openai
                                (same prompt as Stage 1 KG build)
        - match entities    →  entity_matcher.EntityMatcher
                                (same UMLS cascade as Stage 2-A)
        - extract conditions → condition_augmenter.extract_conditions_batch
                                (same prompt as Stage 3)
     Aggregate the entity-CUI list and condition list across sentences.
  4. Retrieve from Neo4j using priority order:
        3-1  entity CUIs match in KG AND any extracted-condition keyword
             appears in r.conditions_json on those edges
        3-2  entity CUIs match but no condition match → 1-hop subgraph on
             the matched CUIs
        3-3  no entity CUIs match → up to 5 distinct similar conditions,
             1 sample triple each
  5. Build final prompt: retrieval_result + options + question. Call the
     same LLM (gpt-5.4-mini, low reasoning effort). Ask for the option
     TEXT (not letter), parse, and compare to row["answer"].
  6. No-RAG baseline: same sample, same model, but the prompt contains only
     [QUESTION] + [OPTIONS] (no retrieval block). Lets us check whether the
     KG retrieval actually contributes signal vs. just adding noise.

Outputs:
  - output/mediq_graphrag_results.json : structured per-sample results
  - output/mediq_graphrag_log.txt      : full-flow human-readable trace
                                         (input_query, entities, conditions,
                                          Cypher params, retrieved triples,
                                          final prompt, LLM response, …)

Run:
    python sample_mediq.py            # once
    python mediq_graphrag_test.py     # repeatable
"""

import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Optional, TextIO

from neo4j import GraphDatabase

import condition_augmenter
import config
import entity_extractor
from cli_utils import setup_logging
from entity_matcher import EntityMatcher
from semantic_types import load_semantic_groups_from_file
from umls_client import UMLSClient

logger = logging.getLogger(__name__)

SAMPLE_PATH = os.path.join(config.OUTPUT_DIR, "mediq_sample.json")
RESULTS_PATH = os.path.join(config.OUTPUT_DIR, "mediq_graphrag_results.json")
TRACE_LOG_PATH = os.path.join(config.OUTPUT_DIR, "mediq_graphrag_log.txt")

# ── Retrieval limits ──
RETRIEVAL_LIMIT_31 = 50
RETRIEVAL_LIMIT_32 = 100
RETRIEVAL_LIMIT_33 = 5
COND_KEYWORD_MIN_LEN = 4   # drop very short tokens to reduce false matches
COND_KEYWORD_MAX = 30      # cap how many keywords we send to Cypher

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


# ──────────────────────────────────────────────────────────────────────
# Trace log helper
# ──────────────────────────────────────────────────────────────────────

class TraceLog:
    """Plain-text trace logger that mirrors every step of the RAG flow."""

    def __init__(self, path: str):
        self.path = path
        self._fh: Optional[TextIO] = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fh:
            self._fh.close()
            self._fh = None

    def write(self, text: str = ""):
        self._fh.write(text + "\n")
        self._fh.flush()

    def header(self, title: str, char: str = "="):
        self.write(char * 78)
        self.write(title)
        self.write(char * 78)

    def section(self, title: str):
        self.write("")
        self.write(f"── {title} " + "─" * max(2, 70 - len(title)))

    def kv(self, key: str, value):
        self.write(f"{key}: {value}")

    def block(self, label: str, body: str):
        self.write(f"[{label}]")
        for line in (body or "").splitlines() or [""]:
            self.write(f"  {line}")

    def jsonblock(self, label: str, obj):
        self.write(f"[{label}]")
        try:
            text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
        except Exception:
            text = repr(obj)
        for line in text.splitlines():
            self.write(f"  {line}")


# ──────────────────────────────────────────────────────────────────────
# Step 2: build input_query
# ──────────────────────────────────────────────────────────────────────

def build_input_query(sample: dict) -> str:
    """Concatenate context (above) + question (below). No extra instructions."""
    ctx = sample.get("context", [])
    if isinstance(ctx, list):
        ctx_text = " ".join(s.strip() for s in ctx if s and s.strip())
    else:
        ctx_text = str(ctx).strip()
    question = (sample.get("question") or "").strip()
    return f"[CONTEXT]\n{ctx_text}\n\n[QUESTION]\n{question}"


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence split that ignores section headers like [CONTEXT]."""
    cleaned = re.sub(r"\[(CONTEXT|QUESTION)\]\s*", "", text)
    parts = _SENTENCE_SPLIT_RE.split(cleaned.strip())
    return [p.strip() for p in parts if p and p.strip()]


# ──────────────────────────────────────────────────────────────────────
# Step 3a: entity extraction & UMLS matching
# ──────────────────────────────────────────────────────────────────────

def extract_entities_per_sentence(sentences: list[str]) -> list[list[dict]]:
    """Reuse Stage-1 prompt by wrapping each sentence as a recommendation.

    Returns a list-of-lists (per-sentence). The caller dedups across the
    whole input_query for matching/retrieval.
    """
    per_sent: list[list[dict]] = []
    for s in sentences:
        rec = {"text": s, "guideline_id": "", "strength": ""}
        try:
            ents = entity_extractor.call_openai(rec)
        except Exception as e:
            logger.warning(f"entity LLM failed on sentence: {e}")
            ents = []
        per_sent.append(ents or [])
    return per_sent


def dedup_entities(per_sent: list[list[dict]]) -> list[dict]:
    seen, out = set(), []
    for ents in per_sent:
        for ent in ents:
            key = (
                ent.get("normalized_form", ent.get("surface_form", ""))
                .lower().strip()
            )
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(ent)
    return out


def match_entities_to_umls(
    entities: list[dict],
    umls_client: UMLSClient,
    tui_to_group: dict,
) -> tuple[list[dict], list[str]]:
    """Same cascading UMLS match used in Stage 2-A. Returns
    (match_results, deduped_cui_list)."""
    matcher = EntityMatcher(umls_client, tui_to_group)
    results = []
    cuis: list[str] = []
    seen_cuis: set[str] = set()
    for ent in entities:
        try:
            r = matcher.match_entity(ent)
        except Exception as e:
            logger.warning(f"UMLS match failed for {ent.get('surface_form')}: {e}")
            continue
        results.append(r)
        if r.get("matched"):
            for m in r.get("matches", []):
                cui = m.get("cui", "")
                if cui.startswith("C") and cui not in seen_cuis:
                    seen_cuis.add(cui)
                    cuis.append(cui)
    return results, cuis


# ──────────────────────────────────────────────────────────────────────
# Step 3b: condition extraction (query-time prompt, no triple anchor)
# ──────────────────────────────────────────────────────────────────────

# The Stage-3 prompt is anchored on a triple and gates extraction by "only
# conditions that constrain the triple relation" — when run against patient
# case sentences with a dummy triple, this empirically returned 0 conditions
# across all sentences. The query-time prompt below drops the triple anchor
# and asks the model to extract eligibility/state/temporal conditions that
# describe the patient/case directly. The 4 condition-type definitions and
# the per-condition JSON schema are reused verbatim from Stage 3; only the
# wrapper switches from `triple_index` to `sentence_index`.

QUERY_CONDITION_SYSTEM_PROMPT = """Extract structured conditions describing the patient or case from each clinical sentence.

[CONDITION TYPES — pick exactly one per condition]
- numeric_threshold: numeric cutoffs / ranges / labs / age. Required: variable, comparator, value. Optional: unit.
- categorical_state: diagnosis / state / risk / patient population (e.g. "pregnant", "type 2 diabetes"). Required: variable, value.
- medication_history: current / prior drug exposure or failure. Required: drug, status. Optional: dose.
- temporal_condition: time windows / follow-up / onset timing (e.g. "within 72h", "stable for 2 years"). Required: event, anchor, comparator. Optional: interval, interval_unit.

[RULES]
- Extract any eligibility / state / temporal condition the sentence asserts about the patient or the case. No triple anchor is required.
- Age ranges split into two numeric_threshold conditions (e.g. "55-74 years" → age>=55 AND age<=74).
- Each condition must include evidence_text (≤50 chars, verbatim phrase from the sentence).

[OUTPUT]
Return a JSON object {"results":[...]} with EXACTLY one entry per input sentence_index (0..N-1). If a sentence has no applicable condition, still include its entry with conditions:[]."""

QUERY_FEW_SHOT_USER = """[SENTENCES]
Sentence 0: "Screening recommended for adults aged 55-74, 30+ pack-years, quit within 15 years."
Sentence 1: "Patient is otherwise healthy."
Sentence 2: "She started taking lisinopril several weeks ago."
"""

QUERY_FEW_SHOT_ASSISTANT = """{"results":[{"sentence_index":0,"conditions":[{"type":"numeric_threshold","variable":"age","comparator":">=","value":55,"unit":"years","evidence_text":"aged 55-74"},{"type":"numeric_threshold","variable":"age","comparator":"<=","value":74,"unit":"years","evidence_text":"aged 55-74"},{"type":"numeric_threshold","variable":"smoking_pack_year","comparator":">=","value":30,"unit":"pack-years","evidence_text":"30+ pack-years"},{"type":"temporal_condition","event":"smoking_cessation","anchor":"presentation","interval":15,"interval_unit":"years","comparator":"<=","evidence_text":"quit within 15 years"}]},{"sentence_index":1,"conditions":[]},{"sentence_index":2,"conditions":[{"type":"medication_history","drug":"lisinopril","status":"current","evidence_text":"started taking lisinopril"},{"type":"temporal_condition","event":"lisinopril_initiation","anchor":"presentation","interval":2,"interval_unit":"weeks","comparator":"~","evidence_text":"several weeks ago"}]}]}"""


def extract_conditions_per_sentence(sentences: list[str]) -> list[list[dict]]:
    """Query-time condition extraction (no triple anchor).

    Reuses the 4 condition-type definitions and the per-condition JSON
    schema from Stage 3, but rewrites the system prompt to extract
    patient-describing conditions directly from each sentence. Required
    because the Stage-3 prompt empirically returned 0 conditions on
    case-description sentences when given a dummy triple.
    """
    if not sentences:
        return []

    user_lines = ["[SENTENCES]"]
    for i, s in enumerate(sentences):
        user_lines.append(f'Sentence {i}: "{s}"')
    user_msg = "\n".join(user_lines)

    client = condition_augmenter._get_openai_client()
    try:
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": QUERY_CONDITION_SYSTEM_PROMPT},
                {"role": "user", "content": QUERY_FEW_SHOT_USER},
                {"role": "assistant", "content": QUERY_FEW_SHOT_ASSISTANT},
                {"role": "user", "content": user_msg},
            ],
            reasoning_effort=config.LLM_REASONING_EFFORT,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"condition LLM failed: {e}")
        return [[] for _ in sentences]

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse query-time condition JSON: {text[:200]}")
        return [[] for _ in sentences]

    results = data.get("results", []) if isinstance(data, dict) else []
    per_sent: list[list[dict]] = [[] for _ in sentences]
    for r in results:
        if not isinstance(r, dict):
            continue
        idx = r.get("sentence_index")
        if isinstance(idx, int) and 0 <= idx < len(sentences):
            for cond in r.get("conditions", []) or []:
                if isinstance(cond, dict):
                    per_sent[idx].append(cond)
    return per_sent


def conditions_to_keywords(conditions: list[dict]) -> list[str]:
    """Pull substring-matchable keywords from extracted conditions.

    The KG stores conditions as a JSON string in r.conditions_json, so we use
    Cypher CONTAINS for cheap fuzzy matching. We pull the most distinctive
    fields: variable / value / drug / event / evidence_text.
    """
    seen: set[str] = set()
    out: list[str] = []
    fields = ("variable", "value", "drug", "event", "anchor", "evidence_text")
    for c in conditions:
        for f in fields:
            v = c.get(f)
            if v is None:
                continue
            s = str(v).strip().lower()
            if len(s) < COND_KEYWORD_MIN_LEN:
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= COND_KEYWORD_MAX:
                return out
    return out


# ──────────────────────────────────────────────────────────────────────
# Step 4: Neo4j retrieval (3-1 → 3-2 → 3-3)
# ──────────────────────────────────────────────────────────────────────

CYPHER_CUI_EXISTS = """
MATCH (c:Concept) WHERE c.id IN $cuis RETURN c.id AS id
"""

# 3-1: entities present AND at least one extracted-condition keyword
# appears in conditions_json on edges incident to those entities.
CYPHER_31 = """
MATCH (h:Concept)-[r:RELATES]->(t:Concept)
WHERE (h.id IN $cuis OR t.id IN $cuis)
  AND r.has_conditions = true
  AND any(kw IN $kws WHERE toLower(r.conditions_json) CONTAINS kw)
RETURN h.name AS head, h.id AS head_id,
       r.relation AS relation, r.guideline_id AS guideline_id,
       r.conditions_json AS conditions_json,
       r.recommendation_strength AS strength,
       t.name AS tail, t.id AS tail_id
LIMIT $limit
"""

# 3-2: entity-CUI 1-hop subgraph (no condition match required).
CYPHER_32 = """
MATCH (h:Concept)-[r:RELATES]->(t:Concept)
WHERE h.id IN $cuis OR t.id IN $cuis
RETURN h.name AS head, h.id AS head_id,
       r.relation AS relation, r.guideline_id AS guideline_id,
       r.conditions_json AS conditions_json,
       r.has_conditions AS has_conditions,
       r.recommendation_strength AS strength,
       t.name AS tail, t.id AS tail_id
LIMIT $limit
"""

# 3-3: no entity match — surface up to N distinct similar conditions, with
# 1 sample triple per condition.
CYPHER_33 = """
MATCH (h:Concept)-[r:RELATES]->(t:Concept)
WHERE r.has_conditions = true
  AND any(kw IN $kws WHERE toLower(r.conditions_json) CONTAINS kw)
WITH r.conditions_json AS conds,
     collect({head: h.name, relation: r.relation, tail: t.name,
              guideline_id: r.guideline_id, strength: r.recommendation_strength})[0]
       AS sample
RETURN conds AS conditions_json, sample
LIMIT $limit
"""


class Neo4jRetriever:
    def __init__(self, uri: str, user: str, password: str, database: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def __enter__(self):
        self.driver.verify_connectivity()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.driver.close()

    def _run(self, cypher: str, **params) -> list[dict]:
        with self.driver.session(database=self.database) as s:
            return [dict(r) for r in s.run(cypher, **params)]

    def cuis_present(self, cuis: list[str]) -> list[str]:
        if not cuis:
            return []
        rows = self._run(CYPHER_CUI_EXISTS, cuis=cuis)
        return [r["id"] for r in rows]

    def retrieve(
        self,
        cuis: list[str],
        cond_keywords: list[str],
        trace: Optional[TraceLog] = None,
    ) -> tuple[list[dict], str, dict]:
        """Returns (rows, strategy, debug). The `debug` dict records which
        Cypher queries were attempted and how many rows each returned, so
        the trace log can show the decision path."""
        debug = {"attempts": [], "present_cuis": []}
        present = self.cuis_present(cuis) if cuis else []
        debug["present_cuis"] = present

        if trace:
            trace.kv("CUIs found in KG", f"{len(present)}/{len(cuis)} → {present}")

        # 3-1
        if present and cond_keywords:
            params = {"cuis": present, "kws": cond_keywords,
                      "limit": RETRIEVAL_LIMIT_31}
            rows = self._run(CYPHER_31, **params)
            debug["attempts"].append(
                {"strategy": "3-1", "cypher": CYPHER_31.strip(),
                 "params": params, "rows_returned": len(rows)}
            )
            if trace:
                trace.section("Retrieval attempt: 3-1 (entity + condition)")
                trace.block("Cypher", CYPHER_31.strip())
                trace.jsonblock("Params", params)
                trace.kv("Rows returned", len(rows))
            if rows:
                return rows, "3-1_entity+condition", debug

        # 3-2
        if present:
            params = {"cuis": present, "limit": RETRIEVAL_LIMIT_32}
            rows = self._run(CYPHER_32, **params)
            debug["attempts"].append(
                {"strategy": "3-2", "cypher": CYPHER_32.strip(),
                 "params": params, "rows_returned": len(rows)}
            )
            if trace:
                trace.section("Retrieval attempt: 3-2 (entity 1-hop)")
                trace.block("Cypher", CYPHER_32.strip())
                trace.jsonblock("Params", params)
                trace.kv("Rows returned", len(rows))
            if rows:
                return rows, "3-2_entity_1hop", debug

        # 3-3
        if cond_keywords:
            params = {"kws": cond_keywords, "limit": RETRIEVAL_LIMIT_33}
            rows = self._run(CYPHER_33, **params)
            debug["attempts"].append(
                {"strategy": "3-3", "cypher": CYPHER_33.strip(),
                 "params": params, "rows_returned": len(rows)}
            )
            if trace:
                trace.section("Retrieval attempt: 3-3 (similar conditions)")
                trace.block("Cypher", CYPHER_33.strip())
                trace.jsonblock("Params", params)
                trace.kv("Rows returned", len(rows))
            if rows:
                return rows, "3-3_similar_conditions", debug

        return [], "empty", debug


def analyze_cui_coverage(
    rows: list[dict],
    queried_cuis: list[str],
    cui_to_name: dict,
    strategy: str,
) -> dict:
    """How many of the retrieved rows contain each queried CUI as head/tail.

    Quantifies whether a single high-degree CUI is saturating the LIMIT.
    Returns:
      - per_cui: list sorted desc by row count, with cui/name/row_count
      - dominant_cui_share: row_count of top CUI / total rows
      - zero_coverage_count: queried CUIs that appeared in 0 rows
      - distinct_head_cuis / distinct_tail_cuis: diversity in the rows
    Strategy 3-3 returns no head_id/tail_id, so per_cui is empty for it.
    """
    if not rows or strategy == "3-3_similar_conditions":
        return {
            "per_cui": [],
            "dominant_cui_share": 0.0,
            "zero_coverage_count": len(queried_cuis),
            "distinct_head_cuis": 0,
            "distinct_tail_cuis": 0,
            "total_rows": len(rows),
        }

    counts = {cui: 0 for cui in queried_cuis}
    head_set: set = set()
    tail_set: set = set()
    for r in rows:
        h = r.get("head_id")
        t = r.get("tail_id")
        if h:
            head_set.add(h)
        if t:
            tail_set.add(t)
        if h in counts:
            counts[h] += 1
        if t in counts:
            counts[t] += 1

    per_cui = [
        {"cui": cui, "name": cui_to_name.get(cui, ""), "row_count": n}
        for cui, n in counts.items()
    ]
    per_cui.sort(key=lambda x: x["row_count"], reverse=True)

    top_count = per_cui[0]["row_count"] if per_cui else 0
    return {
        "per_cui": per_cui,
        "dominant_cui_share": (top_count / len(rows)) if rows else 0.0,
        "zero_coverage_count": sum(1 for c in per_cui if c["row_count"] == 0),
        "distinct_head_cuis": len(head_set),
        "distinct_tail_cuis": len(tail_set),
        "total_rows": len(rows),
    }


def format_retrieval_result(rows: list[dict], strategy: str) -> str:
    """Render rows as compact text for the final-answer prompt."""
    if not rows:
        return "(no relevant knowledge graph triples retrieved)"

    lines = [f"# Retrieval strategy: {strategy}"]

    if strategy == "3-3_similar_conditions":
        for i, r in enumerate(rows, 1):
            sample = r.get("sample") or {}
            cj = r.get("conditions_json", "")
            lines.append(
                f"{i}. ({sample.get('head','')}) -[{sample.get('relation','')}]-> "
                f"({sample.get('tail','')})  | conditions: {cj}"
                f"  | guideline: {sample.get('guideline_id','')}"
                f"  | strength: {sample.get('strength','')}"
            )
        return "\n".join(lines)

    for i, r in enumerate(rows, 1):
        cj = r.get("conditions_json", "[]") or "[]"
        cond_part = f"  | conditions: {cj}" if cj and cj != "[]" else ""
        lines.append(
            f"{i}. ({r.get('head','')}) -[{r.get('relation','')}]-> "
            f"({r.get('tail','')})"
            f"  | guideline: {r.get('guideline_id','')}"
            f"  | strength: {r.get('strength','')}"
            f"{cond_part}"
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Step 4-LLM: final answer
# ──────────────────────────────────────────────────────────────────────

ANSWER_SYSTEM_PROMPT = (
    "You are answering a medical multiple-choice question. "
    "Use the retrieved knowledge-graph triples below as supporting evidence "
    "(they may be partially relevant). Pick exactly one option and output "
    "ONLY the option text (not the letter), as a JSON object: "
    '{"answer": "<the option text>"}.'
)

# No-RAG baseline: the system prompt does not reference any retrieval block,
# so the model must answer from its parametric knowledge alone.
BASELINE_SYSTEM_PROMPT = (
    "You are answering a medical multiple-choice question. "
    "Pick exactly one option and output ONLY the option text (not the letter), "
    'as a JSON object: {"answer": "<the option text>"}.'
)


def build_answer_user_prompt(
    retrieval_text: str, options: dict, question: str,
) -> str:
    options_block = "\n".join(f"{k}. {v}" for k, v in options.items())
    return (
        f"[RETRIEVAL]\n{retrieval_text}\n\n"
        f"[QUESTION]\n{question}\n\n"
        f"[OPTIONS]\n{options_block}"
    )


def build_baseline_user_prompt(options: dict, question: str) -> str:
    """Baseline prompt: question + options only, no retrieval block."""
    options_block = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"[QUESTION]\n{question}\n\n[OPTIONS]\n{options_block}"


def call_answer_llm(
    user_msg: str,
    system_prompt: Optional[str] = None,
) -> tuple[str, str]:
    """Returns (raw_response_text, parsed_answer_text).

    `user_msg` is the fully-formed prompt body. `system_prompt` defaults to
    ANSWER_SYSTEM_PROMPT (the RAG path); pass BASELINE_SYSTEM_PROMPT for the
    No-RAG baseline call.
    """
    if system_prompt is None:
        system_prompt = ANSWER_SYSTEM_PROMPT
    client = condition_augmenter._get_openai_client()
    try:
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            reasoning_effort=config.LLM_REASONING_EFFORT,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"final-answer LLM call failed: {e}")
        return "", ""

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    parsed = ""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            parsed = str(obj.get("answer", "")).strip()
    except json.JSONDecodeError:
        parsed = ""
    return raw, parsed


def parse_answer_with_fallback(
    raw: str, parsed: str, options: dict,
) -> str:
    """If JSON parse failed, fall back to substring matching across options."""
    if parsed:
        return parsed
    if not raw:
        return ""
    for v in options.values():
        if v and v.lower() in raw.lower():
            return v
    return ""


def normalize_for_compare(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# ──────────────────────────────────────────────────────────────────────
# Per-sample driver
# ──────────────────────────────────────────────────────────────────────

def run_one(
    sample: dict,
    umls_client: UMLSClient,
    tui_to_group: dict,
    retriever: Neo4jRetriever,
    trace: TraceLog,
) -> dict:
    sid = sample.get("id")
    logger.info(f"── sample id={sid} ──")
    trace.header(f"SAMPLE id={sid}", char="=")
    trace.kv("question", sample.get("question", ""))
    trace.jsonblock("context", sample.get("context", []))
    trace.jsonblock("options", sample.get("options", {}))
    trace.kv("gold_answer", sample.get("answer", ""))
    trace.kv("gold_answer_idx", sample.get("answer_idx", ""))

    # ── Step 2: input_query ──────────────────────────────────────
    input_query = build_input_query(sample)
    sentences = split_sentences(input_query)
    trace.section("STEP 2 — input_query")
    trace.block("input_query (verbatim, sent to LLM as the unit for steps 3-4)",
                input_query)
    trace.section("STEP 3 — sentence split")
    trace.kv("n_sentences", len(sentences))
    for i, s in enumerate(sentences):
        trace.write(f"  S{i}: {s}")

    # ── Step 3a: entities per sentence + UMLS match ──────────────
    trace.section("STEP 3a — entity extraction (per sentence) + UMLS matching")
    per_sent_entities = extract_entities_per_sentence(sentences)
    for i, ents in enumerate(per_sent_entities):
        trace.write(f"  S{i}  ({len(ents)} entities)")
        for e in ents:
            trace.write(
                f"    - surface={e.get('surface_form','')!r}  "
                f"normalized={e.get('normalized_form','')!r}  "
                f"group={e.get('semantic_group','')}  "
                f"tui={e.get('semantic_type_tui','')}"
            )

    deduped_entities = dedup_entities(per_sent_entities)
    trace.kv("unique entities (across all sentences)", len(deduped_entities))

    match_results, cuis = match_entities_to_umls(
        deduped_entities, umls_client, tui_to_group,
    )
    matched_n = sum(1 for r in match_results if r.get("matched"))
    trace.kv("UMLS match rate",
             f"{matched_n}/{len(deduped_entities)}  ({len(cuis)} unique CUIs)")
    for r in match_results:
        ent = r.get("entity", {})
        if r.get("matched"):
            top = r.get("matches", [])[:3]
            top_strs = [f"{m['cui']}:{m['name']}" for m in top]
            trace.write(
                f"  ✓ {ent.get('normalized_form','')!r}  "
                f"({r.get('match_type','')})  → {top_strs}"
            )
        else:
            trace.write(f"  ✗ {ent.get('normalized_form','')!r}  (no match)")

    # ── Step 3b: conditions per sentence ─────────────────────────
    trace.section("STEP 3b — condition extraction (per sentence)")
    per_sent_conditions = extract_conditions_per_sentence(sentences)
    all_conditions: list[dict] = []
    for i, conds in enumerate(per_sent_conditions):
        trace.write(f"  S{i}  ({len(conds)} conditions)")
        for c in conds:
            trace.write(f"    - {json.dumps(c, ensure_ascii=False)}")
            all_conditions.append(c)
    trace.kv("total conditions", len(all_conditions))

    cond_kws = conditions_to_keywords(all_conditions)
    trace.kv("condition keywords for Cypher CONTAINS", cond_kws)

    # ── Step 4: Neo4j retrieval ──────────────────────────────────
    trace.section("STEP 4 — Neo4j retrieval (priority 3-1 → 3-2 → 3-3)")
    rows, strategy, debug = retriever.retrieve(cuis, cond_kws, trace=trace)
    trace.kv("strategy chosen", strategy)
    trace.kv("rows retrieved", len(rows))

    retrieval_text = format_retrieval_result(rows, strategy)
    trace.block("retrieval_result (formatted text passed to LLM)",
                retrieval_text)

    # ── Step 4-analysis: per-CUI coverage in the retrieved rows ──
    cui_to_name: dict = {}
    for mr in match_results:
        for m in mr.get("matches", []):
            cui = m.get("cui", "")
            if cui and cui not in cui_to_name:
                cui_to_name[cui] = m.get("name", "")

    queried_cuis = debug.get("present_cuis") or cuis
    coverage = analyze_cui_coverage(rows, queried_cuis, cui_to_name, strategy)

    trace.section("STEP 4-analysis — Specific CUI coverage in retrieval rows")
    if strategy == "3-3_similar_conditions":
        trace.write("  (skipped — 3-3 returns conditions_json/sample, no head_id/tail_id)")
    elif not coverage["per_cui"]:
        trace.write("  (no CUIs queried)")
    else:
        top = coverage["per_cui"][0]
        trace.kv(
            "Dominant CUI",
            f"{top['cui']}:{top['name']!r} → {top['row_count']}/{coverage['total_rows']} "
            f"rows ({coverage['dominant_cui_share'] * 100:.1f}%)"
        )
        trace.kv(
            "CUIs with zero rows in retrieval",
            f"{coverage['zero_coverage_count']}/{len(queried_cuis)}"
        )
        trace.kv(
            "Distinct head/tail CUIs across rows",
            f"head={coverage['distinct_head_cuis']}, tail={coverage['distinct_tail_cuis']}"
        )
        trace.write("  Per-CUI breakdown (queried CUI → row count):")
        for c in coverage["per_cui"]:
            mark = "  " if c["row_count"] > 0 else " ✗"
            trace.write(f"   {mark} {c['cui']:<11} {c['row_count']:>4} rows  | {c['name']}")

    # ── Step 4-LLM: final answer ─────────────────────────────────
    trace.section("STEP 4-LLM — final answer prompt + response")
    user_prompt = build_answer_user_prompt(
        retrieval_text, sample.get("options", {}), sample.get("question", ""),
    )
    trace.block("system prompt", ANSWER_SYSTEM_PROMPT)
    trace.block("user prompt (retrieval + options + question)", user_prompt)

    raw, parsed = call_answer_llm(user_prompt)
    predicted = parse_answer_with_fallback(raw, parsed, sample.get("options", {}))

    trace.block("LLM raw response", raw or "(empty)")
    trace.kv("predicted (parsed)", predicted)

    # ── Step 5: RAG compare ──────────────────────────────────────
    gold = sample.get("answer", "")
    correct = normalize_for_compare(predicted) == normalize_for_compare(gold)
    trace.section("STEP 5 — RAG compare")
    trace.kv("gold_answer", gold)
    trace.kv("predicted_answer", predicted)
    trace.kv("correct", correct)

    # ── Step 6: No-RAG baseline ──────────────────────────────────
    trace.section("STEP 6 — No-RAG baseline ([QUESTION] + [OPTIONS] only)")
    baseline_user_prompt = build_baseline_user_prompt(
        sample.get("options", {}), sample.get("question", ""),
    )
    trace.block("baseline system prompt", BASELINE_SYSTEM_PROMPT)
    trace.block("baseline user prompt", baseline_user_prompt)

    baseline_raw, baseline_parsed = call_answer_llm(
        baseline_user_prompt, system_prompt=BASELINE_SYSTEM_PROMPT,
    )
    baseline_predicted = parse_answer_with_fallback(
        baseline_raw, baseline_parsed, sample.get("options", {}),
    )
    baseline_correct = (
        normalize_for_compare(baseline_predicted) == normalize_for_compare(gold)
    )

    trace.block("baseline LLM raw response", baseline_raw or "(empty)")
    trace.kv("baseline_predicted (parsed)", baseline_predicted)
    trace.kv("baseline_correct", baseline_correct)
    trace.write("")

    logger.info(
        f"  RAG[{('OK' if correct else 'MIS')}] pred={predicted!r}  |  "
        f"BASELINE[{('OK' if baseline_correct else 'MIS')}] pred={baseline_predicted!r}  |  "
        f"gold={gold!r}  |  strategy={strategy}"
    )

    return {
        "id": sid,
        "question": sample.get("question", ""),
        "context": sample.get("context", []),
        "options": sample.get("options", {}),
        "gold_answer": gold,
        "gold_answer_idx": sample.get("answer_idx", ""),
        "predicted_answer": predicted,
        "correct": correct,
        "baseline_predicted_answer": baseline_predicted,
        "baseline_correct": baseline_correct,
        "input_query": input_query,
        "n_sentences": len(sentences),
        "entities_per_sentence": per_sent_entities,
        "deduped_entities": deduped_entities,
        "match_results": match_results,
        "matched_cuis": cuis,
        "conditions_per_sentence": per_sent_conditions,
        "all_conditions": all_conditions,
        "condition_keywords": cond_kws,
        "retrieval_strategy": strategy,
        "retrieval_debug": debug,
        "retrieval_rows": rows,
        "retrieval_text": retrieval_text,
        "cui_retrieval_coverage": coverage,
        "final_user_prompt": user_prompt,
        "llm_raw_response": raw,
        "baseline_user_prompt": baseline_user_prompt,
        "baseline_llm_raw_response": baseline_raw,
    }


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    setup_logging("INFO")

    if not os.path.isfile(SAMPLE_PATH):
        logger.error(
            f"Sample file not found at {SAMPLE_PATH}. "
            f"Run `python sample_mediq.py` first."
        )
        sys.exit(1)

    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        sample_data = json.load(f)
    samples = sample_data.get("samples", [])
    logger.info(f"Loaded {len(samples)} samples from {SAMPLE_PATH}")

    tui_to_group, _ = load_semantic_groups_from_file(config.SEMANTIC_GROUPS_FILE)
    logger.info(f"Loaded {len(tui_to_group)} TUI→group mappings")

    umls_client = UMLSClient()

    results: list[dict] = []
    with TraceLog(TRACE_LOG_PATH) as trace, Neo4jRetriever(
        config.NEO4J_URI, config.NEO4J_USER,
        config.NEO4J_PASSWORD, config.NEO4J_DATABASE,
    ) as retriever:
        trace.header("MediQ GraphRAG retrieval trace", char="#")
        trace.kv("timestamp", datetime.now().isoformat())
        trace.kv("model", config.LLM_MODEL)
        trace.kv("reasoning_effort", config.LLM_REASONING_EFFORT)
        trace.kv("neo4j_uri", config.NEO4J_URI)
        trace.kv("neo4j_database", config.NEO4J_DATABASE)
        trace.kv("sample_path", SAMPLE_PATH)
        trace.kv("n_samples", len(samples))
        trace.write("")

        logger.info(
            f"Connected to Neo4j at {config.NEO4J_URI} "
            f"(database: {config.NEO4J_DATABASE})"
        )

        for sample in samples:
            try:
                r = run_one(sample, umls_client, tui_to_group, retriever, trace)
            except Exception as e:
                logger.exception(f"sample id={sample.get('id')} failed: {e}")
                trace.section("EXCEPTION")
                trace.write(f"  {e!r}")
                r = {"id": sample.get("id"), "error": str(e), "correct": False}
            results.append(r)

        # ── Final summary block in the trace (RAG vs Baseline) ──
        total = len(results)
        rag_correct_n = sum(1 for r in results if r.get("correct"))
        baseline_correct_n = sum(1 for r in results if r.get("baseline_correct"))
        rag_accuracy = rag_correct_n / total if total else 0.0
        baseline_accuracy = baseline_correct_n / total if total else 0.0

        both = sum(
            1 for r in results
            if r.get("correct") and r.get("baseline_correct")
        )
        rag_only = sum(
            1 for r in results
            if r.get("correct") and not r.get("baseline_correct")
        )
        baseline_only = sum(
            1 for r in results
            if r.get("baseline_correct") and not r.get("correct")
        )
        neither = total - both - rag_only - baseline_only

        trace.header("SUMMARY", char="#")
        trace.kv("RAG correct", f"{rag_correct_n}/{total}")
        trace.kv("RAG accuracy", f"{rag_accuracy * 100:.1f}%")
        trace.kv("Baseline (No-RAG) correct", f"{baseline_correct_n}/{total}")
        trace.kv("Baseline accuracy", f"{baseline_accuracy * 100:.1f}%")
        trace.write("")
        trace.write("Cross-tab (RAG vs Baseline):")
        trace.write(f"  Both correct:   {both}/{total}")
        trace.write(f"  RAG only:       {rag_only}/{total}")
        trace.write(f"  Baseline only:  {baseline_only}/{total}")
        trace.write(f"  Both wrong:     {neither}/{total}")
        trace.write("")
        for r in results:
            if "error" in r:
                trace.write(f"  id={r['id']}  ERROR: {r['error']}")
                continue
            rag_flag = "OK " if r.get("correct") else "MIS"
            bas_flag = "OK " if r.get("baseline_correct") else "MIS"
            trace.write(
                f"  [RAG {rag_flag} | BAS {bas_flag}] id={r['id']}  "
                f"strategy={r.get('retrieval_strategy','?')}  "
                f"pred={r.get('predicted_answer','')!r}  "
                f"baseline={r.get('baseline_predicted_answer','')!r}  "
                f"gold={r.get('gold_answer','')!r}"
            )

    summary = {
        "metadata": {
            "n_samples": total,
            "rag_correct": rag_correct_n,
            "rag_accuracy": round(rag_accuracy, 4),
            "baseline_correct": baseline_correct_n,
            "baseline_accuracy": round(baseline_accuracy, 4),
            "both_correct": both,
            "rag_only_correct": rag_only,
            "baseline_only_correct": baseline_only,
            "both_wrong": neither,
            "model": config.LLM_MODEL,
            "reasoning_effort": config.LLM_REASONING_EFFORT,
            "neo4j_uri": config.NEO4J_URI,
            "neo4j_database": config.NEO4J_DATABASE,
            "trace_log_path": TRACE_LOG_PATH,
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    logger.info("=" * 60)
    logger.info(f"RAG accuracy:      {rag_correct_n}/{total} = {rag_accuracy * 100:.1f}%")
    logger.info(f"Baseline accuracy: {baseline_correct_n}/{total} = {baseline_accuracy * 100:.1f}%")
    logger.info(f"Cross-tab: both={both}  rag_only={rag_only}  baseline_only={baseline_only}  neither={neither}")
    logger.info(f"Saved results: {RESULTS_PATH}")
    logger.info(f"Saved trace log: {TRACE_LOG_PATH}")
    logger.info("=" * 60)

    # Console table
    print()
    print(f"RAG accuracy:      {rag_correct_n}/{total} ({rag_accuracy * 100:.1f}%)")
    print(f"Baseline accuracy: {baseline_correct_n}/{total} ({baseline_accuracy * 100:.1f}%)")
    print(f"Cross-tab: both={both}  rag_only={rag_only}  baseline_only={baseline_only}  neither={neither}")
    print()
    for r in results:
        if "error" in r:
            print(f"id={r['id']}  ERROR: {r['error']}")
            continue
        rag_flag = "OK " if r.get("correct") else "MIS"
        bas_flag = "OK " if r.get("baseline_correct") else "MIS"
        print(
            f"[RAG {rag_flag} | BAS {bas_flag}] id={r['id']:<4} "
            f"strategy={r.get('retrieval_strategy','?'):<22}  "
            f"pred={r.get('predicted_answer','')!r}  "
            f"baseline={r.get('baseline_predicted_answer','')!r}  "
            f"gold={r.get('gold_answer','')!r}"
        )


if __name__ == "__main__":
    main()
