"""
LLM-based Entity Candidate Extractor (Stage 1)

Uses all 127 UMLS Semantic Types with example entities in the prompt and
calls OpenAI GPT via the Responses API.

Ref: https://developers.openai.com/api/docs/guides/migrate-to-responses
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from semantic_types import build_prompt_semantic_section

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are a medical Named Entity Recognition (NER) specialist \
trained to extract entity candidates from clinical guideline recommendation sentences.

Your task:
Given a clinical guideline recommendation sentence (and optionally its background context), \
extract ALL medical entity candidates that could be mapped to UMLS concepts.

{semantic_types_section}

=== EXTRACTION RULES ===
1. Extract the EXACT medical term as it appears ("surface_form").
2. Also provide a NORMALIZED form ("normalized_form") using standard UMLS-preferred English \
terminology. If the surface form is already standard, repeat it as the normalized form.
3. Classify each entity into its most likely Semantic Group (ACTI, ANAT, CHEM, CONC, DEVI, \
DISO, GENE, GEOG, LIVB, OBJC, OCCU, ORGA, PHEN, PHYS, PROC) and Semantic Type (TUI).
4. PRIORITIZE RECALL: when in doubt, INCLUDE the candidate. It is better to over-extract \
than to miss a medical entity.
5. Extract multi-word terms as complete phrases (e.g. "non-small cell lung cancer" not just "cancer").
6. Also extract important sub-components separately when clinically meaningful \
(e.g. "platinum-based chemotherapy" → extract both "platinum-based chemotherapy" AND "chemotherapy").
7. Include: drug names, diseases, symptoms, procedures, anatomical sites, patient populations, \
lab tests, devices, dosage/staging/grading terms, biomarkers, organisms.
8. Do NOT extract: purely syntactic elements, conjunctions, articles, \
recommendation-strength words (should, may, must, recommend).
9. When the guideline context is provided, use it to resolve ambiguous references \
(pronouns, abbreviations, "the agent", "this condition", etc.) in the recommendation sentence.

=== OUTPUT FORMAT ===
Respond ONLY with a valid JSON object. No markdown, no explanation, no preamble.
{{
  "entities": [
    {{
      "surface_form": "exact text from the recommendation sentence",
      "normalized_form": "UMLS-preferred English term",
      "semantic_group": "GROUP_ABBR",
      "semantic_type_tui": "TUI code",
      "semantic_type_name": "Semantic Type name"
    }}
  ]
}}

If no medical entities are found, return: {{"entities": []}}
"""


FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "[RECOMMENDATION]\n"
            "Patients aged 65 or older with stage III non-small cell lung cancer "
            "and EGFR mutation should receive erlotinib as first-line therapy "
            "rather than platinum-based chemotherapy.\n\n"
            "[GUIDELINE CONTEXT]\n"
            "This guideline addresses the management of advanced non-small cell "
            "lung cancer (NSCLC) in adult patients. Erlotinib (Tarceva) is an "
            "oral EGFR tyrosine kinase inhibitor approved for NSCLC with "
            "activating EGFR mutations."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "entities": [
                    {
                        "surface_form": "Patients aged 65 or older",
                        "normalized_form": "Aged",
                        "semantic_group": "LIVB",
                        "semantic_type_tui": "T100",
                        "semantic_type_name": "Age Group",
                    },
                    {
                        "surface_form": "stage III",
                        "normalized_form": "Stage III",
                        "semantic_group": "CONC",
                        "semantic_type_tui": "T080",
                        "semantic_type_name": "Qualitative Concept",
                    },
                    {
                        "surface_form": "non-small cell lung cancer",
                        "normalized_form": "Non-Small Cell Lung Carcinoma",
                        "semantic_group": "DISO",
                        "semantic_type_tui": "T191",
                        "semantic_type_name": "Neoplastic Process",
                    },
                    {
                        "surface_form": "EGFR mutation",
                        "normalized_form": "EGFR Gene Mutation",
                        "semantic_group": "DISO",
                        "semantic_type_tui": "T049",
                        "semantic_type_name": "Cell or Molecular Dysfunction",
                    },
                    {
                        "surface_form": "EGFR",
                        "normalized_form": "Epidermal Growth Factor Receptor",
                        "semantic_group": "GENE",
                        "semantic_type_tui": "T028",
                        "semantic_type_name": "Gene or Genome",
                    },
                    {
                        "surface_form": "erlotinib",
                        "normalized_form": "Erlotinib",
                        "semantic_group": "CHEM",
                        "semantic_type_tui": "T121",
                        "semantic_type_name": "Pharmacologic Substance",
                    },
                    {
                        "surface_form": "first-line therapy",
                        "normalized_form": "First Line Therapy",
                        "semantic_group": "PROC",
                        "semantic_type_tui": "T061",
                        "semantic_type_name": "Therapeutic or Preventive Procedure",
                    },
                    {
                        "surface_form": "platinum-based chemotherapy",
                        "normalized_form": "Platinum-based Antineoplastic Agent",
                        "semantic_group": "CHEM",
                        "semantic_type_tui": "T121",
                        "semantic_type_name": "Pharmacologic Substance",
                    },
                    {
                        "surface_form": "chemotherapy",
                        "normalized_form": "Chemotherapy",
                        "semantic_group": "PROC",
                        "semantic_type_tui": "T061",
                        "semantic_type_name": "Therapeutic or Preventive Procedure",
                    },
                    {
                        "surface_form": "tyrosine kinase inhibitor",
                        "normalized_form": "Tyrosine Kinase Inhibitor",
                        "semantic_group": "CHEM",
                        "semantic_type_tui": "T121",
                        "semantic_type_name": "Pharmacologic Substance",
                    },
                    {
                        "surface_form": "lung cancer",
                        "normalized_form": "Lung Neoplasm",
                        "semantic_group": "DISO",
                        "semantic_type_tui": "T191",
                        "semantic_type_name": "Neoplastic Process",
                    },
                ],
            },
            ensure_ascii=False,
        ),
    },
]


_cached_system_prompt = None


def _build_system_prompt() -> str:
    """Build the full system prompt with all 127 semantic types (cached after first call)."""
    global _cached_system_prompt
    if _cached_system_prompt is None:
        semantic_section = build_prompt_semantic_section()
        _cached_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            semantic_types_section=semantic_section
        )
    return _cached_system_prompt


def _build_user_message(recommendation: dict) -> str:
    """Build user message from a recommendation dict."""
    parts = [f"[RECOMMENDATION]\n{recommendation['text']}"]

    context = recommendation.get("guideline_context", "")
    if context:
        parts.append(f"\n[GUIDELINE CONTEXT]\n{context}")

    return "\n".join(parts)


def _parse_llm_response(response_text: str) -> list[dict]:
    """Parse LLM JSON response, handling common formatting issues."""
    text = response_text.strip()

    # Remove markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response as JSON: {text[:200]}")
                return []
        else:
            logger.warning(f"No JSON found in LLM response: {text[:200]}")
            return []

    entities = data.get("entities", [])

    # Validate each entity has required fields
    valid_entities = []
    required_fields = {"surface_form", "normalized_form", "semantic_group"}
    for ent in entities:
        if isinstance(ent, dict) and required_fields.issubset(ent.keys()):
            # Ensure string values
            for key in ent:
                if isinstance(ent[key], (int, float)):
                    ent[key] = str(ent[key])
            valid_entities.append(ent)
        else:
            logger.debug(f"Skipping invalid entity: {ent}")

    return valid_entities


# ──────────────────────────────────────────────────────────────────────
# LLM Call (OpenAI Responses API)
# ──────────────────────────────────────────────────────────────────────

# Module-level client singleton (reuse for connection pooling)
_openai_client = None


def _get_openai_client(api_key: str = None):
    """Get or create the OpenAI client singleton."""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "Install the OpenAI SDK: pip install openai>=1.70.0"
            )
        api_key = api_key or config.OPENAI_API_KEY
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def call_openai(
    recommendation: dict,
    api_key: str = None,
    model: str = None,
) -> list[dict]:
    """Call OpenAI Responses API for entity extraction; return parsed entities."""
    client = _get_openai_client(api_key)
    model = model or config.LLM_MODEL

    input_messages = list(FEW_SHOT_EXAMPLES) + [
        {"role": "user", "content": _build_user_message(recommendation)},
    ]

    try:
        response = client.responses.create(
            model=model,
            instructions=_build_system_prompt(),
            input=input_messages,
            temperature=0.0,
            max_output_tokens=config.LLM_MAX_TOKENS,
            store=False,
        )
        return _parse_llm_response(response.output_text)
    except Exception as e:
        logger.error(f"OpenAI Responses API call failed: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────
# Batch Processing
# ──────────────────────────────────────────────────────────────────────

def _extract_one(i: int, rec: dict, llm_fn) -> tuple[list[dict], bool]:
    """Worker: extract entities from one recommendation. Returns (entities, failed)."""
    try:
        entities = llm_fn(rec)
    except Exception as e:
        logger.error(f"Entity extraction failed for rec #{i}: {e}")
        return [], True

    gid = rec.get("guideline_id", "")
    strength = rec.get("strength", "")
    text = rec.get("text", "")
    for ent in entities:
        ent["source_guideline_id"] = gid
        ent["source_strength"] = strength
        ent["source_text"] = text
    return entities, False


def extract_entities_batch(
    recommendations: list[dict],
    llm_fn=None,
    progress_interval: int = 10,
    max_workers: int = None,
) -> list[dict]:
    """
    Run entity extraction on a batch of recommendations, parallelized.

    Args:
        recommendations: list of recommendation dicts from crest_parser
        llm_fn: callable(recommendation_dict) -> list[entity_dicts]
                defaults to call_openai (Responses API)
        progress_interval: log progress every N sentences
        max_workers: parallel LLM workers (defaults to config.LLM_MAX_WORKERS)

    Returns:
        list of entity dicts in original recommendation order, each augmented
        with source metadata
    """
    if llm_fn is None:
        llm_fn = call_openai
    max_workers = max_workers or config.LLM_MAX_WORKERS

    n = len(recommendations)
    per_rec: list[tuple[list[dict], bool]] = [([], False)] * n
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_extract_one, i, rec, llm_fn): i
            for i, rec in enumerate(recommendations)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            per_rec[i] = fut.result()
            completed += 1
            if completed % progress_interval == 0:
                running_total = sum(len(r[0]) for r in per_rec)
                logger.info(
                    f"  Entity extraction progress: {completed}/{n} "
                    f"sentences, {running_total} entities so far"
                )

    all_entities: list[dict] = []
    failed_count = 0
    for ents, failed in per_rec:
        all_entities.extend(ents)
        if failed:
            failed_count += 1

    logger.info(
        f"Entity extraction complete: {len(all_entities)} entities "
        f"from {n} sentences ({failed_count} failures)"
    )
    return all_entities


def deduplicate_entities(entities: list[dict]) -> dict:
    """
    Deduplicate entities by normalized_form (case-insensitive).

    Returns: dict mapping normalized_key -> entity dict with aggregated sources
    """
    unique = {}

    for ent in entities:
        key = ent.get("normalized_form", ent["surface_form"]).lower().strip()

        if key not in unique:
            unique[key] = {
                "surface_form": ent["surface_form"],
                "normalized_form": ent.get("normalized_form", ent["surface_form"]),
                "semantic_group": ent.get("semantic_group", ""),
                "semantic_type_tui": ent.get("semantic_type_tui", ""),
                "semantic_type_name": ent.get("semantic_type_name", ""),
                "source_guidelines": [],
                "source_count": 0,
            }

        entry = unique[key]
        gid = ent.get("source_guideline_id", "")
        if gid and gid not in entry["source_guidelines"]:
            entry["source_guidelines"].append(gid)
        entry["source_count"] += 1

    logger.info(
        f"Deduplication: {len(entities)} total → {len(unique)} unique entities"
    )
    return unique
