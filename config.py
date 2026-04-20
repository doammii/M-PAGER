"""
Configuration for Medical KG Pipeline
Stage 1: Entity Candidate Extraction (LLM + CREST)
Stage 2: UMLS Layer Construction (UMLS REST API)
"""

import os

# ── UMLS REST API ──
UMLS_API_KEY = os.environ.get("UMLS_API_KEY", "YOUR_UMLS_API_KEY_HERE")
UMLS_BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
UMLS_VERSION = "current"
UMLS_RATE_LIMIT_SLEEP = 0.05  # 20 req/s

# ── LLM API (OpenAI GPT) ──
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
LLM_MODEL = "gpt-5.4-mini"
LLM_MAX_TOKENS = 4096

# ── CREST Corpus Paths ──
CREST_XML_DIR = os.environ.get("CREST_XML_DIR", "./crest/xml")
CREST_PRIMARY_DIR = os.environ.get("CREST_PRIMARY_DIR", "./crest/primary")

# ── Semantic Groups File ──
SEMANTIC_GROUPS_FILE = os.environ.get(
    "SEMANTIC_GROUPS_FILE",
    "./UMLS_semantic_network_semantic_groups.txt",
)

# ── Output ──
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
OUTPUT_RECOMMENDATIONS_FILE = "stage0_recommendations.json"
OUTPUT_ENTITIES_FILE = "stage1_entity_candidates.json"
OUTPUT_MATCHED_FILE = "stage2_umls_matched.json"
OUTPUT_TRIPLES_FILE = "stage2_umls_layer_triples.json"
OUTPUT_PIPELINE_LOG = "pipeline_log.json"

# ── Entity Matcher Settings ──
MAX_SEARCH_RESULTS_EXACT = 200
MAX_SEARCH_RESULTS_NORMALIZED = 50
MAX_SEARCH_RESULTS_WORDS = 25
MAX_RELATIONS_PAGE_SIZE = 200

# ── Subgraph Settings ──
SKIP_RELATION_LABELS = {"SIB"}

# ── Semantic Group Filter ──
# Applied in Stage 4 (Layer Integration) to filter tail entities.
# Stage 2 collects all relations without filtering to maximize recall.
RELEVANT_SEMANTIC_GROUPS = {
    "DISO", "CHEM", "PROC", "ANAT", "PHYS",
    "GENE", "LIVB", "DEVI", "PHEN", "CONC",
    "ACTI", "OBJC",
}

# ── Primary HTML Context ──
PRIMARY_CONTEXT_MAX_CHARS = 4000
