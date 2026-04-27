"""
Configuration for Medical KG Pipeline.

Stage 0: CREST Parsing & Recommendation/Context Extraction
Stage 1: Entity Candidate Extraction (LLM)
Stage 2: UMLS Layer Construction (UMLS REST API)
Stage 3: Condition Augmentation (LLM + CREST context)
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
LLM_REASONING_EFFORT = "low"  # none | low | medium | high | xhigh

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

# ── Stage 3 Output ──
OUTPUT_AUGMENTED_TRIPLES_FILE = "stage3_condition_augmented_triples.json"
STAGE3_MAX_RECS_PER_TRIPLE = 3   # max recommendation sentences matched per triple
STAGE3_LLM_CHUNK_SIZE = 12       # triples per LLM call (json_object output → no truncation cap)
STAGE3_REC_TEXT_MAX_CHARS = 320  # truncate long recommendation sentences to save prompt tokens
STAGE3_RETRY_CHUNK_SIZE = 4      # missing triple retry chunk size
STAGE3_MAX_RETRY_DEPTH = 3       # bounded recursive retry depth

# ── Stage 4: Neo4j Knowledge Graph ──
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")
NEO4J_BATCH_SIZE = 500       # rows per UNWIND batch
OUTPUT_NEO4J_SUMMARY_FILE = "stage4_neo4j_summary.json"

# ── Concurrency ──
# UMLS shares a 20 req/s ceiling globally; ~8 workers is enough to saturate it
# given typical 200–500 ms request latency.
UMLS_MAX_WORKERS = 8
# OpenAI tier-dependent; conservative default keeps us under most RPM limits.
LLM_MAX_WORKERS = 8          # doubled for better parallelism

# ── Entity Matcher Settings ──
MAX_SEARCH_RESULTS_EXACT = 200
MAX_SEARCH_RESULTS_NORMALIZED = 50
MAX_SEARCH_RESULTS_WORDS = 25
MAX_RELATIONS_PAGE_SIZE = 200

# ── Primary HTML Context ──
PRIMARY_CONTEXT_MAX_CHARS = 4000