"""
Stage 4: Neo4j Knowledge Graph Construction

Loads stage3_condition_augmented_triples.json and ingests it into Neo4j.

Graph schema:
  (:Concept {id, name, is_cui})
      -[:RELATES { relation, relation_label, root_source,
                   has_conditions, conditions_json, condition_count,
                   condition_logic,
                   guideline_id, evidence_level, evidence_texts,
                   recommendation_strength }]->
  (:Concept {id, name, is_cui})

Identity model:
  - head node id  = head_cui (always a CUI from Stage 2)
  - tail node id  = tail_id when it matches the UMLS CUI pattern
                    (^C\\d{7}$); otherwise namespaced as
                    "<root_source>:<tail_id>" to avoid cross-vocabulary
                    collisions (e.g. HCPCS codes like "C1300").

Edge dedup key (Cypher MERGE): (head, tail, relation, guideline_id).
  Stage 3 currently emits one row per triple, attaching the first matched
  guideline only, so a single triple yields a single edge. The dedup key
  is shaped to support future per-guideline parallel edges without a
  schema change. Triples with no matched recommendation use guideline_id="".

All operations use MERGE → fully idempotent re-runs.
"""

import json
import logging
import re
import time
from datetime import datetime
from typing import Optional

from neo4j import GraphDatabase

import config

# UMLS CUI pattern: literal "C" + 7 digits. Tighter than startswith("C") to
# avoid mis-classifying HCPCS source codes (e.g. "C1300") as CUIs.
_CUI_RE = re.compile(r"^C\d{7}$")

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Cypher
# ──────────────────────────────────────────────────────────────────────

CONSTRAINT_QUERIES = [
    "CREATE CONSTRAINT concept_id IF NOT EXISTS "
    "FOR (c:Concept) REQUIRE c.id IS UNIQUE",
]

INDEX_QUERIES = [
    "CREATE INDEX concept_name IF NOT EXISTS "
    "FOR (c:Concept) ON (c.name)",
    "CREATE INDEX rel_relation IF NOT EXISTS "
    "FOR ()-[r:RELATES]-() ON (r.relation)",
    "CREATE INDEX rel_has_conditions IF NOT EXISTS "
    "FOR ()-[r:RELATES]-() ON (r.has_conditions)",
    "CREATE INDEX rel_guideline IF NOT EXISTS "
    "FOR ()-[r:RELATES]-() ON (r.guideline_id)",
]

UPSERT_QUERY = """
UNWIND $rows AS row
MERGE (h:Concept {id: row.head_id})
  ON CREATE SET h.name = row.head_name, h.is_cui = true
  ON MATCH  SET h.name = coalesce(h.name, row.head_name)
MERGE (t:Concept {id: row.tail_id})
  ON CREATE SET t.name = row.tail_name, t.is_cui = row.tail_is_cui
  ON MATCH  SET t.name = coalesce(t.name, row.tail_name)
MERGE (h)-[r:RELATES {relation: row.relation, guideline_id: row.guideline_id}]->(t)
SET r.relation_label          = row.relation_label,
    r.root_source             = row.root_source,
    r.has_conditions          = row.has_conditions,
    r.conditions_json         = row.conditions_json,
    r.condition_count         = row.condition_count,
    r.condition_logic         = row.condition_logic,
    r.evidence_level          = row.evidence_level,
    r.evidence_texts          = row.evidence_texts,
    r.recommendation_strength = row.recommendation_strength
"""


# ──────────────────────────────────────────────────────────────────────
# Triple → flat row conversion
# ──────────────────────────────────────────────────────────────────────

SKIP_INVALID_ID = "invalid_id"
SKIP_PARSE_FAILED = "parse_failed"
SKIP_PARSE_FAILED_LEGACY = "parse_failed_legacy"


def _is_likely_parse_failed_legacy(t: dict) -> bool:
    """Detect parse failure in Stage 3 outputs that PRE-DATE the explicit
    parse_failed flag (i.e., produced before that field was added).

    Reasoning:
      - has_conditions=True  → LLM definitely returned valid output.
      - recommendation_strength is None/empty → triple never reached the
        LLM (head-tail bidirectional filter rejected it). It's a
        legitimate "no-conditions" triple, NOT a failure.
      - recommendation_strength set but condition_source.guideline_id
        empty → triple WAS sent to the LLM, but the result was lost
        (response truncated and the old apply_conditions_to_triples
        defaulted condition_source to all-empty). This is the unique
        signature of legacy parse failures.
    """
    if t.get("has_conditions"):
        return False

    strength = t.get("recommendation_strength")
    if not strength:  # None or empty string
        return False

    cs = t.get("condition_source") or {}
    guideline = (cs.get("guideline_id") or "").strip()
    return not guideline


def _classify_parse_failed(t: dict) -> Optional[str]:
    """Return the skip-reason string if this triple should be excluded as
    a parse failure, else None.

    Honors the explicit `parse_failed` flag when present (new Stage 3
    runs). For legacy data without the flag, falls back to the
    signature-based heuristic.
    """
    flag = t.get("parse_failed")
    if flag is True:
        return SKIP_PARSE_FAILED
    if flag is False:
        return None  # explicitly OK
    # Field missing → legacy Stage 3 output
    if _is_likely_parse_failed_legacy(t):
        return SKIP_PARSE_FAILED_LEGACY
    return None


def _triple_to_row(t: dict) -> tuple[Optional[dict], Optional[str]]:
    """Convert a Stage 3 triple dict to a flat row for the UNWIND query.

    Returns (row, skip_reason). When `row` is None, `skip_reason` is one of:
      - "invalid_id"           : missing head_cui or tail_id
      - "parse_failed"         : explicit Stage 3 parse_failed=True
      - "parse_failed_legacy"  : heuristic match on legacy Stage 3 output
                                 lacking the parse_failed flag
    """
    # Drop triples whose Stage 3 LLM result was lost. Including them with
    # has_conditions=False would silently misrepresent conditioned triples
    # as unconditioned in the KG.
    parse_skip = _classify_parse_failed(t)
    if parse_skip is not None:
        return None, parse_skip

    head_id = (t.get("head_cui") or "").strip()
    raw_tail = (t.get("tail_id") or "").strip()
    if not head_id or not raw_tail:
        return None, SKIP_INVALID_ID

    tail_is_cui = bool(_CUI_RE.match(raw_tail))
    root_src = (t.get("root_source") or "").strip()
    # Namespace non-CUI tail ids by their source vocabulary to avoid
    # collisions like SNOMEDCT_US "12345" colliding with another vocab's "12345".
    tail_id = raw_tail if tail_is_cui or not root_src else f"{root_src}:{raw_tail}"

    cs = t.get("condition_source") or {}
    conditions = t.get("conditions") or []

    return {
        "head_id": head_id,
        "head_name": t.get("head_name", "") or "",
        "tail_id": tail_id,
        "tail_name": t.get("tail_name", "") or "",
        "tail_is_cui": tail_is_cui,
        "relation": t.get("relation", "") or "",
        "relation_label": t.get("relation_label", "") or "",
        "root_source": root_src,
        "has_conditions": bool(t.get("has_conditions", False)),
        "conditions_json": t.get("conditions_json", "[]") or "[]",
        "condition_count": len(conditions),
        "condition_logic": t.get("condition_logic"),  # None | "AND" | "OR" | "NOT"
        "guideline_id": cs.get("guideline_id", "") or "",
        "evidence_level": cs.get("evidence_level", "") or "",
        "evidence_texts": cs.get("evidence_texts") or [],
        "recommendation_strength": t.get("recommendation_strength"),  # may be None
    }, None


# ──────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────

class Neo4jGraphBuilder:
    """Thin wrapper over the official Neo4j Python driver."""

    def __init__(self, uri: str, user: str, password: str, database: str):
        self.uri = uri
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        self.driver.close()

    def verify(self):
        """Raise if the server is unreachable or auth is wrong."""
        self.driver.verify_connectivity()

    def setup_schema(self):
        """Create constraints + indexes (idempotent: IF NOT EXISTS)."""
        with self.driver.session(database=self.database) as s:
            for q in CONSTRAINT_QUERIES:
                s.run(q)
            for q in INDEX_QUERIES:
                s.run(q)
        logger.info(
            f"Schema ready: {len(CONSTRAINT_QUERIES)} constraint(s), "
            f"{len(INDEX_QUERIES)} index(es)"
        )

    def clear(self):
        """DESTRUCTIVE: delete every node and relationship in the database."""
        logger.warning("Clearing database (MATCH (n) DETACH DELETE n)")
        with self.driver.session(database=self.database) as s:
            s.run("MATCH (n) DETACH DELETE n")

    def upsert_triples(
        self,
        triples: list[dict],
        batch_size: int = 500,
        progress_interval: int = 2000,
    ) -> dict:
        """Bulk-MERGE all triples.

        Returns a dict with: ingested, skipped_invalid_id,
        skipped_parse_failed, skipped_parse_failed_legacy.
        """
        rows: list[dict] = []
        skipped_invalid = 0
        skipped_parse_failed = 0
        skipped_parse_failed_legacy = 0
        for t in triples:
            row, reason = _triple_to_row(t)
            if row is None:
                if reason == SKIP_PARSE_FAILED:
                    skipped_parse_failed += 1
                elif reason == SKIP_PARSE_FAILED_LEGACY:
                    skipped_parse_failed_legacy += 1
                else:
                    skipped_invalid += 1
            else:
                rows.append(row)

        total = len(rows)
        if skipped_invalid:
            logger.warning(
                f"Skipped {skipped_invalid} triples missing head_cui or tail_id"
            )
        if skipped_parse_failed:
            logger.warning(
                f"Skipped {skipped_parse_failed} triples with parse_failed=True "
                f"(explicit Stage 3 flag)"
            )
        if skipped_parse_failed_legacy:
            logger.warning(
                f"Skipped {skipped_parse_failed_legacy} triples flagged by "
                f"legacy heuristic (Stage 3 ran before parse_failed was "
                f"introduced; signature: strength set + guideline_id empty)"
            )
        logger.info(f"Upserting {total} triples in batches of {batch_size}...")

        with self.driver.session(database=self.database) as s:
            for i in range(0, total, batch_size):
                chunk = rows[i:i + batch_size]
                s.run(UPSERT_QUERY, rows=chunk)
                done = min(i + batch_size, total)
                if done % progress_interval < batch_size or done == total:
                    logger.info(f"  Progress: {done}/{total} triples")

        return {
            "ingested": total,
            "skipped_invalid_id": skipped_invalid,
            "skipped_parse_failed": skipped_parse_failed,
            "skipped_parse_failed_legacy": skipped_parse_failed_legacy,
        }

    def summary_stats(self) -> dict:
        """Aggregate node/edge counts post-ingest."""
        queries = {
            "nodes":
                "MATCH (n:Concept) RETURN count(n) AS c",
            "edges":
                "MATCH ()-[r:RELATES]->() RETURN count(r) AS c",
            "edges_with_conditions":
                "MATCH ()-[r:RELATES]->() WHERE r.has_conditions "
                "RETURN count(r) AS c",
            "unique_relations":
                "MATCH ()-[r:RELATES]->() RETURN count(DISTINCT r.relation) AS c",
            "unique_guidelines":
                "MATCH ()-[r:RELATES]->() WHERE r.guideline_id <> '' "
                "RETURN count(DISTINCT r.guideline_id) AS c",
        }
        out = {}
        with self.driver.session(database=self.database) as s:
            for k, q in queries.items():
                rec = s.run(q).single()
                out[k] = rec["c"] if rec else 0
        return out


# ──────────────────────────────────────────────────────────────────────
# Top-level entry point
# ──────────────────────────────────────────────────────────────────────

def build_graph_from_file(
    triples_path: str,
    uri: str = None,
    user: str = None,
    password: str = None,
    database: str = None,
    clear_first: bool = False,
    batch_size: int = None,
) -> dict:
    """Load Stage 3 output JSON and build the Neo4j graph.

    Args:
        triples_path: path to stage3_condition_augmented_triples.json
        uri/user/password/database: Neo4j connection (defaults from config)
        clear_first: wipe the database before ingest (DESTRUCTIVE)
        batch_size: UNWIND batch size

    Returns:
        dict containing ingest counts and graph stats.
    """
    uri = uri or config.NEO4J_URI
    user = user or config.NEO4J_USER
    password = password or config.NEO4J_PASSWORD
    database = database or config.NEO4J_DATABASE
    batch_size = batch_size or config.NEO4J_BATCH_SIZE

    with open(triples_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    triples = data.get("triples", [])
    logger.info(f"Loaded {len(triples)} triples from {triples_path}")

    start = time.time()
    with Neo4jGraphBuilder(uri, user, password, database) as builder:
        builder.verify()
        logger.info(f"Connected to {uri} (database: {database})")

        if clear_first:
            builder.clear()

        builder.setup_schema()
        ingest_result = builder.upsert_triples(triples, batch_size=batch_size)
        stats = builder.summary_stats()

    elapsed = time.time() - start
    logger.info("Neo4j graph build complete:")
    logger.info(f"  Triples ingested:                 {ingest_result['ingested']}")
    logger.info(f"  Skipped (invalid id):             {ingest_result['skipped_invalid_id']}")
    logger.info(f"  Skipped (parse_failed flag):      {ingest_result['skipped_parse_failed']}")
    logger.info(f"  Skipped (legacy heuristic):       {ingest_result['skipped_parse_failed_legacy']}")
    logger.info(f"  Concepts (nodes):                 {stats['nodes']}")
    logger.info(f"  Relationships (edges):            {stats['edges']}")
    logger.info(f"  Edges with conditions:            {stats['edges_with_conditions']}")
    logger.info(f"  Unique relations:                 {stats['unique_relations']}")
    logger.info(f"  Unique guidelines:                {stats['unique_guidelines']}")
    logger.info(f"  Elapsed: {elapsed:.1f}s")

    return {
        **ingest_result,
        "elapsed_seconds": round(elapsed, 1),
        "neo4j_uri": uri,
        "neo4j_database": database,
        "timestamp": datetime.now().isoformat(),
        **stats,
    }
