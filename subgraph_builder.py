"""
1-hop Subgraph Builder (Stage 2-B)

For each seed CUI (matched in Stage 2-A), collects all relations
via UMLS /relations endpoint and converts them to (head, relation, tail) triples.

Design rationale (from research proposal §2.4 Step 2):
  "일치 entity를 seed로 하여 1-hop subgraph를 수집·정합"
"""

import logging

from umls_client import UMLSClient
import config

logger = logging.getLogger(__name__)


def _extract_id_from_uri(uri: str) -> str:
    """Extract concept/source ID from the last segment of a UMLS URI."""
    if not uri:
        return ""
    parts = uri.rstrip("/").split("/")
    return parts[-1] if parts else ""


def build_1hop_subgraph(
    client: UMLSClient,
    seed_cui: str,
    seed_name: str = "",
    skip_relations: set = None,
) -> list[dict]:
    """
    Collect 1-hop subgraph for a seed CUI as a list of triples.

    The seed CUI is always the head. The neighbor entity becomes the tail.
    Each triple includes empty condition fields for Stage 3 to populate.
    """
    skip_relations = skip_relations or config.SKIP_RELATION_LABELS

    if not seed_name:
        concept = client.get_concept(seed_cui)
        seed_name = concept.get("name", seed_cui) if concept else seed_cui

    relations = client.get_relations(seed_cui)

    triples = []
    for rel in relations:
        rel_label = rel.get("relationLabel", "")
        if rel_label in skip_relations:
            continue

        additional_label = rel.get("additionalRelationLabel", "")
        related_name = rel.get("relatedIdName", "")
        related_uri = rel.get("relatedId", "")
        from_id = _extract_id_from_uri(rel.get("relatedFromId", ""))
        root_source = rel.get("rootSource", "")

        if not (from_id and related_name):
            continue

        relation_type = additional_label if additional_label else rel_label
        tail_id = _extract_id_from_uri(related_uri)

        triples.append({
            "head_cui": seed_cui,
            "head_name": seed_name,
            "relation": relation_type,
            "relation_label": rel_label,
            "tail_id": tail_id,
            "tail_name": related_name,
            "root_source": root_source,
            "seed_cui": seed_cui,
            "conditions": [],
            "condition_source": [],
            "recommendation_strength": None,
        })

    return triples


def build_subgraphs_batch(
    client: UMLSClient,
    matched_cuis: dict,
    progress_interval: int = 50,
) -> list[dict]:
    """Build 1-hop subgraphs for all matched CUIs."""
    all_triples = []
    cui_list = list(matched_cuis.items())

    for i, (cui, info) in enumerate(cui_list):
        triples = build_1hop_subgraph(client, cui, seed_name=info.get("name", ""))
        all_triples.extend(triples)

        if (i + 1) % progress_interval == 0:
            logger.info(
                f"  Subgraph progress: {i + 1}/{len(cui_list)} CUIs, "
                f"{len(all_triples)} triples collected"
            )

    logger.info(
        f"Subgraph collection complete: {len(all_triples)} triples "
        f"from {len(cui_list)} seed CUIs"
    )

    if all_triples:
        logger.info(
            f"Triple stats: {len(set(t['relation'] for t in all_triples))} unique relations, "
            f"{len(set(t['root_source'] for t in all_triples))} source vocabularies, "
            f"{len(set(t['tail_id'] for t in all_triples))} unique tail entities"
        )

    return all_triples


def deduplicate_triples(triples: list[dict]) -> list[dict]:
    """Remove duplicate triples (same head, relation, tail)."""
    seen = set()
    unique = []
    for t in triples:
        key = (t["head_cui"], t["relation"], t["tail_id"])
        if key not in seen:
            seen.add(key)
            unique.append(t)

    logger.info(f"Triple deduplication: {len(triples)} → {len(unique)}")
    return unique
