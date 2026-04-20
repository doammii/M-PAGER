"""
Medical KG Pipeline — Main Orchestration
Stage 1: Entity Candidate Extraction (CREST + LLM)
Stage 2: UMLS Layer Construction (UMLS REST API)

Usage:
    python pipeline.py --umls-key YOUR_KEY --openai-key YOUR_KEY
    python pipeline.py --xml-dir ./crest/xml --primary-dir ./crest/primary
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import config
from crest_parser import extract_from_both_sources
from entity_extractor import (
    call_openai,
    extract_entities_batch,
    deduplicate_entities,
)
from semantic_types import load_semantic_groups_from_file
from umls_client import UMLSClient
from entity_matcher import match_entities_batch
from subgraph_builder import build_subgraphs_batch, deduplicate_triples

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def save_json(data, filepath: str):
    """Save data as JSON with UTF-8 encoding."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Saved: {filepath}")


def run_stage1(
    recommendations: list[dict],
    llm_fn=None,
    progress_interval: int = 10,
) -> tuple[list[dict], dict]:
    """
    Stage 1: Entity Candidate Extraction

    Input: CREST recommendation sentences (with guideline context)
    Output: deduplicated entity candidates with source metadata
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: Entity Candidate Extraction")
    logger.info("=" * 60)

    if llm_fn is None:
        llm_fn = call_openai

    all_entities = extract_entities_batch(
        recommendations, llm_fn=llm_fn, progress_interval=progress_interval,
    )

    unique_entities = deduplicate_entities(all_entities)
    return all_entities, unique_entities


def run_stage2(
    unique_entities: dict,
    umls_client: UMLSClient,
    tui_to_group: dict,
    progress_interval: int = 50,
) -> tuple[list[dict], dict, list[dict]]:
    """
    Stage 2: UMLS Layer Construction

    Input: deduplicated entity candidates
    Output: match results, matched CUIs, 1-hop triples
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: UMLS Layer Construction")
    logger.info("=" * 60)

    logger.info("Stage 2-A: Entity Candidate → UMLS CUI matching")
    match_results, matched_cuis = match_entities_batch(
        unique_entities, umls_client, tui_to_group,
        progress_interval=progress_interval,
    )

    logger.info("Stage 2-B: 1-hop subgraph collection")
    all_triples = build_subgraphs_batch(
        umls_client, matched_cuis, progress_interval=progress_interval,
    )

    unique_triples = deduplicate_triples(all_triples)
    return match_results, matched_cuis, unique_triples


def run_pipeline(
    xml_dir: str = None,
    primary_dir: str = None,
    semantic_groups_file: str = None,
    umls_api_key: str = None,
    openai_api_key: str = None,
    output_dir: str = None,
    max_recommendations: int = None,
):
    """Run the full Stage 1 + Stage 2 pipeline."""
    start_time = time.time()

    if umls_api_key:
        config.UMLS_API_KEY = umls_api_key
    if openai_api_key:
        config.OPENAI_API_KEY = openai_api_key

    xml_dir = xml_dir or config.CREST_XML_DIR
    primary_dir = primary_dir or config.CREST_PRIMARY_DIR
    semantic_groups_file = semantic_groups_file or config.SEMANTIC_GROUPS_FILE
    output_dir = output_dir or config.OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Medical KG Pipeline — Stage 1 & Stage 2")
    logger.info(f"  CREST xml/: {xml_dir}")
    logger.info(f"  CREST primary/: {primary_dir}")
    logger.info(f"  Output: {output_dir}")

    # ── Load Semantic Groups ──
    tui_to_group, tui_to_name = load_semantic_groups_from_file(semantic_groups_file)
    logger.info(f"Loaded {len(tui_to_group)} semantic type → group mappings")

    # ── Parse CREST Corpus ──
    recommendations = extract_from_both_sources(
        xml_dir=xml_dir, primary_dir=primary_dir,
    )

    if not recommendations:
        logger.error("No recommendations extracted. Check CREST paths.")
        return

    if max_recommendations:
        recommendations = recommendations[:max_recommendations]
        logger.info(f"Limited to {max_recommendations} recommendations (test mode)")

    # ── Save recommendations for Stage 3 reuse ──
    save_json(
        {
            "total_recommendations": len(recommendations),
            "recommendations": recommendations,
        },
        os.path.join(output_dir, config.OUTPUT_RECOMMENDATIONS_FILE),
    )

    # ── Stage 1 ──
    all_entities, unique_entities = run_stage1(recommendations)

    save_json(
        {
            "total_raw_entities": len(all_entities),
            "total_unique_entities": len(unique_entities),
            "entities": list(unique_entities.values()),
        },
        os.path.join(output_dir, config.OUTPUT_ENTITIES_FILE),
    )

    # ── Stage 2 ──
    umls_client = UMLSClient()
    match_results, matched_cuis, unique_triples = run_stage2(
        unique_entities, umls_client, tui_to_group,
    )

    save_json(
        {
            "total_match_results": len(match_results),
            "matched_count": sum(1 for r in match_results if r["matched"]),
            "total_unique_cuis": len(matched_cuis),
            "match_results": match_results,
            "matched_cuis": matched_cuis,
        },
        os.path.join(output_dir, config.OUTPUT_MATCHED_FILE),
    )

    save_json(
        {
            "total_triples": len(unique_triples),
            "triples": unique_triples,
        },
        os.path.join(output_dir, config.OUTPUT_TRIPLES_FILE),
    )

    # ── Pipeline Summary ──
    elapsed = time.time() - start_time
    summary = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "crest_recommendations": len(recommendations),
        "stage1_raw_entities": len(all_entities),
        "stage1_unique_entities": len(unique_entities),
        "stage2_matched_entities": sum(1 for r in match_results if r["matched"]),
        "stage2_match_rate": round(
            sum(1 for r in match_results if r["matched"])
            / max(len(match_results), 1) * 100, 1,
        ),
        "stage2_unique_cuis": len(matched_cuis),
        "stage2_total_triples": len(unique_triples),
        "umls_api_requests": umls_client.request_count,
    }

    save_json(summary, os.path.join(output_dir, config.OUTPUT_PIPELINE_LOG))

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Time elapsed: {elapsed:.1f}s")
    logger.info(f"  Recommendations processed: {summary['crest_recommendations']}")
    logger.info(f"  Entity candidates (unique): {summary['stage1_unique_entities']}")
    logger.info(f"  UMLS match rate: {summary['stage2_match_rate']}%")
    logger.info(f"  Unique CUIs: {summary['stage2_unique_cuis']}")
    logger.info(f"  UMLS Layer triples: {summary['stage2_total_triples']}")
    logger.info(f"  UMLS API requests: {summary['umls_api_requests']}")
    logger.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Medical KG Pipeline: Stage 1 (Entity Extraction) + Stage 2 (UMLS Layer)"
    )
    parser.add_argument("--xml-dir", type=str, help="Path to CREST xml/ folder")
    parser.add_argument("--primary-dir", type=str, help="Path to CREST primary/ folder")
    parser.add_argument("--semantic-groups", type=str, help="Path to semantic groups .txt")
    parser.add_argument("--umls-key", type=str, help="UMLS REST API key")
    parser.add_argument("--openai-key", type=str, help="OpenAI API key")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--max-recs", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")

    args = parser.parse_args()
    setup_logging(args.log_level)

    run_pipeline(
        xml_dir=args.xml_dir,
        primary_dir=args.primary_dir,
        semantic_groups_file=args.semantic_groups,
        umls_api_key=args.umls_key,
        openai_api_key=args.openai_key,
        output_dir=args.output_dir,
        max_recommendations=args.max_recs,
    )


if __name__ == "__main__":
    main()
