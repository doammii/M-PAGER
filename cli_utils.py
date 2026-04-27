"""
Shared CLI helpers used by pipeline.py.

Centralizes logging setup, JSON I/O, and prerequisite-file checks so each
stage handler stays a thin wrapper around its module's core logic.
"""

import json
import logging
import os
import sys

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """Configure root logging for stage runners (idempotent across stages)."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_json(filepath: str) -> dict:
    """Load JSON from disk (UTF-8)."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, filepath: str):
    """Save data as UTF-8 JSON, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Saved: {filepath}")


def require_files(paths: dict, hint: str = ""):
    """Verify all named input files exist; sys.exit(1) if any are missing.

    `paths` is {label: filepath}; the label is shown in the error message so
    the user knows which prerequisite stage failed to produce its output.
    """
    missing = [
        f"  - {label}: {path}"
        for label, path in paths.items()
        if not os.path.isfile(path)
    ]
    if missing:
        logger.error("Required input file(s) not found:")
        for m in missing:
            logger.error(m)
        if hint:
            logger.error(hint)
        sys.exit(1)
