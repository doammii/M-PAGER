"""
CREST Corpus Parser
Extracts recommendation sentences from xml/ and background context from primary/.

CREST structure (Read et al., 2016):
  - xml/   : Recommendation sections with <p|li|span recommendation="grade"> markup
  - primary/ : Full HTML guidelines from guidelines.gov

Role separation:
  - xml/     → Primary input: structured recommendation sentences with strength labels
  - primary/ → Supplementary: background context for reference resolution & additional entities
"""

import os
import logging
from bs4 import BeautifulSoup, Tag

import config

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    """Normalize whitespace (any run → single space) and strip.

    `' '.join(s.split())` is faster than a regex on Python whitespace and
    avoids importing `re` just for this.
    """
    return " ".join(text.split())


def _extract_recommendations_from_xml(xml_path: str) -> list[dict]:
    """
    Parse a single CREST XML file and extract recommendation sentences.

    CREST XML uses:
      <p recommendation="B">...</p>
      <li recommendation="A">...</li>
      <span recommendation="C">...</span>
    """
    results = []

    try:
        with open(xml_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except (IOError, OSError) as e:
        logger.warning(f"Cannot read XML file {xml_path}: {e}")
        return results

    # CREST xml/ files may not be strict XML; use html.parser as fallback
    try:
        soup = BeautifulSoup(content, "lxml-xml")
    except Exception:
        soup = BeautifulSoup(content, "html.parser")

    # Find all elements with a 'recommendation' attribute
    for elem in soup.find_all(attrs={"recommendation": True}):
        text = _clean_text(elem.get_text())
        if len(text) < 10:
            continue

        # Skip <span> children if the parent <p>/<li> also has recommendation attr
        # to avoid double-counting. But keep <span> if it has a DIFFERENT grade.
        if elem.name == "span":
            parent = elem.parent
            if (
                parent
                and isinstance(parent, Tag)
                and parent.get("recommendation") == elem.get("recommendation")
            ):
                continue

        results.append(
            {
                "strength": elem["recommendation"],
                "text": text,
                "tag": elem.name,
            }
        )

    return results


def _extract_context_from_primary(html_path: str, max_chars: int) -> str:
    """
    Parse a CREST primary/ HTML file and extract background text as context.

    Removes navigation, scripts, styles. Returns plain text truncated to max_chars.
    """
    try:
        with open(html_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except (IOError, OSError) as e:
        logger.warning(f"Cannot read primary HTML {html_path}: {e}")
        return ""

    soup = BeautifulSoup(content, "html.parser")

    # Remove non-content elements
    for tag in soup.find_all(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    full_text = _clean_text(soup.get_text(separator="\n"))

    # Truncate to max_chars (try to break at sentence boundary)
    if len(full_text) > max_chars:
        truncated = full_text[:max_chars]
        last_period = truncated.rfind(".")
        if last_period > max_chars * 0.7:
            truncated = truncated[: last_period + 1]
        full_text = truncated

    return full_text


def _resolve_guideline_id(filename: str) -> str:
    """Extract guideline ID from filename (remove extension)."""
    base = os.path.basename(filename)
    # Remove common extensions
    for ext in [".xml", ".html", ".htm", ".xhtml"]:
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    return base


def extract_from_both_sources(
    xml_dir: str = None,
    primary_dir: str = None,
    primary_context_max_chars: int = None,
) -> list[dict]:
    """
    Extract recommendation sentences from xml/ and background context from primary/.

    Returns list of dicts:
        {
            "guideline_id": str,
            "strength": str,         # recommendation grade (A, B, C, ...)
            "text": str,             # recommendation sentence text
            "tag": str,              # HTML tag (p, li, span)
            "guideline_context": str # background text from primary/ HTML
        }
    """
    xml_dir = xml_dir or config.CREST_XML_DIR
    primary_dir = primary_dir or config.CREST_PRIMARY_DIR
    primary_context_max_chars = (
        primary_context_max_chars or config.PRIMARY_CONTEXT_MAX_CHARS
    )

    # ── Step 1: Build guideline context from primary/ ──
    guideline_contexts = {}

    if os.path.isdir(primary_dir):
        for fname in sorted(os.listdir(primary_dir)):
            fpath = os.path.join(primary_dir, fname)
            if not os.path.isfile(fpath):
                continue
            gid = _resolve_guideline_id(fname)
            ctx = _extract_context_from_primary(fpath, primary_context_max_chars)
            if ctx:
                guideline_contexts[gid] = ctx
        logger.info(
            f"Loaded context from {len(guideline_contexts)} primary/ guidelines"
        )
    else:
        logger.warning(f"primary/ directory not found: {primary_dir}")

    # ── Step 2: Extract recommendations from xml/ ──
    all_recommendations = []

    if not os.path.isdir(xml_dir):
        logger.error(f"xml/ directory not found: {xml_dir}")
        return all_recommendations

    xml_files = sorted(
        f for f in os.listdir(xml_dir) if os.path.isfile(os.path.join(xml_dir, f))
    )
    logger.info(f"Processing {len(xml_files)} XML files from xml/")

    for fname in xml_files:
        fpath = os.path.join(xml_dir, fname)
        gid = _resolve_guideline_id(fname)

        recs = _extract_recommendations_from_xml(fpath)

        for rec in recs:
            rec["guideline_id"] = gid
            rec["guideline_context"] = guideline_contexts.get(gid, "")
            all_recommendations.append(rec)

    logger.info(
        f"Extracted {len(all_recommendations)} recommendation sentences "
        f"from {len(xml_files)} guidelines"
    )

    # ── Step 3: Log statistics ──
    strength_counts = {}
    for rec in all_recommendations:
        s = rec["strength"]
        strength_counts[s] = strength_counts.get(s, 0) + 1
    logger.info(f"Strength distribution: {strength_counts}")

    with_context = sum(1 for r in all_recommendations if r["guideline_context"])
    logger.info(
        f"Recommendations with primary/ context: {with_context}/{len(all_recommendations)}"
    )

    return all_recommendations
