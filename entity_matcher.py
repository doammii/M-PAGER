"""
Entity Matcher (Stage 2-A)
Cascading UMLS match: Exact → NormalizedString → Words
with rule-based normalization fallback at each stage.

Design rationale (from research proposal §2.4 Step 2):
  "추출된 entity candidate를 실제 UMLS entity name과 exact match로 전수 비교,
   일치 entity를 seed로 하여 1-hop subgraph를 수집·정합"
"""

import re
import logging
from typing import Optional

from umls_client import UMLSClient

logger = logging.getLogger(__name__)


ABBREVIATION_RULES = {
    "NSCLC": "Non-Small Cell Lung Carcinoma",
    "SCLC": "Small Cell Lung Carcinoma",
    "CKD": "Chronic Kidney Disease",
    "CHF": "Congestive Heart Failure",
    "COPD": "Chronic Obstructive Pulmonary Disease",
    "DM": "Diabetes Mellitus",
    "T2DM": "Diabetes Mellitus, Type 2",
    "T1DM": "Diabetes Mellitus, Type 1",
    "HTN": "Hypertension",
    "MI": "Myocardial Infarction",
    "AMI": "Acute Myocardial Infarction",
    "DVT": "Deep Vein Thrombosis",
    "PE": "Pulmonary Embolism",
    "VTE": "Venous Thromboembolism",
    "CAD": "Coronary Artery Disease",
    "AF": "Atrial Fibrillation",
    "AFIB": "Atrial Fibrillation",
    "RA": "Rheumatoid Arthritis",
    "OA": "Osteoarthritis",
    "SLE": "Systemic Lupus Erythematosus",
    "IBD": "Inflammatory Bowel Disease",
    "IBS": "Irritable Bowel Syndrome",
    "UC": "Ulcerative Colitis",
    "CD": "Crohn Disease",
    "GERD": "Gastroesophageal Reflux Disease",
    "UTI": "Urinary Tract Infection",
    "AKI": "Acute Kidney Injury",
    "ARDS": "Acute Respiratory Distress Syndrome",
    "TIA": "Transient Ischemic Attack",
    "CVA": "Cerebrovascular Accident",
    "PAD": "Peripheral Arterial Disease",
    "OSA": "Obstructive Sleep Apnea",
    "ADHD": "Attention Deficit Hyperactivity Disorder",
    "PTSD": "Post-Traumatic Stress Disorder",
    "ASD": "Autism Spectrum Disorder",
    "MS": "Multiple Sclerosis",
    "PD": "Parkinson Disease",
    "ALS": "Amyotrophic Lateral Sclerosis",
    "HIV": "Human Immunodeficiency Virus",
    "AIDS": "Acquired Immunodeficiency Syndrome",
    "HCV": "Hepatitis C Virus",
    "HBV": "Hepatitis B Virus",
    "HPV": "Human Papillomavirus",
    "RSV": "Respiratory Syncytial Virus",
    "TB": "Tuberculosis",
    "MRSA": "Methicillin-Resistant Staphylococcus aureus",
    "BMI": "Body Mass Index",
    "BP": "Blood Pressure",
    "HR": "Heart Rate",
    "RR": "Respiratory Rate",
    "SpO2": "Oxygen Saturation",
    "HbA1c": "Hemoglobin A1c",
    "LDL": "Low-Density Lipoprotein",
    "HDL": "High-Density Lipoprotein",
    "TG": "Triglycerides",
    "CRP": "C-Reactive Protein",
    "ESR": "Erythrocyte Sedimentation Rate",
    "PSA": "Prostate-Specific Antigen",
    "TSH": "Thyroid Stimulating Hormone",
    "INR": "International Normalized Ratio",
    "GFR": "Glomerular Filtration Rate",
    "eGFR": "Estimated Glomerular Filtration Rate",
    "BUN": "Blood Urea Nitrogen",
    "WBC": "White Blood Cell Count",
    "RBC": "Red Blood Cell Count",
    "Hb": "Hemoglobin",
    "Hct": "Hematocrit",
    "PLT": "Platelet Count",
    "ALT": "Alanine Aminotransferase",
    "AST": "Aspartate Aminotransferase",
    "CT": "Computed Tomography",
    "MRI": "Magnetic Resonance Imaging",
    "PET": "Positron Emission Tomography",
    "CXR": "Chest X-Ray",
    "ECG": "Electrocardiography",
    "EKG": "Electrocardiography",
    "EEG": "Electroencephalography",
    "EMG": "Electromyography",
    "US": "Ultrasonography",
    "TEE": "Transesophageal Echocardiography",
    "TTE": "Transthoracic Echocardiography",
    "ERCP": "Endoscopic Retrograde Cholangiopancreatography",
    "CABG": "Coronary Artery Bypass Grafting",
    "PCI": "Percutaneous Coronary Intervention",
    "TURP": "Transurethral Resection of Prostate",
    "IV": "Intravenous",
    "IM": "Intramuscular",
    "SC": "Subcutaneous",
    "PO": "Oral Administration",
    "ACE": "Angiotensin-Converting Enzyme",
    "ACEi": "Angiotensin-Converting Enzyme Inhibitor",
    "ACE inhibitor": "Angiotensin-Converting Enzyme Inhibitor",
    "ARB": "Angiotensin Receptor Blocker",
    "SSRI": "Selective Serotonin Reuptake Inhibitor",
    "SNRI": "Serotonin-Norepinephrine Reuptake Inhibitor",
    "TCA": "Tricyclic Antidepressant",
    "MAOI": "Monoamine Oxidase Inhibitor",
    "NSAID": "Non-Steroidal Anti-Inflammatory Drug",
    "NSAIDs": "Non-Steroidal Anti-Inflammatory Drug",
    "PPI": "Proton Pump Inhibitor",
    "H2RA": "Histamine-2 Receptor Antagonist",
    "LMWH": "Low Molecular Weight Heparin",
    "UFH": "Unfractionated Heparin",
    "DOAC": "Direct Oral Anticoagulant",
    "ESA": "Erythropoiesis Stimulating Agent",
    "G-CSF": "Granulocyte Colony-Stimulating Factor",
    "GM-CSF": "Granulocyte-Macrophage Colony-Stimulating Factor",
    "TNF": "Tumor Necrosis Factor",
    "IL-6": "Interleukin-6",
    "THC": "Tetrahydrocannabinol",
    "CBD": "Cannabidiol",
    "EGFR": "Epidermal Growth Factor Receptor",
    "HER2": "Human Epidermal Growth Factor Receptor 2",
    "VEGF": "Vascular Endothelial Growth Factor",
    "ALK": "Anaplastic Lymphoma Kinase",
    "BRCA": "BRCA Gene",
    "BRCA1": "BRCA1 Gene",
    "BRCA2": "BRCA2 Gene",
    "KRAS": "KRAS Gene",
    "TP53": "TP53 Gene",
}

COLLOQUIAL_RULES = {
    "blood thinners": "Anticoagulants",
    "blood thinner": "Anticoagulants",
    "heart attack": "Myocardial Infarction",
    "stroke": "Cerebrovascular Accident",
    "high blood pressure": "Hypertension",
    "low blood pressure": "Hypotension",
    "high cholesterol": "Hypercholesterolemia",
    "high blood sugar": "Hyperglycemia",
    "low blood sugar": "Hypoglycemia",
    "sugar diabetes": "Diabetes Mellitus",
    "kidney failure": "Renal Failure",
    "kidney disease": "Kidney Disease",
    "liver failure": "Hepatic Failure",
    "liver disease": "Liver Disease",
    "water pills": "Diuretics",
    "pain killers": "Analgesics",
    "painkillers": "Analgesics",
    "blood clot": "Thrombosis",
    "blood clots": "Thrombosis",
    "heartburn": "Gastroesophageal Reflux",
    "chest pain": "Chest Pain",
    "shortness of breath": "Dyspnea",
    "belly pain": "Abdominal Pain",
    "back pain": "Back Pain",
    "joint pain": "Arthralgia",
    "muscle pain": "Myalgia",
    "nerve pain": "Neuralgia",
}

SUFFIX_PATTERNS = [
    (r"^(\w+)ive\s+patients?$", lambda m: m.group(1) + "ion"),
    (r"^(\w+)ic\s+patients?$", lambda m: m.group(1).rstrip("t") + "es"),
    (r"^(\w+)ous$", lambda m: m.group(1)),
]


class EntityMatcher:
    """
    Maps entity candidates to UMLS CUIs using cascading match + rule-based fallback.

    Match cascade (per candidate term):
      1) Exact match → if group-filtered results exist, return
      2) NormalizedString match → if group-filtered results exist, return
    After all candidates exhausted:
      3) Words match (loosest, on normalized_form and surface_form)

    CUI concept lookups are cached to avoid redundant API calls.
    """

    def __init__(self, umls_client: UMLSClient, tui_to_group: dict):
        self.client = umls_client
        self.tui_to_group = tui_to_group
        self._concept_cache: dict[str, Optional[dict]] = {}

    def _get_concept_cached(self, cui: str) -> Optional[dict]:
        """Retrieve concept info with caching to reduce API calls."""
        if cui not in self._concept_cache:
            self._concept_cache[cui] = self.client.get_concept(cui)
        return self._concept_cache[cui]

    def match_entity(self, entity: dict) -> dict:
        """
        Match a single entity candidate to UMLS CUI(s).

        Returns dict with: entity, matched, matches, match_type
        """
        surface = entity.get("surface_form", "").strip()
        normalized = entity.get("normalized_form", surface).strip()
        target_group = entity.get("semantic_group", None)

        candidates = self._generate_candidates(surface, normalized)

        # ── Cascade: Exact → Normalized (per candidate) ──
        for term in candidates:
            results = self.client.search_exact(term)
            if results:
                filtered = self._filter_by_group(results, target_group)
                if filtered:
                    return self._build_result(
                        entity, filtered, match_type="exact", match_query=term,
                    )
                # group 불일치 → 다음 candidate로 계속 시도

            results = self.client.search_normalized(term)
            if results:
                filtered = self._filter_by_group(results, target_group)
                if filtered:
                    return self._build_result(
                        entity, filtered[:5],
                        match_type="normalizedString", match_query=term,
                    )

        # ── Fallback: Words match (loosest) ──
        for term in [normalized, surface]:
            if not term:
                continue
            results = self.client.search_words(term)
            if results:
                filtered = self._filter_by_group(results, target_group)
                if filtered:
                    return self._build_result(
                        entity, filtered[:5],
                        match_type="words", match_query=term,
                    )

        # ── Last resort: Exact without group filter ──
        for term in candidates[:2]:
            results = self.client.search_exact(term)
            if results:
                return self._build_result(
                    entity, results[:3],
                    match_type="exact_unfiltered", match_query=term,
                )

        return {
            "entity": entity,
            "matched": False,
            "matches": [],
            "match_type": "none",
        }

    def _generate_candidates(self, surface: str, normalized: str) -> list[str]:
        """Generate match candidate terms in priority order."""
        candidates = []
        seen = set()

        def add(term: str):
            t = term.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                candidates.append(t)

        add(normalized)
        add(surface)

        for key in [surface.upper().strip(), surface.strip()]:
            if key in ABBREVIATION_RULES:
                add(ABBREVIATION_RULES[key])

        lower = surface.lower().strip()
        if lower in COLLOQUIAL_RULES:
            add(COLLOQUIAL_RULES[lower])

        for pattern, transform in SUFFIX_PATTERNS:
            m = re.match(pattern, surface, re.IGNORECASE)
            if m:
                try:
                    add(transform(m))
                except Exception:
                    pass

        if "(" in surface:
            without_paren = re.sub(r"\s*\([^)]*\)", "", surface).strip()
            add(without_paren)
            for content in re.findall(r"\(([^)]+)\)", surface):
                add(content.strip())

        if "-" in surface:
            add(surface.replace("-", " "))
            add(surface.replace("-", ""))

        return candidates

    def _filter_by_group(
        self, results: list[dict], target_group: Optional[str]
    ) -> list[dict]:
        """Filter search results by target semantic group (uses concept cache)."""
        if not target_group:
            return results

        filtered = []
        for r in results:
            cui = r.get("ui", "")
            if not cui.startswith("C"):
                continue

            concept = self._get_concept_cached(cui)
            if not concept:
                continue

            sem_types = concept.get("semanticTypes", [])
            for st in sem_types:
                tui = self._extract_tui(st)
                if tui and self.tui_to_group.get(tui) == target_group:
                    r["_matched_semantic_types"] = sem_types
                    r["_concept_name"] = concept.get("name", "")
                    filtered.append(r)
                    break

        return filtered

    def _extract_tui(self, semantic_type_obj: dict) -> str:
        """Extract TUI from a semantic type object's URI."""
        uri = semantic_type_obj.get("uri", "")
        if "/TUI/" in uri:
            return uri.split("/TUI/")[-1]
        return ""

    def _build_result(
        self, entity: dict, matches: list[dict],
        match_type: str, match_query: str,
    ) -> dict:
        """Build a match result dict."""
        processed = []
        for m in matches:
            processed.append({
                "cui": m.get("ui", ""),
                "name": m.get("name", m.get("_concept_name", "")),
                "root_source": m.get("rootSource", ""),
                "match_type": match_type,
                "match_query": match_query,
                "semantic_types": m.get("_matched_semantic_types", []),
            })

        return {
            "entity": entity,
            "matched": True,
            "matches": processed,
            "match_type": match_type,
        }


def match_entities_batch(
    unique_entities: dict,
    umls_client: UMLSClient,
    tui_to_group: dict,
    progress_interval: int = 50,
) -> tuple[list[dict], dict]:
    """Run UMLS matching on all unique entities."""
    matcher = EntityMatcher(umls_client, tui_to_group)

    match_results = []
    matched_cuis = {}

    entities_list = list(unique_entities.values())

    for i, entity in enumerate(entities_list):
        result = matcher.match_entity(entity)
        match_results.append(result)

        if result["matched"]:
            for m in result["matches"]:
                cui = m["cui"]
                if cui.startswith("C") and cui not in matched_cuis:
                    matched_cuis[cui] = {
                        "cui": cui,
                        "name": m["name"],
                        "match_type": m["match_type"],
                        "root_source": m.get("root_source", ""),
                    }

        if (i + 1) % progress_interval == 0:
            matched_so_far = sum(1 for r in match_results if r["matched"])
            logger.info(
                f"  Matching progress: {i + 1}/{len(entities_list)} entities, "
                f"{matched_so_far} matched, {len(matched_cuis)} unique CUIs"
            )

    matched_count = sum(1 for r in match_results if r["matched"])
    total = len(match_results)
    pct = (matched_count / total * 100) if total > 0 else 0

    logger.info(
        f"Matching complete: {matched_count}/{total} ({pct:.1f}%) entities matched, "
        f"{len(matched_cuis)} unique CUIs, "
        f"{len(matcher._concept_cache)} concept lookups cached"
    )

    type_dist = {}
    for r in match_results:
        mt = r["match_type"]
        type_dist[mt] = type_dist.get(mt, 0) + 1
    logger.info(f"Match type distribution: {type_dist}")

    return match_results, matched_cuis
