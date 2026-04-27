"""
All 127 UMLS Semantic Types organized by 15 Semantic Groups.
Each type includes 1-2 example entities for LLM few-shot prompting.

Source: UMLS Semantic Network (SemGroups.txt)
Reference: McCray et al. (2001), Bodenreider (2004)
"""

# Format: { "GROUP_ABBR": { "group_name": str, "types": [ (TUI, type_name, [examples]) ] } }

SEMANTIC_GROUPS_WITH_EXAMPLES = {
    "ACTI": {
        "group_name": "Activities & Behaviors",
        "types": [
            ("T052", "Activity", ["physical activity", "exercise"]),
            ("T053", "Behavior", ["smoking", "alcohol use"]),
            ("T056", "Daily or Recreational Activity", ["walking", "swimming"]),
            ("T051", "Event", ["exposure", "relapse"]),
            ("T064", "Governmental or Regulatory Activity", ["FDA approval", "drug regulation"]),
            ("T055", "Individual Behavior", ["medication adherence", "compliance"]),
            ("T066", "Machine Activity", ["mechanical ventilation", "dialysis cycle"]),
            ("T057", "Occupational Activity", ["nursing care", "patient counseling"]),
            ("T054", "Social Behavior", ["social isolation", "caregiver support"]),
        ],
    },
    "ANAT": {
        "group_name": "Anatomy",
        "types": [
            ("T017", "Anatomical Structure", ["joint", "cardiac valve"]),
            ("T029", "Body Location or Region", ["abdomen", "thorax"]),
            ("T023", "Body Part, Organ, or Organ Component", ["liver", "kidney"]),
            ("T030", "Body Space or Junction", ["pleural cavity", "peritoneal space"]),
            ("T031", "Body Substance", ["blood", "cerebrospinal fluid"]),
            ("T022", "Body System", ["cardiovascular system", "nervous system"]),
            ("T025", "Cell", ["lymphocyte", "neutrophil"]),
            ("T026", "Cell Component", ["mitochondria", "cell membrane"]),
            ("T018", "Embryonic Structure", ["neural tube", "fetal heart"]),
            ("T021", "Fully Formed Anatomical Structure", ["skeleton", "vascular tree"]),
            ("T024", "Tissue", ["mucosa", "cartilage"]),
        ],
    },
    "CHEM": {
        "group_name": "Chemicals & Drugs",
        "types": [
            ("T116", "Amino Acid, Peptide, or Protein", ["insulin", "albumin"]),
            ("T195", "Antibiotic", ["amoxicillin", "vancomycin"]),
            ("T123", "Biologically Active Substance", ["cytokine", "growth factor"]),
            ("T122", "Biomedical or Dental Material", ["bone cement", "dental amalgam"]),
            ("T103", "Chemical", ["sodium chloride", "glucose"]),
            ("T120", "Chemical Viewed Functionally", ["antioxidant", "surfactant"]),
            ("T104", "Chemical Viewed Structurally", ["steroid", "polysaccharide"]),
            ("T200", "Clinical Drug", ["aspirin 81mg tablet", "metformin 500mg"]),
            ("T196", "Element, Ion, or Isotope", ["potassium", "calcium ion"]),
            ("T126", "Enzyme", ["angiotensin-converting enzyme", "thrombin"]),
            ("T131", "Hazardous or Poisonous Substance", ["lead", "asbestos"]),
            ("T125", "Hormone", ["estrogen", "cortisol"]),
            ("T129", "Immunologic Factor", ["interferon", "monoclonal antibody"]),
            ("T130", "Indicator, Reagent, or Diagnostic Aid", ["contrast agent", "troponin assay"]),
            ("T197", "Inorganic Chemical", ["calcium carbonate", "iron oxide"]),
            ("T114", "Nucleic Acid, Nucleoside, or Nucleotide", ["DNA", "mRNA"]),
            ("T109", "Organic Chemical", ["ethanol", "methanol"]),
            ("T121", "Pharmacologic Substance", ["metformin", "warfarin"]),
            ("T192", "Receptor", ["EGFR", "HER2 receptor"]),
            ("T127", "Vitamin", ["vitamin D", "folic acid"]),
        ],
    },
    "CONC": {
        "group_name": "Concepts & Ideas",
        "types": [
            ("T185", "Classification", ["TNM staging", "NYHA classification"]),
            ("T077", "Conceptual Entity", ["risk", "prognosis"]),
            ("T169", "Functional Concept", ["contraindication", "indication"]),
            ("T102", "Group Attribute", ["ethnicity", "socioeconomic status"]),
            ("T078", "Idea or Concept", ["informed consent", "palliative care"]),
            ("T170", "Intellectual Product", ["clinical guideline", "treatment protocol"]),
            ("T171", "Language", ["medical terminology", "clinical nomenclature"]),
            ("T080", "Qualitative Concept", ["Stage III", "severe"]),
            ("T081", "Quantitative Concept", ["dosage", "threshold value"]),
            ("T089", "Regulation or Law", ["prescription regulation", "informed consent law"]),
            ("T082", "Spatial Concept", ["proximal", "bilateral"]),
            ("T079", "Temporal Concept", ["acute", "chronic"]),
        ],
    },
    "DEVI": {
        "group_name": "Devices",
        "types": [
            ("T203", "Drug Delivery Device", ["insulin pump", "metered-dose inhaler"]),
            ("T074", "Medical Device", ["coronary stent", "pacemaker"]),
            ("T075", "Research Device", ["flow cytometer", "spectrophotometer"]),
        ],
    },
    "DISO": {
        "group_name": "Disorders",
        "types": [
            ("T020", "Acquired Abnormality", ["hernia", "fistula"]),
            ("T190", "Anatomical Abnormality", ["stenosis", "aneurysm"]),
            ("T049", "Cell or Molecular Dysfunction", ["EGFR mutation", "chromosomal deletion"]),
            ("T019", "Congenital Abnormality", ["spina bifida", "cleft palate"]),
            ("T047", "Disease or Syndrome", ["diabetes mellitus", "hypertension"]),
            ("T050", "Experimental Model of Disease", ["xenograft tumor model", "knockout mouse model"]),
            ("T033", "Finding", ["elevated blood pressure", "positive blood culture"]),
            ("T037", "Injury or Poisoning", ["hip fracture", "drug overdose"]),
            ("T048", "Mental or Behavioral Dysfunction", ["major depression", "anxiety disorder"]),
            ("T191", "Neoplastic Process", ["non-small cell lung cancer", "melanoma"]),
            ("T046", "Pathologic Function", ["inflammation", "ischemia"]),
            ("T184", "Sign or Symptom", ["fever", "dyspnea"]),
        ],
    },
    "GENE": {
        "group_name": "Genes & Molecular Sequences",
        "types": [
            ("T087", "Amino Acid Sequence", ["BRCA1 protein sequence", "p53 protein"]),
            ("T088", "Carbohydrate Sequence", ["glycan structure", "oligosaccharide chain"]),
            ("T028", "Gene or Genome", ["EGFR gene", "BRCA1"]),
            ("T085", "Molecular Sequence", ["promoter region", "signal peptide sequence"]),
            ("T086", "Nucleotide Sequence", ["codon 12 mutation", "microsatellite repeat"]),
        ],
    },
    "GEOG": {
        "group_name": "Geographic Areas",
        "types": [
            ("T083", "Geographic Area", ["sub-Saharan Africa", "Southeast Asia"]),
        ],
    },
    "LIVB": {
        "group_name": "Living Beings",
        "types": [
            ("T100", "Age Group", ["elderly", "neonate"]),
            ("T011", "Amphibian", ["frog", "salamander"]),
            ("T008", "Animal", ["mouse", "rat"]),
            ("T194", "Archaeon", ["methanogenic archaeon", "halophilic archaeon"]),
            ("T007", "Bacterium", ["Staphylococcus aureus", "Escherichia coli"]),
            ("T012", "Bird", ["chicken", "pigeon"]),
            ("T204", "Eukaryote", ["Saccharomyces cerevisiae", "Plasmodium"]),
            ("T099", "Family Group", ["first-degree relative", "sibling"]),
            ("T013", "Fish", ["zebrafish", "salmon"]),
            ("T004", "Fungus", ["Candida albicans", "Aspergillus"]),
            ("T096", "Group", ["control group", "study cohort"]),
            ("T016", "Human", ["patient", "clinician"]),
            ("T015", "Mammal", ["primate", "canine"]),
            ("T001", "Organism", ["pathogen", "commensal organism"]),
            ("T101", "Patient or Disabled Group", ["diabetic patient", "immunocompromised patient"]),
            ("T002", "Plant", ["Ginkgo biloba", "St. John's wort"]),
            ("T098", "Population Group", ["pregnant women", "African American"]),
            ("T097", "Professional or Occupational Group", ["nurse", "surgeon"]),
            ("T014", "Reptile", ["turtle", "lizard"]),
            ("T010", "Vertebrate", ["vertebrate model", "mammalian host"]),
            ("T005", "Virus", ["HIV", "hepatitis B virus"]),
        ],
    },
    "OBJC": {
        "group_name": "Objects",
        "types": [
            ("T071", "Entity", ["medical record", "specimen"]),
            ("T168", "Food", ["dairy products", "whole grains"]),
            ("T073", "Manufactured Object", ["catheter", "suture material"]),
            ("T072", "Physical Object", ["prosthesis", "implant"]),
            ("T167", "Substance", ["allergen", "carcinogen"]),
        ],
    },
    "OCCU": {
        "group_name": "Occupations",
        "types": [
            ("T091", "Biomedical Occupation or Discipline", ["oncology", "cardiology"]),
            ("T090", "Occupation or Discipline", ["primary care medicine", "general surgery"]),
        ],
    },
    "ORGA": {
        "group_name": "Organizations",
        "types": [
            ("T093", "Health Care Related Organization", ["hospital", "outpatient clinic"]),
            ("T092", "Organization", ["WHO", "CDC"]),
            ("T094", "Professional Society", ["American Heart Association", "ASCO"]),
            ("T095", "Self-help or Relief Organization", ["support group", "patient advocacy group"]),
        ],
    },
    "PHEN": {
        "group_name": "Phenomena",
        "types": [
            ("T038", "Biologic Function", ["immune response", "metabolism"]),
            ("T069", "Environmental Effect of Humans", ["pollution exposure", "deforestation impact"]),
            ("T068", "Human-caused Phenomenon or Process", ["radiation exposure", "noise pollution"]),
            ("T034", "Laboratory or Test Result", ["positive culture result", "elevated serum creatinine"]),
            ("T070", "Natural Phenomenon or Process", ["aging", "wound healing"]),
            ("T067", "Phenomenon or Process", ["drug interaction", "adverse drug reaction"]),
        ],
    },
    "PHYS": {
        "group_name": "Physiology",
        "types": [
            ("T043", "Cell Function", ["apoptosis", "cell proliferation"]),
            ("T201", "Clinical Attribute", ["blood pressure", "heart rate"]),
            ("T045", "Genetic Function", ["gene expression", "DNA repair"]),
            ("T041", "Mental Process", ["cognition", "memory"]),
            ("T044", "Molecular Function", ["enzyme catalysis", "receptor binding"]),
            ("T032", "Organism Attribute", ["body weight", "gestational age"]),
            ("T040", "Organism Function", ["respiration", "digestion"]),
            ("T042", "Organ or Tissue Function", ["renal function", "cardiac output"]),
            ("T039", "Physiologic Function", ["hemostasis", "thermoregulation"]),
        ],
    },
    "PROC": {
        "group_name": "Procedures",
        "types": [
            ("T060", "Diagnostic Procedure", ["MRI scan", "biopsy"]),
            ("T065", "Educational Activity", ["patient education", "health literacy program"]),
            ("T058", "Health Care Activity", ["screening", "triage"]),
            ("T059", "Laboratory Procedure", ["blood culture", "urinalysis"]),
            ("T063", "Molecular Biology Research Technique", ["PCR", "gene sequencing"]),
            ("T062", "Research Activity", ["clinical trial", "cohort study"]),
            ("T061", "Therapeutic or Preventive Procedure", ["chemotherapy", "vaccination"]),
        ],
    },
}


def build_prompt_semantic_section() -> str:
    """
    Build the semantic types section for the LLM extraction prompt.
    Includes ALL 127 types with 1-2 example entities per type.
    """
    lines = []
    lines.append("=== UMLS SEMANTIC TYPES REFERENCE ===")
    lines.append("Extract entities matching ANY of the following types.\n")

    for group_abbr, group_data in SEMANTIC_GROUPS_WITH_EXAMPLES.items():
        group_name = group_data["group_name"]
        lines.append(f"[{group_abbr}] {group_name}")

        for tui, type_name, examples in group_data["types"]:
            examples_str = ", ".join(f'"{e}"' for e in examples)
            lines.append(f"  - {tui} {type_name}  (e.g. {examples_str})")

        lines.append("")

    return "\n".join(lines)


def load_semantic_groups_from_file(filepath: str) -> tuple[dict, dict]:
    """
    Load TUI mappings from the UMLS semantic groups file.
    File format: GROUP_ABBR|Group Name|TUI|Semantic Type Name

    Returns: (tui_to_group, tui_to_name)
    """
    tui_to_group = {}
    tui_to_name = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 4:
                    group_abbr = parts[0]
                    tui = parts[2]
                    type_name = parts[3]
                    tui_to_group[tui] = group_abbr
                    tui_to_name[tui] = type_name
    except FileNotFoundError:
        # Fallback: build from in-memory data
        for group_abbr, group_data in SEMANTIC_GROUPS_WITH_EXAMPLES.items():
            for tui, type_name, _ in group_data["types"]:
                tui_to_group[tui] = group_abbr
                tui_to_name[tui] = type_name

    return tui_to_group, tui_to_name
