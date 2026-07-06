"""Configuration for the NDD Gene Evidence pipeline."""
import os
from dotenv import load_dotenv

APP_ROOT = os.path.dirname(os.path.realpath(__file__))
load_dotenv(os.path.join(APP_ROOT, '.env'))

# --- API Keys ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set. Add it to ddd_agent/.env")

NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")

# --- URLs ---
BASE_URL_NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# --- Paths ---
DATA_DIR = os.path.join(APP_ROOT, 'Data')
PROMPTS_DIR = os.path.join(APP_ROOT, 'prompts')
OUTPUT_DIR = os.path.join(APP_ROOT, 'output')
GENES_FILE = os.path.join(DATA_DIR, 'NDD_genes_novel.txt')
ASHG_FILE = os.path.join(DATA_DIR, 'ASHG_2009_2023_v2.csv')

# --- LLM Settings ---
DEFAULT_MODEL = "claude-haiku-4-5"
LLM_MAX_RETRIES = 5
LLM_INITIAL_DELAY = 2
LLM_MAX_TOKENS = 16384

# --- PubMed Settings ---
PUBMED_COOLDOWN = 0.05

# --- Agent output limits ---
DEFAULT_MAX_PAPERS = 10

# --- Search keywords per evidence category ---
CLINICAL_KEYWORDS = [
    "neurodevelopmental", "autism", "developmental",
    "intellectual disability", "dysmorphia", "autism spectrum disorder",
    "epilepsy", "malformations",
]

MODEL_ORGANISM_KEYWORDS = [
    "knockout", "mouse model", "zebrafish", "xenopus", "model organism",
]

BIOLOGICAL_KEYWORDS = [
    "function", "pathway", "expression", "cell culture", "screening",
]

# --- PubMed retmax per category ---
CLINICAL_RETMAX = 20
MODEL_ORGANISM_RETMAX = 20
BIOLOGICAL_RETMAX = 20

# --- Disease-specific run configurations ---
# Each entry drives a full disease-aware run via the --disorder CLI flag:
#   - genes_file:        gene candidate list for this disease
#   - clinical_keywords: REPLACE the generic CLINICAL_KEYWORDS for PubMed/ASHG clinical search
#   - mouse_phenotypes:  ADDED to the model_organism prompt's priority guidance (search keywords stay generic)
#   - disease_label:     injected into the synthesis prompt in place of "neurodevelopmental disorder"
DISEASES = {
    "asd": {
        "genes_file": os.path.join(DATA_DIR, "ASD_genes_union.txt"),
        "clinical_keywords": [
            "Autism", "Autistic features", "ASD", "repetitive behavior",
            "Asperger syndrome", "Stereotypies", "Rett syndrome", "Rett-like",
        ],
        "mouse_phenotypes": "general developmental delay, social/behavioral phenotypes, brain abnormalities",
        "disease_label": "autism spectrum disorder",
    },
    "chd": {
        "genes_file": os.path.join(DATA_DIR, "CHD_genes.txt"),
        "clinical_keywords": [
            "Diaphragmatic hernia", "septal", "ventricular",
            "Transposition of the great arteries", "congenital heart disease",
            "cardiac anomaly", "Endocardial cushion defect",
            "Hypoplastic left heart syndrome", "aorta valve", "pulmonary valve",
            "Tetralogy of Fallot", "Mitral valve anomaly", "Valve dysplasia",
            "cardiogenesis",
        ],
        "mouse_phenotypes": "general developmental delay, cardiac / lipid / blood vessel phenotypes",
        "disease_label": "congenital heart disease / congenital diaphragmatic hernia",
    },
    "ee": {
        "genes_file": os.path.join(DATA_DIR, "EE_genes.txt"),
        "clinical_keywords": [
            "Encephalopathic epilepsy", "seizures", "EEG abnormalities",
            "encephalopathy", "neonatal seizures", "infantile seizures",
            "spasm", "spasticity",
        ],
        "mouse_phenotypes": "general developmental delay, behavioral / seizure phenotypes, brain abnormalities",
        "disease_label": "epileptic encephalopathy",
    },
    "ndd": {
        # Original NDD behaviour: generic clinical keywords, no extra mouse
        # phenotypes, generic "neurodevelopmental disorder" synthesis label.
        "genes_file": os.path.join(DATA_DIR, "NDD_genes_04_04.txt"),
        "clinical_keywords": list(CLINICAL_KEYWORDS),
        "mouse_phenotypes": "",
        "disease_label": "neurodevelopmental disorder",
    },
}
