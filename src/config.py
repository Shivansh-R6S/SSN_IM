# src/config.py
import os

# ── PATHS ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOG_DIR    = os.path.join(BASE_DIR, "logs")

FILE_PATH  = os.path.join(DATA_DIR, "soc-sign-Slashdot081106.txt")

# ── OUTPUT SUBDIRECTORIES ────────────────────────────────────
COMMUNITY_DIR  = os.path.join(OUTPUT_DIR, "community")
SCORING_DIR    = os.path.join(OUTPUT_DIR, "scoring")
SEEDS_DIR      = os.path.join(OUTPUT_DIR, "seeds")
EVAL_DIR       = os.path.join(OUTPUT_DIR, "evaluation")

# ── ALGORITHM PARAMETERS ─────────────────────────────────────
N_CLUSTERS   = 10       # number of communities for spectral clustering
RANDOM_SEED  = 42       # for reproducibility across all steps
SEED_BUDGET  = 10       # k — number of seed nodes to select
DECAY_ALPHA  = 0.5      # SAP attenuation parameter
SIM_RUNS     = 100      # Monte Carlo simulation runs for SAP
SCORE_THRESH = 0.0      # SAP score threshold to count a node as activated
GIANT_THRESH = 0.005    # PageRank threshold for sleeping giant detection
SEED_BUDGET  = 24

# ── CREATE DIRECTORIES IF THEY DON'T EXIST ───────────────────
for d in [DATA_DIR, COMMUNITY_DIR, SCORING_DIR, SEEDS_DIR, EVAL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)