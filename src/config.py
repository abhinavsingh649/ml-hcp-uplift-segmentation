from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "synthetic_hcp_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "uplift_scored.csv"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "xlearner_model.joblib"

# Model hyperparameters
TREEMETHOD = "hist"
N_ESTIMATORS = 300
MAX_DEPTH = 5
LEARNING_RATE = 0.05
