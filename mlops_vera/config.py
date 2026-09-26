from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import yaml

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# Pipeline data locations (outputs of the DVC stages in dvc.yaml)
RAW_DEFACTIFY_DIR = RAW_DATA_DIR / "defactify"
PREPROCESSED_DIR = PROCESSED_DATA_DIR / "defactify_224"
SPLITS_DIR = PROCESSED_DATA_DIR / "splits"

# Single source of truth for pipeline hyper-parameters (tracked by DVC)
PARAMS_PATH = PROJ_ROOT / "params.yaml"


def load_params(section: str | None = None) -> dict:
    """Read params.yaml, optionally returning a single section."""
    with open(PARAMS_PATH, encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params[section] if section else params


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
