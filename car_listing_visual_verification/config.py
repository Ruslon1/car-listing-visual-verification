import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"


def _env_path(name: str, default: Path) -> Path:
    """Read Path from env var and fallback to default."""
    value = os.getenv(name)
    if value:
        return Path(value).expanduser()
    return default


DROM_MANIFEST_FILE = PROCESSED_DATA_DIR / "manifest.parquet"
DROM_CLASS_MAPPING_FILE = PROCESSED_DATA_DIR / "class_mapping.parquet"

HF_RELEASE_DIR = _env_path("CLVV_HF_RELEASE_DIR", PROCESSED_DATA_DIR / "hf_release")
HF_RELEASE_IMAGES_DIR = HF_RELEASE_DIR / "images"
HF_RELEASE_README_FILE = HF_RELEASE_DIR / "README.md"
HF_RELEASE_STATS_FILE = HF_RELEASE_DIR / "release_stats.json"

HF_RELEASE_METADATA_FILE = HF_RELEASE_DIR / "metadata.parquet"
HF_RELEASE_CLASS_MAPPING_FILE = HF_RELEASE_DIR / "class_mapping.parquet"
HF_RELEASE_TRAIN_FILE = HF_RELEASE_DIR / "train.parquet"
HF_RELEASE_VALIDATION_FILE = HF_RELEASE_DIR / "validation.parquet"
HF_RELEASE_TEST_FILE = HF_RELEASE_DIR / "test.parquet"

HF_RELEASE_SPLIT_FILES = {
    "train": HF_RELEASE_TRAIN_FILE,
    "validation": HF_RELEASE_VALIDATION_FILE,
    "test": HF_RELEASE_TEST_FILE,
}


def hf_release_split_file(split: str) -> Path:
    split_key = split.lower().strip()
    if split_key not in HF_RELEASE_SPLIT_FILES:
        options = ", ".join(HF_RELEASE_SPLIT_FILES)
        raise ValueError(f"Unknown split '{split}'. Expected one of: {options}")
    return HF_RELEASE_SPLIT_FILES[split_key]

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
