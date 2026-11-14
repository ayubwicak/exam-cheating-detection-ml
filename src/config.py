"""Konfigurasi global untuk pipeline eksperimen."""

from pathlib import Path

SEED = 42
N_SPLITS = 5

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

TARGET = "is_cheating"
ID_COLUMN = "student_id"


def ensure_output_dir(path: Path = OUTPUT_DIR) -> Path:
    """Buat folder output jika belum ada."""
    path.mkdir(parents=True, exist_ok=True)
    return path
