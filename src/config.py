from pathlib import Path

THIS_FILE = Path(__file__).resolve()

SRC_DIR = THIS_FILE.parent
BASE_DIR = SRC_DIR.parent
DATA_DIR = BASE_DIR / "data" / "raw"