from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_FOLDER = PROJECT_ROOT / "logs"
MODEL_FILES_FOLDER = PROJECT_ROOT / "models"

if not LOGS_FOLDER.exists():
    LOGS_FOLDER.mkdir(parents=True, exist_ok=True)

if not MODEL_FILES_FOLDER.exists():
    MODEL_FILES_FOLDER.mkdir(parents=True, exist_ok=True)

DEFAULT_LOG_FILE = LOGS_FOLDER / "page-parser.log"
