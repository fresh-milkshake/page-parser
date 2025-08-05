from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_FOLDER = PROJECT_ROOT / "logs"
MODEL_FILES_FOLDER = PROJECT_ROOT / "models"

if not LOGS_FOLDER.exists():
    LOGS_FOLDER.mkdir(parents=True, exist_ok=True)

if not MODEL_FILES_FOLDER.exists():
    MODEL_FILES_FOLDER.mkdir(parents=True, exist_ok=True)

current_datetime = datetime.now().strftime(r"%Y-%m-%d")
DEFAULT_LOG_FILE = LOGS_FOLDER / f"{current_datetime}.log"
DEFAULT_SETTINGS_FILE = PROJECT_ROOT / "settings.toml"
