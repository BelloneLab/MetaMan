import os
from pathlib import Path

# defaults (change if you like; the UI will remember your changes)
DEFAULT_DATA_ROOT = os.path.join(os.path.expanduser("~"), "MetaManData")
DEFAULT_RAW_ROOT = os.path.join(DEFAULT_DATA_ROOT, "raw")
DEFAULT_PROCESSED_ROOT = os.path.join(DEFAULT_DATA_ROOT, "processed")

APP_NAME = "neuro_meta_app_qt"
SETTINGS_FILE = Path.home() / f".{APP_NAME}_settings.json"

SESSION_META_JSON = "metadata.json"
SESSION_META_CSV  = "metadata.csv"
SESSION_META_H5   = "metadata.h5"

PROJECT_INFO_JSON = "project_info.json"
EXPERIMENT_INFO_JSON = "experiment_info.json"
SUBJECT_INFO_JSON = "subject_info.json"
ANIMAL_INFO_JSON = SUBJECT_INFO_JSON  # backward compatibility

APP_TITLE = "MetaMan"
WINDOW_GEOMETRY = (1250, 780)

COPY_CHUNK_BYTES = 4 * 1024 * 1024  # 4MB

def normpath(p: str) -> str:
    return os.path.normpath(p)
