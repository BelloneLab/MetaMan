import json
import os
from typing import Dict, Optional, List
from .config import (
    SESSION_META_JSON, SESSION_META_CSV, SESSION_META_H5,
    PROJECT_INFO_JSON, EXPERIMENT_INFO_JSON, SUBJECT_INFO_JSON, ANIMAL_INFO_JSON
)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_json(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_json(path: str, data: Dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_session_triplet(session_dir: str, meta: Dict, logger=None):
    # metadata.json is the single source of truth; the .csv/.h5 are *derived*
    # snapshots regenerated from it on every save and tagged as such so a stale
    # export can never be mistaken for authoritative data.
    from datetime import datetime
    save_json(os.path.join(session_dir, SESSION_META_JSON), meta)
    if logger: logger(f"Saved {SESSION_META_JSON}")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # CSV
    try:
        import pandas as pd
        row = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in meta.items()}
        row["_derived_from"] = SESSION_META_JSON
        row["_generated_at"] = generated_at
        pd.DataFrame([row]).to_csv(os.path.join(session_dir, SESSION_META_CSV), index=False)
        if logger: logger(f"Saved {SESSION_META_CSV}")
    except Exception as e:
        if logger: logger(f"[warn] CSV save skipped ({e})")
    # H5
    try:
        import h5py
        dt = h5py.string_dtype(encoding="utf-8")
        with h5py.File(os.path.join(session_dir, SESSION_META_H5), "w") as h5:
            g = h5.create_group("metadata")
            for k, v in meta.items():
                if isinstance(v, (dict, list)):
                    g.create_dataset(f"{k}_json", data=json.dumps(v, ensure_ascii=False), dtype=dt)
                else:
                    g.create_dataset(k, data="" if v is None else str(v), dtype=dt)
            g.create_dataset("all_json", data=json.dumps(meta, ensure_ascii=False), dtype=dt)
            g.create_dataset("_derived_from", data=SESSION_META_JSON, dtype=dt)
            g.create_dataset("_generated_at", data=generated_at, dtype=dt)
        if logger: logger(f"Saved {SESSION_META_H5}")
    except Exception as e:
        if logger: logger(f"[warn] H5 save skipped ({e})")

def load_session_metadata(session_dir: str) -> Optional[Dict]:
    return load_json(os.path.join(session_dir, SESSION_META_JSON))

def save_project_info(project_dir: str, info: Dict):
    save_json(os.path.join(project_dir, PROJECT_INFO_JSON), info)

def load_project_info(project_dir: str) -> Dict:
    return load_json(os.path.join(project_dir, PROJECT_INFO_JSON)) or {}

# Sidecar that records a project's folder-structure schema *inside* the project
# folder, so the schema travels with the data (e.g. to a server) and the Server
# browser can render the real hierarchy instead of guessing the default.
STRUCTURE_SIDECAR = "_metaman_structure.json"

def save_structure_sidecar(project_dir: str, schema: Dict):
    try:
        save_json(os.path.join(project_dir, STRUCTURE_SIDECAR), schema or {})
    except Exception:
        pass

def load_structure_sidecar(project_dir: str) -> Optional[Dict]:
    return load_json(os.path.join(project_dir, STRUCTURE_SIDECAR))

def save_experiment_info(experiment_dir: str, info: Dict):
    save_json(os.path.join(experiment_dir, EXPERIMENT_INFO_JSON), info)

def load_experiment_info(experiment_dir: str) -> Dict:
    return load_json(os.path.join(experiment_dir, EXPERIMENT_INFO_JSON)) or {}

def save_subject_info(subject_dir: str, info: Dict):
    save_json(os.path.join(subject_dir, SUBJECT_INFO_JSON), info)

def load_subject_info(subject_dir: str) -> Dict:
    # fallback to legacy filename for compatibility
    return (
        load_json(os.path.join(subject_dir, SUBJECT_INFO_JSON))
        or load_json(os.path.join(subject_dir, ANIMAL_INFO_JSON))
        or {}
    )

def save_animal_info(animal_dir: str, info: Dict):
    save_subject_info(animal_dir, info)

def load_animal_info(animal_dir: str) -> Dict:
    return load_subject_info(animal_dir)

def list_projects(raw_root: str):
    try:
        return sorted([d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, d))])
    except Exception:
        return []

def list_experiments(project_dir: str):
    try:
        return sorted([d for d in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, d))])
    except Exception:
        return []

def list_subjects(experiment_dir: str):
    try:
        return sorted([d for d in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, d))])
    except Exception:
        return []

def list_sessions(subject_dir: str):
    try:
        return sorted([d for d in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, d))])
    except Exception:
        return []
