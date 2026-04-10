# MetaMan

<div align="center">

**Neuro data organization without the folder chaos**

[![Python](https://img.shields.io/badge/Python-3.10%2B-2f6db3)](https://www.python.org/)
[![UI](https://img.shields.io/badge/UI-PySide6-0e7a5f)](https://doc.qt.io/qtforpython/)
[![Data](https://img.shields.io/badge/Data-CSV%20%7C%20HDF5-8a4f7d)](#)

</div>

---

## Why MetaMan?

MetaMan helps you manage neuroscience project metadata and keep raw/processed data in a clean hierarchy:

```text
data_root/
  raw/
    <project>/<experiment>/<subject>/<session>/...
  processed/
    <project>/<experiment>/<subject>/<session>/...
```

It is built for fast navigation, safe copy workflows, metadata consistency, and reproducible structure.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch

```bash
python run_app.py
```

---

## Feature Tour

### Navigation tab
- Browse project -> experiment -> subject -> session
- View and edit metadata at all hierarchy levels
- Load subject metadata from CSV for one or multiple subjects
- Copy/open paths quickly

### Recording tab
- Create and update recording/session metadata
- Navigate existing project hierarchy via dropdowns
- Update file list and metadata triplet outputs (`json/csv/h5`)

### Preprocessing tab
- Track preprocessing steps and completion status
- Store step parameters and comments
- Import parameters from CSV/JSON
- Attach per-step results folders

### Data reorganizer tab
- Load metadata plans (`csv/tsv/xlsx`)
- Map columns (`subject_id`, `session_id`, `trial_id`, custom fields)
- Match files using deterministic keys
- Scan **multiple raw and processed source roots**
- Dry run by default (safe mode)
- Execute copy with overwrite policy controls
- Generate match report, run log, and session/subject metadata outputs

### Backup tools
- Manual backup to **Server**, **External HDD**, or **Both**
- Scheduled daily backups per project
- Optional experiment-level backup selection
- Last-used backup roots and schedules persisted

---

## Data Reorganizer Outputs

MetaMan writes:
- `experiment_plan_normalized.csv`
- `match_report.csv`
- `run_log.txt`
- `subject_metadata.csv` and `subject_metadata.h5`
- `session_metadata.csv` and `session_metadata.h5`

Output roots follow:
- `target_raw_root/<project>/<experiment>/...`
- `target_processed_root/<project>/<experiment>/...`

---

## Design Principles

- Safe by default: dry run + no blind overwrite
- Transparent operations: logs, preview tables, match reports
- Responsive UI: background worker threads for long operations
- Deterministic matching: reproducible plan-to-file mapping

---

## Project Layout

```text
MetaMan/
  main.py
  config.py
  state.py
  io_ops.py
  tabs/
    navigation_tab.py
    recording_tab.py
    preprocessing_tab.py
    data_reorganizer_tab.py
  services/
    data_reorganizer.py
    file_scanner.py
    server_sync.py
    search_service.py
```

---

## Tips

- Keep one canonical `data_root` with `raw/` and `processed/` subfolders.
- Use Data reorganizer in dry run first, then execute.
- Save/load reorganizer configs for recurring pipelines.
- Prefer explicit `session_id` in plans when possible.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'PySide6'`

Install dependencies:

```bash
pip install -r requirements.txt
```

### Old drive path errors (for example `B:\` not found)

MetaMan now falls back to a safe local root (`~/MetaManData`).  
You can also set your preferred root from **File -> Set Data Root...**.

---

## One-line summary

MetaMan turns scattered files and ad-hoc metadata into a structure you can trust.
