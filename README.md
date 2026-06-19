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

### Navigation tab (Browse)
- Two tabs share one metadata view: **Local** (your data root) and **Server**
  (a network share). Both navigate the same project -> experiment -> subject ->
  session hierarchy driven by each project's structure schema
- On the **Server** tab, point at the share holding the projects and browse it
  exactly like the local tree (browsing the server never changes your active
  local project)
- View and edit metadata at all hierarchy levels
- Load subject metadata from CSV for one or multiple subjects
- Copy/open paths quickly
- **Right-click any node** for dataset actions: open, reveal, copy path, load a
  session into Record/Process, create a child (experiment/subject/session),
  **Rename** and **Delete** (guarded: type-to-confirm, recycle bin where
  available)
- **Make local copy** (Server tab): right-click a server project / experiment /
  session -> it is reconstructed under your canonical local
  `rawData/<project>/<experiment>/…` so you can pull data down for analysis (it
  appears in the Local tab immediately)

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

### Backup history & reports (Transfer ▸ History)
- Every backup run is recorded with full metadata: timestamp, scope,
  destination, duration, files copied / updated / skipped / failed / verified /
  pruned, bytes copied and average throughput
- A **Last backup** card summarises the active project's most recent run
- A report file (`report_<ts>.json` + readable `.txt`, plus `history.csv`) is
  written to `<destination>/_metaman_backup/<project>/` so the record travels
  with the data

### Safe copying
- **Atomic writes**: files are written to a temporary `.mmpart` then renamed, so
  an interrupted run never leaves a truncated file at the real path
- **Change-aware**: an unchanged file is **skipped**, a same-name file whose
  source is newer/different is **updated** (size + modification time)
- **Verify** (opt-in): re-reads each copy and compares a SHA-256 checksum
- **Mirror / prune** (opt-in, off by default): also deletes destination files
  that no longer exist in the source. Without it, backup is a safe additive copy
- **Free-space precheck**: a backup that cannot fit is refused before it starts
- **Cancel**: long copies/backups can be stopped, and closing the app stops any
  in-flight job cleanly
- **Preview changes**: a dry run shows what a backup would copy / update / prune
  without copying anything
- A structure sidecar (`_metaman_structure.json`) is written at the project root
  so a backed-up project carries its folder schema to the server

### Find Sessions (Project ▸ Find Sessions…)
- Structured query across a project's session metadata, e.g.
  `Region = CA1` AND `Auto: sample rate (Hz) > 30000`. Double-click a result to
  load it into Record / Process.

> Operational notes: actions and errors are logged to `~/.metaman/metaman.log`;
> settings are saved atomically. NWB / BIDS export is **not** included yet (it
> needs a dedicated data-mapping design and the `pynwb` dependency).

### Staging tab (linked recordings)
- Record new sessions **locally** without downloading server projects
- Browse server root to pick target project, experiment, and subject
- Create linked recordings in a local staging area (`data_root/staging/`)
- Each recording carries metadata tagging its server destination
- **Sync all pending** or selected recordings to the server with one click
- Staged recordings are also auto-synced during scheduled backups
- Status tracking: Pending → Synced / Error with re-queue support
- Open local staging folder or individual recording folders directly

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
    transfer_tab.py
    data_reorganizer_tab.py
    staging_tab.py
  services/
    data_reorganizer.py
    file_scanner.py
    fs_ops.py
    server_sync.py
    backup_report.py
    staging_service.py
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
