<div align="center">

<img src="MetaMan/assests/metaman.png" alt="MetaMan logo" width="190" />

# MetaMan

### 🧠 Neuro data organization without the folder chaos

*Browse it. Record it. Process it. Back it up. Reorganize it. All in one calm, violet little workspace.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-2f6db3?logo=python&logoColor=white)](https://www.python.org/)
[![UI](https://img.shields.io/badge/UI-PySide6-0e7a5f?logo=qt&logoColor=white)](https://doc.qt.io/qtforpython/)
[![Data](https://img.shields.io/badge/Data-CSV%20%7C%20HDF5%20%7C%20JSON-8a4f7d)](#)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-5a3fe0)](#)
[![Safe copies](https://img.shields.io/badge/Copies-atomic%20%2B%20verified-1f9d57)](#-safe-by-default)

</div>

---

<p align="center">
  <img src="docs/screenshots/01_browse.png" alt="MetaMan Browse tab: project tree with colour-coded levels and live session metadata" width="100%" />
</p>

<p align="center"><sub>The Browse workspace: one schema-driven tree for your whole lab, rich metadata on the right, live stats per node.</sub></p>

---

## ✨ Why MetaMan?

Neuroscience data is a hierarchy: **project → experiment → subject → session → files**. The problem is never the data, it is everything *around* it: folders that drift apart between rigs, metadata that lives in someone's head, backups nobody is sure ran, and that one heroic spreadsheet holding the lab together.

MetaMan turns that into a structure you can trust:

```text
data_root/
  rawData/        <project>/<experiment>/<subject>/<session>/...
  processedData/  <project>/<experiment>/<subject>/<session>/...
```

The hierarchy is **yours to design** (drag blocks around, see [Structure playground](#-design-your-own-hierarchy)), the metadata travels *with* the data as `json + csv + h5`, and every copy is atomic, change-aware and optionally checksum-verified.

---

## 🧭 The five-step workspace

MetaMan is a single window with a violet nav rail. Each stop is one job, and they all follow the **active project** you pick at the top.

| | Stop | What it does |
|---|---|---|
| 🗂️ | **Browse** | Walk the local + server trees, view/edit metadata at every level, right-click for dataset actions |
| ⏺️ | **Record** | Create session metadata, auto-scrape acquisition files, write the `json/csv/h5` triplet |
| ⚙️ | **Process** | Track preprocessing steps, parameters and results folders per session |
| ☁️ | **Transfer** | Back up to server / external HDD, stage recordings, schedule daily jobs, read backup reports |
| 📥 | **Import** | Reorganize messy file dumps into the canonical structure from a metadata plan |

---

## 📸 Take the tour

### 🗂️ Browse: your whole lab as one tidy tree

Local and Server tabs share the same metadata panels. Colour-coded dots mark each level (project, experiment, subject, session), and selecting a node computes live stats (sessions, files, total size, modalities) off the UI thread so the window never freezes. Right-click any node to open, reveal, rename, delete (guarded), create children, or pull a server dataset down to a local copy.

<p align="center">
  <img src="docs/screenshots/01_browse.png" alt="Browse tab" width="100%" />
</p>

### ⏺️ Record: metadata that writes itself (almost)

Point at a session, hit **Auto-scrape**, and MetaMan reads your acquisition files to fill in modality, sample rate, channel count and the file list. Edit anything by hand, save the `json/csv/h5` triplet, and reuse a default template across sessions.

<p align="center">
  <img src="docs/screenshots/02_record.png" alt="Record tab" width="100%" />
</p>

### ⚙️ Process: a paper trail for every preprocessing step

Spike sorting, curation, time-sync, histology, DLC... track each step's status (planned / ongoing / completed), stash its parameters as JSON, import params from CSV/JSON, and attach a results folder under `processedData`.

<p align="center">
  <img src="docs/screenshots/03_process.png" alt="Process tab" width="100%" />
</p>

### ☁️ Transfer: backups you can actually prove happened

Back up the active project to a **Server**, an **External HDD**, or **Both**, with opt-in checksum verify and mirror/prune. Prefer to set it and forget it? Schedule a daily run. Every run is recorded with full metadata and a portable report.

<table>
<tr>
<td width="50%"><img src="docs/screenshots/04_transfer_backup.png" alt="Backup panel" /></td>
<td width="50%"><img src="docs/screenshots/05_transfer_history.png" alt="Backup history and reports" /></td>
</tr>
<tr>
<td align="center"><sub>Backup: destinations, verify, mirror, dry-run preview</sub></td>
<td align="center"><sub>History: every run, status, throughput, full report</sub></td>
</tr>
</table>

### 📥 Import: turn a file dump into the canonical structure

Load a metadata plan (`csv/tsv/xlsx`), map your columns, scan multiple raw and processed source roots, and let deterministic key-matching pair files to sessions. **Dry run by default**, then execute the copy with overwrite-policy controls and a full match report.

<p align="center">
  <img src="docs/screenshots/06_import.png" alt="Import / Data reorganizer tab" width="100%" />
</p>

### 🧩 Design your own hierarchy

Not every lab nests folders the same way. The **Structure playground** is a drag-and-drop block editor: reorder levels, toggle them on/off, rename labels, and watch the live filesystem preview update. Save it as the default or per-project, and the schema travels with the data as a sidecar.

<p align="center">
  <img src="docs/screenshots/07_structure_designer.png" alt="Structure Designer: drag-and-drop folder hierarchy editor" width="92%" />
</p>

---

## 🚀 Quick start

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. launch
python run_app.py
```

That is it. On first run MetaMan creates a safe local data root (`~/MetaManData`) with `rawData/` and `processedData/`. Point it anywhere you like from **File ▸ Set Data Root...**, design your structure, and start browsing.

---

## 🛡️ Safe by default

Moving terabytes off an acquisition machine is the scary part. MetaMan is built so an interrupted job never corrupts your data:

- **Atomic writes** - files land in a temporary `.mmpart` then get renamed, so a crash never leaves a truncated file at the real path.
- **Change-aware** - unchanged files are *skipped*; a same-name file whose source is newer/different is *updated* (size + mtime).
- **Verify (opt-in)** - re-reads each copy and compares a SHA-256 checksum.
- **Mirror / prune (opt-in, off by default)** - also deletes destination files that no longer exist in the source. Without it, backup is a purely additive copy.
- **Free-space precheck** - a backup that cannot fit is refused before it starts.
- **Cancel anytime** - long copies stop cleanly, and closing the app halts any in-flight job.
- **Preview changes** - a dry run shows exactly what a backup would copy / update / prune.
- **Structure sidecar** - `_metaman_structure.json` is written at the project root so a backed-up project carries its folder schema with it.

---

## 🔎 Feature deep-dive

<details>
<summary><b>🗂️ Navigation (Browse)</b></summary>

- Two tabs share one metadata view: **Local** (your data root) and **Server** (a network share). Both navigate the same project → experiment → subject → session hierarchy driven by each project's structure schema.
- On the **Server** tab, point at the share holding the projects and browse it exactly like the local tree (browsing the server never changes your active local project).
- View and edit metadata at all hierarchy levels.
- Load subject metadata from CSV for one or multiple subjects.
- Copy/open paths quickly.
- **Right-click any node** for dataset actions: open, reveal, copy path, load a session into Record/Process, create a child (experiment/subject/session), **Rename** and **Delete** (guarded: type-to-confirm, recycle bin where available).
- **Make local copy** (Server tab): right-click a server project / experiment / session and it is reconstructed under your canonical local `rawData/<project>/<experiment>/...` so you can pull data down for analysis (it appears in the Local tab immediately).

</details>

<details>
<summary><b>⏺️ Recording</b></summary>

- Create and update recording/session metadata.
- Navigate the existing project hierarchy via dropdowns.
- Auto-scrape acquisition files to infer modality, sample rate, channels and the file list.
- Update file list and metadata triplet outputs (`json/csv/h5`).

</details>

<details>
<summary><b>⚙️ Preprocessing</b></summary>

- Track preprocessing steps and completion status.
- Store step parameters and comments.
- Import parameters from CSV/JSON.
- Attach per-step results folders.

</details>

<details>
<summary><b>📥 Data reorganizer (Import)</b></summary>

- Load metadata plans (`csv/tsv/xlsx`).
- Map columns (`subject_id`, `session_id`, `trial_id`, custom fields).
- Match files using deterministic keys.
- Scan **multiple raw and processed source roots**.
- Dry run by default (safe mode).
- Execute copy with overwrite-policy controls.
- Generate match report, run log, and session/subject metadata outputs.

MetaMan writes:
- `experiment_plan_normalized.csv`
- `match_report.csv`
- `run_log.txt`
- `subject_metadata.csv` and `subject_metadata.h5`
- `session_metadata.csv` and `session_metadata.h5`

Output roots follow `target_raw_root/<project>/<experiment>/...` and `target_processed_root/<project>/<experiment>/...`.

</details>

<details>
<summary><b>☁️ Backup, schedule, history & reports</b></summary>

- Manual backup to **Server**, **External HDD**, or **Both**.
- Scheduled daily backups per project, with optional experiment-level selection.
- Last-used backup roots and schedules persisted.
- Every backup run is recorded with full metadata: timestamp, scope, destination, duration, files copied / updated / skipped / failed / verified / pruned, bytes copied and average throughput.
- A **Last backup** card summarises the active project's most recent run.
- A report file (`report_<ts>.json` + readable `.txt`, plus `history.csv`) is written to `<destination>/_metaman_backup/<project>/` so the record travels with the data.

</details>

<details>
<summary><b>🔗 Staging (linked recordings)</b></summary>

- Record new sessions **locally** without downloading server projects.
- Browse the server root to pick the target project, experiment and subject.
- Create linked recordings in a local staging area (`data_root/staging/`); each carries metadata tagging its server destination.
- **Sync all pending** or selected recordings to the server with one click.
- Staged recordings are also auto-synced during scheduled backups.
- Status tracking: Pending → Synced / Error with re-queue support.

</details>

<details>
<summary><b>🔍 Find Sessions (structured query)</b></summary>

Run a structured query across a project's session metadata, e.g. `Region = CA1` AND `Auto: sample rate (Hz) > 30000`. Double-click a result to load it into Record / Process. Available under **Project ▸ Find Sessions...**.

</details>

> Operational notes: actions and errors are logged to `~/.metaman/metaman.log`; settings are saved atomically. NWB / BIDS export is **not** included yet (it needs a dedicated data-mapping design and the `pynwb` dependency).

---

## 🎯 Design principles

- **Safe by default**: dry run + no blind overwrite.
- **Transparent operations**: logs, preview tables, match reports.
- **Responsive UI**: background worker threads for long operations.
- **Deterministic matching**: reproducible plan-to-file mapping.

---

## 🗃️ Project layout

```text
MetaMan/
  main.py            # window, menus, backup orchestration
  config.py          # paths, defaults, constants
  state.py           # settings + active-project state
  io_ops.py          # metadata triplet read/write
  theme.py           # the violet workspace stylesheet
  nav_rail.py        # left navigation rail
  structure_designer.py
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
    structure_schema.py
```

---

## 💡 Tips

- Keep one canonical `data_root` with `rawData/` and `processedData/` subfolders.
- Use the Import dry run first, then execute.
- Save/load reorganizer configs for recurring pipelines.
- Prefer an explicit `session_id` in plans when possible.

---

## 🩹 Troubleshooting

**`ModuleNotFoundError: No module named 'PySide6'`** - install dependencies with `pip install -r requirements.txt`.

**Old drive path errors (for example `B:\` not found)** - MetaMan falls back to a safe local root (`~/MetaManData`). You can also set your preferred root from **File ▸ Set Data Root...**.

---

<div align="center">

### One-line summary

**MetaMan turns scattered files and ad-hoc metadata into a structure you can trust.**

<sub>Made with 🧠 and a lot of violet for the BelloneLab.</sub>

</div>
