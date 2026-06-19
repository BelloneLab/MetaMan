# MetaMan analysis API

Query and analyse a whole project straight from Python / Jupyter. The GUI "Find
Sessions" dialog uses the **same** engine, so an interactive query and an
analysis-script query return identical rows.

Two Qt-free modules:

| Module | Use it for |
|---|---|
| `MetaMan.services.query` | search sessions, build tidy DataFrames, summarise a project |
| `MetaMan.services.preprocessing_ops` | track / drive preprocessing steps across the project |

Both understand the two metadata dialects that coexist in real projects (the
canonical `metadata.json` **and** acquisition `*_metadata.json` files) and derive
identity from the folder tree, so a stale `Subject: "rawData"` in a file can
never lie to your analysis.

---

## Querying sessions

```python
from MetaMan.services.query import ProjectQuery

pq = ProjectQuery(r"B:/NPX/rawData/mPFC-NAc")

df = pq.to_dataframe()              # one tidy row per session
df.groupby("session").size()       # how many sessions per condition

# Chainable filters (AND). Rich operators, all case-insensitive:
#   = != contains icontains startswith endswith regex
#   > >= < <=  in  "not in"  between  exists  missing
hits = (pq.where("subject", "=", "51542")
          .where("Auto: modality", "contains", "Neuropixels")
          .where("date", "between", "2026-06-01..2026-06-30"))

print(len(hits))                   # match count
hits.to_csv("nac_51542.csv")       # export the result table
paths = hits.paths()               # session folders, ready to feed a loader
```

`scrape=True` runs the file scraper first, so you can query on derived fields
(`Auto: modality`, `Auto: total size`, `Auto: sample rate (Hz)`, audio kind, ...):

```python
pq = ProjectQuery(project_dir, scrape=True, deep=True)
usv = pq.where("Auto: audio kind", "contains", "ultrasonic")
```

### Virtual fields and helpers

```python
pq.values("subject")               # ['51542', '51543', '51556']
pq.group_counts("session")         # {'healthy': 3, 'object': 3, 'sick': 3, ...}
pq.summary()                       # aggregate dict (see below)
pq.project_info()                  # project_info.json
pq.subjects_info()                 # {subject: subject_info.json}

# anything the operators cannot express:
pq.filter(lambda rec: rec["meta"].get("Auto: duration (s)", 0) and
                      float(rec["meta"]["Auto: duration (s)"]) > 600)
```

### One-liners

```python
from MetaMan.services.query import sessions_dataframe, project_summary
sessions_dataframe(project_dir)    # DataFrame of every session
project_summary(project_dir)       # {sessions, subjects, modalities, date_range, ...}
```

`summary()` returns, for the current match set: session counts, the set of
subjects / experiments / modalities / experimenters, the session date range and
the preprocessing completion percentage.

---

## Driving preprocessing

```python
from MetaMan.services import preprocessing_ops as pp

pp.status_table(project_dir)            # wide DataFrame: one column per step
pp.long_status_table(project_dir)       # tidy (session, step, status) rows
pp.progress_summary(project_dir)        # % complete per step + overall
pp.pending_sessions(project_dir, "spike_sorting")   # what is left to do

# Mutations write the canonical metadata.json/.csv/.h5 triplet, exactly as the
# GUI would leave it:
pp.set_step_status(session_dir, "curation", "completed")
pp.set_step_params(session_dir, "spike_sorting", {"sorter": "kilosort4", "Th": 9})
pp.apply_step_template(project_dir)     # seed default steps where missing

# Mark many sessions at once, optionally with a where-clause or a predicate:
pp.bulk_set_status(project_dir, "histology", "completed",
                   where=("subject", "=", "51542"))
```

Default step templates live in `pp.STEP_TEMPLATES` (`neuropixels`, `fiber`,
`behavior`, `imaging`); `pp.steps_for_modality("Neuropixels SpikeGLX")` picks one
from a free-text hint.

---

## Example: a quick analysis pass

```python
from MetaMan.services.query import ProjectQuery
from MetaMan.services import preprocessing_ops as pp

P = r"B:/NPX/rawData/mPFC-NAc"

# 1. which Neuropixels sessions still need sorting?
todo = pp.pending_sessions(P, "spike_sorting")

# 2. pull a tidy table of just the ultrasonic-audio sessions for a figure
df = (ProjectQuery(P, scrape=True)
        .where("Auto: audio kind", "contains", "ultrasonic")
        .to_dataframe())

# 3. project health at a glance
print(ProjectQuery(P).summary())
print(pp.progress_summary(P))
```
