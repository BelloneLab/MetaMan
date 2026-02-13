# Data Reorganizer Workflow

The `Data reorganizer` tab ingests an experiment plan and reorganizes raw/processed data into:

- `target_raw_root/<project>/<experiment>/<subject>/<session>/<datatype>/...`
- `target_processed_root/<project>/<experiment>/<subject>/<session>/<datatype>/...`
- optional processed group-preserving mode:
  - `target_processed_root/<project>/<experiment>/<group>/<subject>/<session>/<datatype>/...`

## Step-by-step

1. Configure project/experiment and source/target roots.
2. Load metadata plan (CSV/TSV/Excel).
3. Assign columns (`subject_id` required; optional `session_id`, `trial_id`, `genotype`, `condition`).
4. Choose match-key mode:
   - subject only
   - subject + session
   - custom columns
5. Scan source folders.
6. Build match plan and review:
   - matched counts
   - unmatched files
   - conflicts (files matching multiple rows)
7. Run with:
   - dry run (default)
   - overwrite strategy (`skip`, `rename`, `overwrite + explicit confirmation`)
   - optional copy-size verification

## Matching rules

- Match keys are normalized:
  - trim
  - lowercase unless case-sensitive mode is enabled
  - spaces to `_`
  - remove non `[A-Za-z0-9_-]`
- Key format: selected column values joined by `__`.
- File matching precedence:
  1. exact stem match
  2. token-sequence match in path
  3. normalized path substring
  4. token-set inclusion
- If one file matches multiple rows: conflict, not copied by default.

## Metadata outputs

Experiment-level:

- `experiment_plan_normalized.csv`
- `match_report.csv`
- `run_log.txt`

Subject-level:

- `subject_metadata.csv`
- `subject_metadata.h5`

Session-level:

- `session_metadata.csv`
- `session_metadata.h5`

HDF5 schema uses stable groups:

- `/subject` for subject metadata + file-action table
- `/sessions/<session_id>` for session metadata + file table
