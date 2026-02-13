import csv
import json
import os
import re
import shutil
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


LOG_CB = Callable[[str], None]
PROGRESS_CB = Callable[[str, int], None]
DEFAULT_SESSION_ID = "1"


def _safe_log(log_cb: Optional[LOG_CB], message: str):
    if log_cb:
        log_cb(message)


def _safe_progress(progress_cb: Optional[PROGRESS_CB], stage: str, percent: int):
    if progress_cb:
        progress_cb(stage, max(0, min(100, int(percent))))


def normalize_text(value: Any, case_sensitive: bool = False) -> str:
    s = "" if value is None else str(value)
    s = s.strip()
    s = s.replace(" ", "_")
    if not case_sensitive:
        s = s.lower()
    s = re.sub(r"[^0-9a-zA-Z_\-]+", "", s)
    s = re.sub(r"_+", "_", s)
    return s


def sanitize_folder_name(value: Any, default: str = "unknown") -> str:
    s = "" if value is None else str(value).strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[<>:\\\"/|?*\x00-\x1F]+", "", s)
    s = s.strip("._")
    return s if s else default


def _split_tokens(value: str) -> List[str]:
    if not value:
        return []
    tmp = value
    for sep in ("\\", "/", ".", "-", "_"):
        tmp = tmp.replace(sep, " ")
    return [t for t in tmp.split() if t]


def _looks_like_session_token(token: str) -> bool:
    t = normalize_text(token)
    return bool(re.match(r"^(session[_-]?\d+|s\d+|day\d+|d\d+)$", t))


def _infer_datatype_from_extension(ext: str) -> str:
    ext_map = {
        ".tif": "imaging",
        ".tiff": "imaging",
        ".nwb": "nwb",
        ".csv": "table",
        ".tsv": "table",
        ".xlsx": "table",
        ".xls": "table",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
        ".bin": "binary",
        ".mat": "matlab",
        ".avi": "video",
        ".mp4": "video",
    }
    return ext_map.get(ext.lower(), "unknown")


def _detect_delimiter(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
        sniffed = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        delim = sniffed.delimiter
        if delim in (",", "\t", ";", "|"):
            return delim
        return ","
    except Exception:
        if path.lower().endswith(".tsv"):
            return "\t"
        return ","


def _read_table_any(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls", ".xlsm"):
        import pandas as pd

        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception:
            return pd.read_excel(path)

    delimiter = _detect_delimiter(path)

    import pandas as pd

    encodings = ["utf-8-sig", "utf-16", "utf-16le", "utf-16be", "cp1252", "latin-1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, delimiter=delimiter)
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    raise RuntimeError("Failed to load metadata table.")


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and value != value:
            return ""
    except Exception:
        pass
    return str(value)


def _normalize_source_dirs(source_dirs: Any) -> List[str]:
    raw_values: List[str] = []
    if source_dirs is None:
        return []
    if isinstance(source_dirs, str):
        raw_values = [source_dirs]
    elif isinstance(source_dirs, (list, tuple, set)):
        raw_values = [str(x) for x in source_dirs]
    else:
        raw_values = [str(source_dirs)]

    out: List[str] = []
    seen: Set[str] = set()
    for raw in raw_values:
        for piece in re.split(r"[;\n\r]+", str(raw or "")):
            p = os.path.normpath(piece.strip())
            if not p:
                continue
            low = p.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(p)
    return out


@dataclass
class ColumnAssignment:
    subject_col: str
    session_col: str = ""
    trial_col: str = ""
    genotype_col: str = ""
    condition_col: str = ""
    extra_cols: List[str] = field(default_factory=list)
    match_cols: List[str] = field(default_factory=list)
    case_sensitive: bool = False


@dataclass
class PlanRow:
    row_index: int
    values: Dict[str, Any]
    subject_id: str
    session_id: str
    trial_id: str
    genotype: str
    condition: str
    extra: Dict[str, str]
    match_key: str
    match_key_parts: List[str]
    unmatchable_reason: str = ""


@dataclass
class PlanLoadResult:
    path: str
    delimiter: str
    columns: List[str]
    row_count: int
    preview_rows: List[Dict[str, Any]]
    dataframe: Any


@dataclass
class DiscoveredFile:
    file_id: str
    source_kind: str
    abs_path: str
    rel_path: str
    ext: str
    size: int
    stem_norm: str
    rel_norm: str
    tokens: List[str]
    datatype_detected: str
    group: str = ""


@dataclass
class FileMatch:
    file_id: str
    matched_by: str
    score: int


@dataclass
class RowMatchResult:
    row_index: int
    subject_id: str
    session_id: str
    match_key: str
    raw_matches: List[FileMatch] = field(default_factory=list)
    processed_matches: List[FileMatch] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    conflict_file_ids: Set[str] = field(default_factory=set)


@dataclass
class ScanResult:
    raw_files: List[DiscoveredFile]
    processed_files: List[DiscoveredFile]
    raw_ext_counts: Dict[str, int]
    processed_ext_counts: Dict[str, int]


@dataclass
class MatchPlan:
    row_results: List[RowMatchResult]
    unmatched_raw_file_ids: List[str]
    unmatched_processed_file_ids: List[str]
    conflict_file_ids: Set[str]
    files_index: Dict[str, DiscoveredFile]


@dataclass
class CopyAction:
    timestamp: str
    plan_row_index: int
    subject_id: str
    session_id: str
    trial_id: str
    source_kind: str
    source_path: str
    destination_path: str
    datatype: str
    file_extension: str
    matched_by: str
    copy_status: str
    group: str = ""
    message: str = ""


@dataclass
class RunSummary:
    project: str
    experiment: str
    dry_run: bool = False
    cancelled: bool = False
    total_subjects: int = 0
    total_sessions: int = 0
    raw_files_copied: int = 0
    processed_files_copied: int = 0
    raw_files_planned: int = 0
    processed_files_planned: int = 0
    skipped_files: int = 0
    conflict_files: int = 0
    unmatched_files: int = 0
    error_files: int = 0


@dataclass
class ReorganizerConfig:
    project_name: str
    experiment_name: str
    source_raw_dir: str
    source_processed_dir: str
    target_raw_root: str
    target_processed_root: str
    column_assignment: ColumnAssignment
    source_raw_dirs: List[str] = field(default_factory=list)
    source_processed_dirs: List[str] = field(default_factory=list)
    datatype_map: Dict[str, str] = field(default_factory=dict)
    dry_run: bool = True
    preserve_group_hierarchy: bool = False
    verify_size: bool = False
    overwrite_strategy: str = "skip"  # skip | rename | overwrite
    overwrite_confirm: bool = False

    @property
    def target_raw_experiment_root(self) -> str:
        return os.path.join(self.target_raw_root, self.project_name, self.experiment_name)

    @property
    def target_processed_experiment_root(self) -> str:
        return os.path.join(self.target_processed_root, self.project_name, self.experiment_name)


class MetadataPlanLoader:
    def load(self, path: str) -> PlanLoadResult:
        if not path:
            raise ValueError("Metadata plan path is empty.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata plan not found: {path}")

        delimiter = _detect_delimiter(path)
        df = _read_table_any(path)
        columns = [str(c) for c in list(df.columns)]
        preview: List[Dict[str, Any]] = []
        for _, row in df.head(30).iterrows():
            preview.append({str(k): row[k] for k in df.columns})

        return PlanLoadResult(
            path=path,
            delimiter=delimiter,
            columns=columns,
            row_count=int(len(df)),
            preview_rows=preview,
            dataframe=df,
        )


class PlanNormalizer:
    def normalize(self, load_result: PlanLoadResult, assignment: ColumnAssignment) -> Tuple[List[PlanRow], Any]:
        df = load_result.dataframe.copy()
        cols = [str(c) for c in df.columns]
        if assignment.subject_col not in cols:
            raise ValueError("subject_id column assignment is missing or invalid.")

        if not assignment.match_cols:
            match_cols = [assignment.subject_col]
            if assignment.session_col and assignment.session_col in cols:
                match_cols.append(assignment.session_col)
            assignment.match_cols = match_cols

        rows: List[PlanRow] = []
        normalized_records: List[Dict[str, Any]] = []

        for idx, (_, row) in enumerate(df.iterrows()):
            values = {str(c): row[c] for c in df.columns}
            subject_raw = _as_str(values.get(assignment.subject_col, "")).strip()
            subject_id = subject_raw

            session_raw = ""
            if assignment.session_col and assignment.session_col in values:
                session_raw = _as_str(values.get(assignment.session_col, "")).strip()

            if session_raw:
                session_id = session_raw
            else:
                # If session is not assigned (or missing in the row), treat it as a single default session.
                session_id = DEFAULT_SESSION_ID

            trial_id = _as_str(values.get(assignment.trial_col, "")).strip() if assignment.trial_col else ""
            genotype = _as_str(values.get(assignment.genotype_col, "")).strip() if assignment.genotype_col else ""
            condition = _as_str(values.get(assignment.condition_col, "")).strip() if assignment.condition_col else ""

            extras: Dict[str, str] = {}
            for col in assignment.extra_cols:
                if col in values:
                    extras[col] = _as_str(values.get(col, "")).strip()

            match_parts: List[str] = []
            missing_match_cols: List[str] = []
            for col in assignment.match_cols:
                v = _as_str(values.get(col, "")).strip()
                if not v:
                    missing_match_cols.append(col)
                    continue
                match_parts.append(normalize_text(v, assignment.case_sensitive))

            match_parts = [p for p in match_parts if p]
            match_key = "__".join(match_parts)
            unmatchable_reason = ""
            if not match_key:
                if missing_match_cols:
                    unmatchable_reason = f"Missing values in match columns: {', '.join(missing_match_cols)}"
                else:
                    unmatchable_reason = "Could not compute match key"

            plan_row = PlanRow(
                row_index=idx,
                values=values,
                subject_id=subject_id,
                session_id=session_id,
                trial_id=trial_id,
                genotype=genotype,
                condition=condition,
                extra=extras,
                match_key=match_key,
                match_key_parts=match_parts,
                unmatchable_reason=unmatchable_reason,
            )
            rows.append(plan_row)

            normalized_records.append(
                {
                    **{str(k): values.get(k) for k in values.keys()},
                    "__subject_id": subject_id,
                    "__session_id": session_id,
                    "__trial_id": trial_id,
                    "__genotype": genotype,
                    "__condition": condition,
                    "__match_key": match_key,
                    "__unmatchable_reason": unmatchable_reason,
                }
            )

        import pandas as pd

        normalized_df = pd.DataFrame(normalized_records)
        return rows, normalized_df


class FileScanner:
    def scan(
        self,
        source_raw_dirs: Any,
        source_processed_dirs: Any,
        case_sensitive: bool,
        cancel_event,
        log_cb: Optional[LOG_CB] = None,
        progress_cb: Optional[PROGRESS_CB] = None,
    ) -> ScanResult:
        _safe_progress(progress_cb, "scanning", 0)
        raw_roots = _normalize_source_dirs(source_raw_dirs)
        processed_roots = _normalize_source_dirs(source_processed_dirs)
        total_roots = max(1, len(raw_roots) + len(processed_roots))
        done_roots = 0

        raw_index: Dict[str, DiscoveredFile] = {}
        processed_index: Dict[str, DiscoveredFile] = {}

        for root in raw_roots:
            if cancel_event and cancel_event.is_set():
                break
            _safe_log(log_cb, f"[scan] raw root: {root}")
            files, _ = self._scan_one(
                source_dir=root,
                source_kind="raw",
                case_sensitive=case_sensitive,
                cancel_event=cancel_event,
                log_cb=log_cb,
            )
            for f in files:
                raw_index[f.file_id] = f
            done_roots += 1
            _safe_progress(progress_cb, "scanning", int(done_roots * 100 / total_roots))

        for root in processed_roots:
            if cancel_event and cancel_event.is_set():
                break
            _safe_log(log_cb, f"[scan] processed root: {root}")
            files, _ = self._scan_one(
                source_dir=root,
                source_kind="processed",
                case_sensitive=case_sensitive,
                cancel_event=cancel_event,
                log_cb=log_cb,
            )
            for f in files:
                processed_index[f.file_id] = f
            done_roots += 1
            _safe_progress(progress_cb, "scanning", int(done_roots * 100 / total_roots))

        raw_files = sorted(raw_index.values(), key=lambda x: x.abs_path)
        processed_files = sorted(processed_index.values(), key=lambda x: x.abs_path)
        raw_ext = self._build_ext_counts(raw_files)
        proc_ext = self._build_ext_counts(processed_files)
        _safe_progress(progress_cb, "scanning", 100)
        _safe_log(log_cb, f"Scan complete: raw={len(raw_files)} processed={len(processed_files)}")
        return ScanResult(
            raw_files=raw_files,
            processed_files=processed_files,
            raw_ext_counts=raw_ext,
            processed_ext_counts=proc_ext,
        )

    def _build_ext_counts(self, files: Sequence[DiscoveredFile]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for f in files:
            out[f.ext] = out.get(f.ext, 0) + 1
        return out

    def _scan_one(
        self,
        source_dir: str,
        source_kind: str,
        case_sensitive: bool,
        cancel_event,
        log_cb: Optional[LOG_CB],
    ) -> Tuple[List[DiscoveredFile], Dict[str, int]]:
        files: List[DiscoveredFile] = []
        ext_counts: Dict[str, int] = {}
        if not source_dir or not os.path.isdir(source_dir):
            _safe_log(log_cb, f"[scan] {source_kind} source missing or not a directory: {source_dir}")
            return files, ext_counts

        scanned = 0
        root_path = os.path.abspath(source_dir)
        for root, _, names in os.walk(root_path):
            if cancel_event and cancel_event.is_set():
                _safe_log(log_cb, "[scan] Cancel requested.")
                break
            for fname in names:
                if cancel_event and cancel_event.is_set():
                    break
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, root_path)
                ext = os.path.splitext(fname)[1].lower()
                ext_counts[ext] = ext_counts.get(ext, 0) + 1

                try:
                    size = int(os.path.getsize(abs_path))
                except Exception:
                    size = -1

                token_source = rel_path.replace("\\", "/")
                rel_norm = normalize_text(token_source.replace("/", "_"), case_sensitive)
                stem_norm = normalize_text(os.path.splitext(fname)[0], case_sensitive)
                raw_tokens = re.split(r"[\\/._-]+", token_source)
                tokens = []
                for tok in raw_tokens:
                    nt = normalize_text(tok, case_sensitive)
                    if nt:
                        tokens.append(nt)

                datatype, group = self._infer_labels(rel_path, source_kind, ext)

                file_id = f"{source_kind}:{abs_path}"
                files.append(
                    DiscoveredFile(
                        file_id=file_id,
                        source_kind=source_kind,
                        abs_path=abs_path,
                        rel_path=rel_path,
                        ext=ext,
                        size=size,
                        stem_norm=stem_norm,
                        rel_norm=rel_norm,
                        tokens=tokens,
                        datatype_detected=datatype,
                        group=group,
                    )
                )
                scanned += 1
                if scanned % 500 == 0:
                    _safe_log(log_cb, f"[scan] {source_kind}: {scanned} files indexed...")

        _safe_log(log_cb, f"[scan] {source_kind}: {scanned} files")
        return files, ext_counts

    def _infer_labels(self, rel_path: str, source_kind: str, ext: str) -> Tuple[str, str]:
        parts = Path(rel_path).parts
        group = ""

        if len(parts) >= 2:
            parent = parts[-2]
        else:
            parent = ""

        datatype = parent

        if source_kind == "processed":
            if len(parts) >= 3:
                grand = parts[-3]
                if _looks_like_session_token(grand):
                    datatype = parent
                    group = ""
                else:
                    datatype = grand
                    group = parent

        datatype = sanitize_folder_name(datatype, default="unknown")
        if not datatype or datatype == "unknown":
            datatype = _infer_datatype_from_extension(ext)

        if not datatype:
            datatype = "unknown"
        if not group:
            group = ""
        return datatype, sanitize_folder_name(group, default="")


class Matcher:
    def build_match_plan(
        self,
        plan_rows: Sequence[PlanRow],
        scan_result: ScanResult,
        case_sensitive: bool,
        cancel_event,
        log_cb: Optional[LOG_CB] = None,
        progress_cb: Optional[PROGRESS_CB] = None,
    ) -> MatchPlan:
        _safe_progress(progress_cb, "matching", 0)

        files_index: Dict[str, DiscoveredFile] = {}
        for f in scan_result.raw_files + scan_result.processed_files:
            files_index[f.file_id] = f

        row_results: List[RowMatchResult] = []
        file_to_rows: Dict[str, Set[int]] = {}

        total_rows = max(1, len(plan_rows))
        for i, row in enumerate(plan_rows):
            if cancel_event and cancel_event.is_set():
                _safe_log(log_cb, "[match] Cancel requested.")
                break

            rr = RowMatchResult(
                row_index=row.row_index,
                subject_id=row.subject_id,
                session_id=row.session_id,
                match_key=row.match_key,
            )

            if row.unmatchable_reason:
                rr.warnings.append(row.unmatchable_reason)
                row_results.append(rr)
                _safe_progress(progress_cb, "matching", int((i + 1) * 100 / total_rows))
                continue

            for f in scan_result.raw_files:
                match = self._match_file(row, f, case_sensitive)
                if match:
                    rr.raw_matches.append(match)
                    file_to_rows.setdefault(f.file_id, set()).add(row.row_index)

            for f in scan_result.processed_files:
                match = self._match_file(row, f, case_sensitive)
                if match:
                    rr.processed_matches.append(match)
                    file_to_rows.setdefault(f.file_id, set()).add(row.row_index)

            if not rr.raw_matches and not rr.processed_matches:
                rr.warnings.append("No files matched this plan row")

            row_results.append(rr)
            _safe_progress(progress_cb, "matching", int((i + 1) * 100 / total_rows))

        self._apply_session_inference_disambiguation(row_results, files_index)

        file_to_rows = {}
        for rr in row_results:
            for m in rr.raw_matches + rr.processed_matches:
                file_to_rows.setdefault(m.file_id, set()).add(rr.row_index)

        conflict_file_ids = {fid for fid, rows in file_to_rows.items() if len(rows) > 1}

        for rr in row_results:
            conflicts = {
                m.file_id
                for m in (rr.raw_matches + rr.processed_matches)
                if m.file_id in conflict_file_ids
            }
            rr.conflict_file_ids = conflicts
            if conflicts:
                rr.warnings.append(f"{len(conflicts)} conflicting file(s) matched multiple plan rows")

            rr.raw_matches.sort(key=lambda m: (m.score, m.file_id))
            rr.processed_matches.sort(key=lambda m: (m.score, m.file_id))

        matched_raw_ids = {
            m.file_id
            for rr in row_results
            for m in rr.raw_matches
            if m.file_id not in conflict_file_ids
        }
        matched_proc_ids = {
            m.file_id
            for rr in row_results
            for m in rr.processed_matches
            if m.file_id not in conflict_file_ids
        }

        unmatched_raw = [f.file_id for f in scan_result.raw_files if f.file_id not in matched_raw_ids and f.file_id not in conflict_file_ids]
        unmatched_proc = [f.file_id for f in scan_result.processed_files if f.file_id not in matched_proc_ids and f.file_id not in conflict_file_ids]

        _safe_log(
            log_cb,
            (
                "Match plan built: "
                f"rows={len(row_results)} conflicts={len(conflict_file_ids)} "
                f"unmatched_raw={len(unmatched_raw)} unmatched_processed={len(unmatched_proc)}"
            ),
        )
        _safe_progress(progress_cb, "matching", 100)

        return MatchPlan(
            row_results=row_results,
            unmatched_raw_file_ids=unmatched_raw,
            unmatched_processed_file_ids=unmatched_proc,
            conflict_file_ids=conflict_file_ids,
            files_index=files_index,
        )

    def _apply_session_inference_disambiguation(
        self,
        row_results: List[RowMatchResult],
        files_index: Dict[str, DiscoveredFile],
    ):
        rows_by_key: Dict[str, List[RowMatchResult]] = {}
        for rr in row_results:
            if rr.match_key:
                rows_by_key.setdefault(rr.match_key, []).append(rr)

        for _, grouped_rows in rows_by_key.items():
            if len(grouped_rows) <= 1:
                continue

            file_to_rows: Dict[str, List[RowMatchResult]] = {}
            for rr in grouped_rows:
                for m in rr.raw_matches + rr.processed_matches:
                    file_to_rows.setdefault(m.file_id, []).append(rr)

            for fid, rrs in file_to_rows.items():
                if len(rrs) <= 1:
                    continue
                f = files_index.get(fid)
                if not f:
                    continue
                inferred = self._infer_session_from_tokens(f.tokens)
                if not inferred:
                    continue

                keep = []
                for rr in rrs:
                    rr_sess = normalize_text(rr.session_id, case_sensitive=False)
                    if rr_sess == inferred:
                        keep.append(rr)
                if len(keep) != 1:
                    continue

                keep_idx = keep[0].row_index
                for rr in rrs:
                    if rr.row_index == keep_idx:
                        continue
                    rr.raw_matches = [m for m in rr.raw_matches if m.file_id != fid]
                    rr.processed_matches = [m for m in rr.processed_matches if m.file_id != fid]

    def _infer_session_from_tokens(self, tokens: Sequence[str]) -> str:
        for tok in tokens:
            t = normalize_text(tok, case_sensitive=False)
            m = re.match(r"^(?:session|s|day|d)_?(\d+)$", t)
            if m:
                return f"session_{int(m.group(1))}"
        return ""

    def _match_file(self, row: PlanRow, file: DiscoveredFile, case_sensitive: bool) -> Optional[FileMatch]:
        key = row.match_key
        if not key:
            return None

        parts = row.match_key_parts
        if not parts:
            return None

        if key == file.stem_norm:
            return FileMatch(file_id=file.file_id, matched_by="stem_exact", score=0)

        if self._contains_token_sequence(file.tokens, parts):
            return FileMatch(file_id=file.file_id, matched_by="path_token_sequence", score=1)

        if key in file.rel_norm:
            return FileMatch(file_id=file.file_id, matched_by="path_substring", score=2)

        if all(p in set(file.tokens) for p in parts):
            return FileMatch(file_id=file.file_id, matched_by="token_set", score=3)

        return None

    def _contains_token_sequence(self, tokens: Sequence[str], parts: Sequence[str]) -> bool:
        if not parts:
            return False
        if len(parts) == 1:
            return parts[0] in tokens
        n = len(parts)
        for i in range(0, len(tokens) - n + 1):
            if list(tokens[i : i + n]) == list(parts):
                return True
        return False


class CopyExecutor:
    def execute(
        self,
        config: ReorganizerConfig,
        plan_rows: Sequence[PlanRow],
        normalized_df,
        match_plan: MatchPlan,
        cancel_event,
        log_cb: Optional[LOG_CB] = None,
        progress_cb: Optional[PROGRESS_CB] = None,
        write_outputs: bool = True,
    ) -> Tuple[List[CopyAction], RunSummary]:
        _safe_progress(progress_cb, "folder_creation", 0)

        run_logs: List[str] = []

        def log_local(message: str):
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] {message}"
            run_logs.append(line)
            _safe_log(log_cb, line)

        self._prepare_output_roots(config, log_local)
        _safe_progress(progress_cb, "folder_creation", 100)

        row_by_index = {row.row_index: row for row in plan_rows}

        actions: List[CopyAction] = []
        all_match_items: List[Tuple[RowMatchResult, FileMatch, str]] = []
        for rr in match_plan.row_results:
            for m in rr.raw_matches:
                all_match_items.append((rr, m, "raw"))
            for m in rr.processed_matches:
                all_match_items.append((rr, m, "processed"))

        total = max(1, len(all_match_items))
        _safe_progress(progress_cb, "copy_operations", 0)

        for idx, (rr, fm, expected_kind) in enumerate(all_match_items):
            if cancel_event and cancel_event.is_set():
                log_local("Cancel requested. Stopping copy operations.")
                break

            file = match_plan.files_index.get(fm.file_id)
            if not file:
                continue

            if file.source_kind != expected_kind:
                continue

            if file.file_id in match_plan.conflict_file_ids:
                actions.append(
                    self._make_action(
                        rr,
                        row_by_index,
                        file,
                        "",
                        fm,
                        copy_status="conflict",
                        message="File matches multiple plan rows. Not copied by default.",
                    )
                )
                _safe_progress(progress_cb, "copy_operations", int((idx + 1) * 100 / total))
                continue

            dest_path = self._build_destination_path(config, row_by_index.get(rr.row_index), file)
            resolved_dest, status, status_msg = self._resolve_destination(config, dest_path)
            if not resolved_dest:
                actions.append(
                    self._make_action(
                        rr,
                        row_by_index,
                        file,
                        dest_path,
                        fm,
                        copy_status=status,
                        message=status_msg,
                    )
                )
                _safe_progress(progress_cb, "copy_operations", int((idx + 1) * 100 / total))
                continue

            os.makedirs(os.path.dirname(resolved_dest), exist_ok=True)

            if config.dry_run:
                msg = "Dry run: no file copy performed"
                if status_msg:
                    msg += f" ({status_msg})"
                actions.append(
                    self._make_action(
                        rr,
                        row_by_index,
                        file,
                        resolved_dest,
                        fm,
                        copy_status="planned_copy",
                        message=msg,
                    )
                )
                _safe_progress(progress_cb, "copy_operations", int((idx + 1) * 100 / total))
                continue

            try:
                shutil.copy2(file.abs_path, resolved_dest)
                if config.verify_size:
                    src_size = os.path.getsize(file.abs_path)
                    dst_size = os.path.getsize(resolved_dest)
                    if src_size != dst_size:
                        raise RuntimeError(f"size mismatch src={src_size} dst={dst_size}")
                actions.append(
                    self._make_action(
                        rr,
                        row_by_index,
                        file,
                        resolved_dest,
                        fm,
                        copy_status="copied",
                        message="",
                    )
                )
            except Exception as e:
                actions.append(
                    self._make_action(
                        rr,
                        row_by_index,
                        file,
                        resolved_dest,
                        fm,
                        copy_status="error",
                        message=str(e),
                    )
                )

            _safe_progress(progress_cb, "copy_operations", int((idx + 1) * 100 / total))

        if write_outputs:
            _safe_progress(progress_cb, "metadata_writing", 0)
            writer = MetadataWriter()
            writer.write_outputs(
                config=config,
                normalized_df=normalized_df,
                plan_rows=plan_rows,
                match_plan=match_plan,
                actions=actions,
                run_logs=run_logs,
                cancel_event=cancel_event,
                log_cb=log_local,
            )
            _safe_progress(progress_cb, "metadata_writing", 100)
        else:
            _safe_progress(progress_cb, "metadata_writing", 100)

        summary = self._build_summary(config, actions, plan_rows, match_plan, cancel_event)
        return actions, summary

    def _prepare_output_roots(self, config: ReorganizerConfig, log_cb: LOG_CB):
        os.makedirs(config.target_raw_root, exist_ok=True)
        os.makedirs(config.target_processed_root, exist_ok=True)

        raw_root = config.target_raw_experiment_root
        proc_root = config.target_processed_experiment_root

        for path in (raw_root, proc_root):
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                log_cb(f"Created experiment folder: {path}")
            else:
                if any(True for _ in os.scandir(path)):
                    log_cb(f"Experiment folder already exists and is not empty: {path}")
                else:
                    log_cb(f"Using existing empty experiment folder: {path}")

    def _map_datatype(self, config: ReorganizerConfig, detected: str) -> str:
        key = sanitize_folder_name(detected, default="unknown")
        if key in config.datatype_map:
            return sanitize_folder_name(config.datatype_map[key], default="unknown")
        key_lower = key.lower()
        for src, dst in config.datatype_map.items():
            if src.lower() == key_lower:
                return sanitize_folder_name(dst, default="unknown")
        return key or "unknown"

    def _build_destination_path(self, config: ReorganizerConfig, row: Optional[PlanRow], file: DiscoveredFile) -> str:
        subject = sanitize_folder_name(row.subject_id if row else "unknown_subject", default="unknown_subject")
        session = sanitize_folder_name(row.session_id if row else DEFAULT_SESSION_ID, default=DEFAULT_SESSION_ID)
        dtype = sanitize_folder_name(self._map_datatype(config, file.datatype_detected), default="unknown")

        if file.source_kind == "raw":
            base_dir = os.path.join(config.target_raw_experiment_root, subject, session, dtype)
        else:
            if config.preserve_group_hierarchy and file.group:
                group = sanitize_folder_name(file.group, default="group_unknown")
                base_dir = os.path.join(config.target_processed_experiment_root, group, subject, session, dtype)
            else:
                base_dir = os.path.join(config.target_processed_experiment_root, subject, session, dtype)

        return os.path.join(base_dir, os.path.basename(file.abs_path))

    def _resolve_destination(self, config: ReorganizerConfig, destination: str) -> Tuple[str, str, str]:
        strategy = (config.overwrite_strategy or "skip").strip().lower()
        exists = os.path.exists(destination)

        if not exists:
            return destination, "ready", ""

        if strategy == "overwrite":
            if config.overwrite_confirm:
                return destination, "ready", "overwrite enabled"
            return "", "skipped", "Overwrite strategy selected but confirmation toggle is not enabled"

        if strategy == "rename":
            base, ext = os.path.splitext(destination)
            i = 2
            while True:
                candidate = f"{base}_copy_{i}{ext}"
                if not os.path.exists(candidate):
                    return candidate, "renamed", f"Renamed destination to avoid overwrite ({os.path.basename(candidate)})"
                i += 1

        return "", "skipped", "Destination exists and strategy is skip"

    def _make_action(
        self,
        rr: RowMatchResult,
        row_by_index: Dict[int, PlanRow],
        file: DiscoveredFile,
        destination_path: str,
        fm: FileMatch,
        copy_status: str,
        message: str,
    ) -> CopyAction:
        row = row_by_index.get(rr.row_index)
        datatype = sanitize_folder_name(file.datatype_detected, default="unknown")
        return CopyAction(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            plan_row_index=rr.row_index,
            subject_id=(row.subject_id if row else rr.subject_id),
            session_id=(row.session_id if row else rr.session_id),
            trial_id=(row.trial_id if row else ""),
            source_kind=file.source_kind,
            source_path=file.abs_path,
            destination_path=destination_path,
            datatype=datatype,
            file_extension=file.ext,
            matched_by=fm.matched_by,
            copy_status=copy_status,
            group=file.group,
            message=message,
        )

    def _build_summary(
        self,
        config: ReorganizerConfig,
        actions: Sequence[CopyAction],
        plan_rows: Sequence[PlanRow],
        match_plan: MatchPlan,
        cancel_event,
    ) -> RunSummary:
        summary = RunSummary(
            project=config.project_name,
            experiment=config.experiment_name,
            dry_run=bool(config.dry_run),
        )

        subjects = {sanitize_folder_name(r.subject_id, default="unknown_subject") for r in plan_rows}
        sessions = {
            (sanitize_folder_name(r.subject_id, default="unknown_subject"), sanitize_folder_name(r.session_id, default=DEFAULT_SESSION_ID))
            for r in plan_rows
        }
        summary.total_subjects = len(subjects)
        summary.total_sessions = len(sessions)

        for a in actions:
            if a.copy_status == "copied":
                if a.source_kind == "raw":
                    summary.raw_files_copied += 1
                else:
                    summary.processed_files_copied += 1
            elif a.copy_status in ("planned_copy", "renamed"):
                if a.source_kind == "raw":
                    summary.raw_files_planned += 1
                else:
                    summary.processed_files_planned += 1
            elif a.copy_status == "conflict":
                summary.conflict_files += 1
            elif a.copy_status.startswith("skipped") or a.copy_status == "skipped":
                summary.skipped_files += 1
            elif a.copy_status == "error":
                summary.error_files += 1

        summary.unmatched_files = len(match_plan.unmatched_raw_file_ids) + len(match_plan.unmatched_processed_file_ids)
        summary.cancelled = bool(cancel_event and cancel_event.is_set())
        return summary


class MetadataWriter:
    def write_outputs(
        self,
        config: ReorganizerConfig,
        normalized_df,
        plan_rows: Sequence[PlanRow],
        match_plan: MatchPlan,
        actions: Sequence[CopyAction],
        run_logs: Sequence[str],
        cancel_event,
        log_cb: Optional[LOG_CB] = None,
    ):
        os.makedirs(config.target_raw_experiment_root, exist_ok=True)
        os.makedirs(config.target_processed_experiment_root, exist_ok=True)

        report_rows = self._build_match_report_rows(match_plan, actions)

        normalized_csv = os.path.join(config.target_raw_experiment_root, "experiment_plan_normalized.csv")
        report_csv = os.path.join(config.target_raw_experiment_root, "match_report.csv")
        run_log_path = os.path.join(config.target_raw_experiment_root, "run_log.txt")

        self._write_dataframe_csv(normalized_df, normalized_csv)
        self._write_dict_rows_csv(report_rows, report_csv)
        self._write_text_lines(run_log_path, run_logs)

        _safe_log(log_cb, f"Wrote experiment_plan_normalized.csv: {normalized_csv}")
        _safe_log(log_cb, f"Wrote match_report.csv: {report_csv}")
        _safe_log(log_cb, f"Wrote run_log.txt: {run_log_path}")

        if config.dry_run:
            _safe_log(log_cb, "Dry run enabled: subject/session metadata files were not written.")
            return

        self._write_subject_and_session_metadata(config, plan_rows, actions, cancel_event, log_cb)

    def _write_subject_and_session_metadata(
        self,
        config: ReorganizerConfig,
        plan_rows: Sequence[PlanRow],
        actions: Sequence[CopyAction],
        cancel_event,
        log_cb: Optional[LOG_CB],
    ):
        row_by_index: Dict[int, PlanRow] = {int(r.row_index): r for r in plan_rows}
        rows_by_subject: Dict[str, List[PlanRow]] = {}
        for row in plan_rows:
            sid = sanitize_folder_name(row.subject_id, default="unknown_subject")
            rows_by_subject.setdefault(sid, []).append(row)

        actions_by_subject: Dict[str, List[CopyAction]] = {}
        actions_by_session: Dict[Tuple[str, str], List[CopyAction]] = {}
        for action in actions:
            sid = sanitize_folder_name(action.subject_id, default="unknown_subject")
            sess = sanitize_folder_name(action.session_id, default=DEFAULT_SESSION_ID)
            actions_by_subject.setdefault(sid, []).append(action)
            actions_by_session.setdefault((sid, sess), []).append(action)

        for subject, subject_rows in rows_by_subject.items():
            if cancel_event and cancel_event.is_set():
                _safe_log(log_cb, "Metadata writing cancelled.")
                break

            subject_actions = actions_by_subject.get(subject, [])
            unique_sessions = sorted({sanitize_folder_name(r.session_id, default=DEFAULT_SESSION_ID) for r in subject_rows})

            subject_meta = {
                "subject_id": subject,
                "genotype": self._join_unique([r.genotype for r in subject_rows]),
                "condition": self._join_unique([r.condition for r in subject_rows]),
                "session_count": len(unique_sessions),
                "raw_file_count": len([a for a in subject_actions if a.source_kind == "raw" and a.copy_status in ("copied", "planned_copy", "renamed")]),
                "processed_file_count": len([a for a in subject_actions if a.source_kind == "processed" and a.copy_status in ("copied", "planned_copy", "renamed")]),
            }

            all_extra_keys = sorted({k for r in subject_rows for k in r.extra.keys()})
            for key in all_extra_keys:
                subject_meta[key] = self._join_unique([r.extra.get(key, "") for r in subject_rows])

            raw_subject_dir = os.path.join(config.target_raw_experiment_root, subject)
            proc_subject_dir = os.path.join(config.target_processed_experiment_root, subject)
            os.makedirs(raw_subject_dir, exist_ok=True)
            os.makedirs(proc_subject_dir, exist_ok=True)

            self._write_dict_rows_csv([subject_meta], os.path.join(raw_subject_dir, "subject_metadata.csv"))
            self._write_dict_rows_csv([subject_meta], os.path.join(proc_subject_dir, "subject_metadata.csv"))

            self._write_h5_metadata(
                os.path.join(raw_subject_dir, "subject_metadata.h5"),
                attrs=subject_meta,
                rows=[a.__dict__ for a in subject_actions],
                dataset_name="file_actions",
            )
            self._write_h5_metadata(
                os.path.join(proc_subject_dir, "subject_metadata.h5"),
                attrs=subject_meta,
                rows=[a.__dict__ for a in subject_actions],
                dataset_name="file_actions",
            )

            for session in unique_sessions:
                session_rows = [r for r in subject_rows if sanitize_folder_name(r.session_id, default=DEFAULT_SESSION_ID) == session]
                session_actions = actions_by_session.get((subject, session), [])
                trial_ids = self._join_unique([r.trial_id for r in session_rows])
                groups = self._join_unique([a.group for a in session_actions if a.group])
                genotype = self._join_unique([r.genotype for r in session_rows])
                condition = self._join_unique([r.condition for r in session_rows])

                session_meta = {
                    "subject_id": subject,
                    "session_id": session,
                    "trial_id": trial_ids,
                    "genotype": genotype,
                    "condition": condition,
                    "group": groups,
                    "file_count": len(session_actions),
                }
                all_extra_keys = sorted({k for r in session_rows for k in r.extra.keys()})
                for key in all_extra_keys:
                    session_meta[key] = self._join_unique([r.extra.get(key, "") for r in session_rows])
                file_rows = self._build_session_file_rows(session_actions, row_by_index, session_meta)

                raw_session_dir = os.path.join(config.target_raw_experiment_root, subject, session)
                proc_session_dir = os.path.join(config.target_processed_experiment_root, subject, session)
                os.makedirs(raw_session_dir, exist_ok=True)
                os.makedirs(proc_session_dir, exist_ok=True)

                self._write_dict_rows_csv(
                    file_rows,
                    os.path.join(raw_session_dir, "session_metadata.csv"),
                    allow_empty=True,
                )
                self._write_dict_rows_csv(
                    file_rows,
                    os.path.join(proc_session_dir, "session_metadata.csv"),
                    allow_empty=True,
                )

                self._write_h5_metadata(
                    os.path.join(raw_session_dir, "session_metadata.h5"),
                    attrs=session_meta,
                    rows=file_rows,
                    dataset_name="files",
                )
                self._write_h5_metadata(
                    os.path.join(proc_session_dir, "session_metadata.h5"),
                    attrs=session_meta,
                    rows=file_rows,
                    dataset_name="files",
                )

    def _build_session_file_rows(
        self,
        session_actions: Sequence[CopyAction],
        row_by_index: Dict[int, PlanRow],
        session_meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for action in session_actions:
            if not str(action.destination_path or "").strip():
                continue
            row: Dict[str, Any] = {"path": action.destination_path}
            try:
                plan_row = row_by_index.get(int(action.plan_row_index))
            except Exception:
                plan_row = None

            if plan_row is not None:
                row["subject_id"] = plan_row.subject_id
                row["session_id"] = plan_row.session_id
                if str(plan_row.trial_id or "").strip():
                    row["trial_id"] = plan_row.trial_id
                if str(plan_row.genotype or "").strip():
                    row["genotype"] = plan_row.genotype
                if str(plan_row.condition or "").strip():
                    row["condition"] = plan_row.condition
                for k, v in (plan_row.extra or {}).items():
                    if str(k).strip():
                        row[str(k)] = _as_str(v).strip()
            else:
                for k in ("subject_id", "session_id", "trial_id", "genotype", "condition"):
                    v = _as_str(session_meta.get(k, "")).strip()
                    if v:
                        row[k] = v
                for k, v in session_meta.items():
                    if k in ("subject_id", "session_id", "trial_id", "genotype", "condition", "group", "file_count"):
                        continue
                    sv = _as_str(v).strip()
                    if sv:
                        row[str(k)] = sv

            if str(action.datatype or "").strip():
                row["datatype"] = action.datatype
            if str(action.file_extension or "").strip():
                row["file_extension"] = action.file_extension
            if str(action.source_kind or "").strip():
                row["source_kind"] = action.source_kind
            if str(action.group or "").strip():
                row["group"] = action.group
            rows.append(row)
        return rows

    def _build_match_report_rows(self, match_plan: MatchPlan, actions: Sequence[CopyAction]) -> List[Dict[str, Any]]:
        rows = [a.__dict__.copy() for a in actions]

        for fid in match_plan.unmatched_raw_file_ids:
            f = match_plan.files_index.get(fid)
            if not f:
                continue
            rows.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "plan_row_index": -1,
                    "subject_id": "",
                    "session_id": "",
                    "trial_id": "",
                    "source_kind": f.source_kind,
                    "source_path": f.abs_path,
                    "destination_path": "",
                    "datatype": f.datatype_detected,
                    "file_extension": f.ext,
                    "matched_by": "",
                    "copy_status": "unmatched",
                    "group": f.group,
                    "message": "No plan row matched this file",
                }
            )

        for fid in match_plan.unmatched_processed_file_ids:
            f = match_plan.files_index.get(fid)
            if not f:
                continue
            rows.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "plan_row_index": -1,
                    "subject_id": "",
                    "session_id": "",
                    "trial_id": "",
                    "source_kind": f.source_kind,
                    "source_path": f.abs_path,
                    "destination_path": "",
                    "datatype": f.datatype_detected,
                    "file_extension": f.ext,
                    "matched_by": "",
                    "copy_status": "unmatched",
                    "group": f.group,
                    "message": "No plan row matched this file",
                }
            )

        return rows

    def _join_unique(self, values: Iterable[str]) -> str:
        clean = []
        seen = set()
        for v in values:
            s = _as_str(v).strip()
            if not s:
                continue
            if s not in seen:
                seen.add(s)
                clean.append(s)
        return "|".join(clean)

    def _write_dataframe_csv(self, df, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

    def _write_dict_rows_csv(self, rows: List[Dict[str, Any]], path: str, allow_empty: bool = False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not rows:
            if allow_empty:
                import pandas as pd

                pd.DataFrame([]).to_csv(path, index=False)
                return
            rows = [{"note": "no_rows"}]
        import pandas as pd

        pd.DataFrame(rows).to_csv(path, index=False)

    def _write_text_lines(self, path: str, lines: Sequence[str]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(str(line).rstrip() + "\n")

    def _write_h5_metadata(self, path: str, attrs: Dict[str, Any], rows: List[Dict[str, Any]], dataset_name: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            import h5py
        except Exception:
            return

        dt = h5py.string_dtype(encoding="utf-8")
        with h5py.File(path, "w") as h5:
            h5.attrs["schema_version"] = "metaman_reorg_v1"
            payload = [json.dumps(r, ensure_ascii=False) for r in rows]

            if dataset_name == "file_actions":
                g = h5.create_group("subject")
                for k, v in attrs.items():
                    g.attrs[str(k)] = _as_str(v)
                g.create_dataset("file_actions_json", data=payload, dtype=dt)
            elif dataset_name == "files":
                sessions = h5.create_group("sessions")
                sid = sanitize_folder_name(attrs.get("session_id", DEFAULT_SESSION_ID), default=DEFAULT_SESSION_ID)
                sg = sessions.create_group(sid)
                for k, v in attrs.items():
                    sg.attrs[str(k)] = _as_str(v)
                sg.create_dataset("files_json", data=payload, dtype=dt)
            else:
                g = h5.create_group("metadata")
                for k, v in attrs.items():
                    g.attrs[str(k)] = _as_str(v)
                g.create_dataset(dataset_name, data=payload, dtype=dt)

            h5.create_dataset("all_json", data=json.dumps({"attrs": attrs, "rows": rows}, ensure_ascii=False), dtype=dt)


class DataReorganizerService:
    def __init__(self):
        self.plan_loader = MetadataPlanLoader()
        self.plan_normalizer = PlanNormalizer()
        self.file_scanner = FileScanner()
        self.matcher = Matcher()
        self.copy_executor = CopyExecutor()

    def load_plan(self, path: str) -> PlanLoadResult:
        return self.plan_loader.load(path)

    def normalize_plan(self, load_result: PlanLoadResult, assignment: ColumnAssignment) -> Tuple[List[PlanRow], Any]:
        return self.plan_normalizer.normalize(load_result, assignment)

    def scan_sources(
        self,
        source_raw_dirs: Any,
        source_processed_dirs: Any,
        case_sensitive: bool,
        cancel_event,
        log_cb: Optional[LOG_CB] = None,
        progress_cb: Optional[PROGRESS_CB] = None,
    ) -> ScanResult:
        return self.file_scanner.scan(
            source_raw_dirs=source_raw_dirs,
            source_processed_dirs=source_processed_dirs,
            case_sensitive=case_sensitive,
            cancel_event=cancel_event,
            log_cb=log_cb,
            progress_cb=progress_cb,
        )

    def build_match_plan(
        self,
        plan_rows: Sequence[PlanRow],
        scan_result: ScanResult,
        case_sensitive: bool,
        cancel_event,
        log_cb: Optional[LOG_CB] = None,
        progress_cb: Optional[PROGRESS_CB] = None,
    ) -> MatchPlan:
        return self.matcher.build_match_plan(
            plan_rows=plan_rows,
            scan_result=scan_result,
            case_sensitive=case_sensitive,
            cancel_event=cancel_event,
            log_cb=log_cb,
            progress_cb=progress_cb,
        )

    def execute(
        self,
        config: ReorganizerConfig,
        plan_rows: Sequence[PlanRow],
        normalized_df,
        match_plan: MatchPlan,
        cancel_event,
        log_cb: Optional[LOG_CB] = None,
        progress_cb: Optional[PROGRESS_CB] = None,
        write_outputs: bool = True,
    ) -> Tuple[List[CopyAction], RunSummary]:
        return self.copy_executor.execute(
            config=config,
            plan_rows=plan_rows,
            normalized_df=normalized_df,
            match_plan=match_plan,
            cancel_event=cancel_event,
            log_cb=log_cb,
            progress_cb=progress_cb,
            write_outputs=write_outputs,
        )

    def plan_actions(
        self,
        config: ReorganizerConfig,
        plan_rows: Sequence[PlanRow],
        normalized_df,
        match_plan: MatchPlan,
        cancel_event,
        log_cb: Optional[LOG_CB] = None,
        progress_cb: Optional[PROGRESS_CB] = None,
    ) -> List[CopyAction]:
        plan_cfg = replace(config, dry_run=True)
        actions, _ = self.copy_executor.execute(
            config=plan_cfg,
            plan_rows=plan_rows,
            normalized_df=normalized_df,
            match_plan=match_plan,
            cancel_event=cancel_event,
            log_cb=log_cb,
            progress_cb=progress_cb,
            write_outputs=False,
        )
        return actions
