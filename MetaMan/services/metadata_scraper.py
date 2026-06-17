"""Automatic metadata scraping.

Given a session folder, pull out as much metadata as we can *automatically*:

* file inventory (count, total size, file-type histogram, modified range)
* the recording **modality** (Neuropixels / video / pose tracking / fiber / NWB …)
* **SpikeGLX** ``*.meta`` fields (sample rate, duration, channels, probe, date)
* **video** properties via OpenCV when available (fps, frames, resolution, length)

All keys are prefixed ``Auto: `` so they read as machine-derived and never clash
with user-entered fields. Merging is non-destructive: user values win.
"""

import glob
import os
from datetime import datetime
from typing import Dict, List, Tuple

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg"}


def is_probably_local(path: str) -> bool:
    """Heuristic: True unless *path* is a UNC network share (``\\\\server\\…``).

    Auto-scraping walks the folder, so we skip it on the UI thread for network
    paths (which can be very slow); the manual button still works everywhere.
    """
    p = str(path or "").strip()
    return bool(p) and not (p.startswith("\\\\") or p.startswith("//"))


# ── small helpers ─────────────────────────────────────────────────────────

def _human_size(n: int) -> str:
    f = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if f < 1024 or unit == "PB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.2f} {unit}"
        f /= 1024.0
    return f"{f:.2f} PB"


def _fmt_time(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def _fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


# ── modality classification ────────────────────────────────────────────────

def classify_modality(exts: Dict[str, int], names: List[str]) -> str:
    low = [n.lower() for n in names]
    mods: List[str] = []
    if any(n.endswith(".ap.bin") or n.endswith(".lf.bin") or n.endswith(".ap.meta")
           or n.endswith(".lf.meta") for n in low) or ".imec" in " ".join(low):
        mods.append("Neuropixels (SpikeGLX)")
    if VIDEO_EXTS & set(exts):
        mods.append("Behavioral video")
    if any(("dlc" in n or "deeplabcut" in n) and n.endswith((".csv", ".h5"))
           for n in low):
        mods.append("Pose tracking (DLC)")
    if any(n.endswith(".nwb") for n in low):
        mods.append("NWB")
    if any(n.endswith((".doric", ".ppd")) or "fip" in n or "photometry" in n for n in low):
        mods.append("Fiber photometry")
    if ".rhd" in exts or ".rhs" in exts:
        mods.append("Intan")
    return ", ".join(dict.fromkeys(mods))


# ── SpikeGLX .meta ──────────────────────────────────────────────────────────

def _parse_meta_file(path: str) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                kv[k.strip().lstrip("~")] = v.strip()
    except Exception:
        pass
    return kv


def _scrape_spikeglx(session_dir: str) -> Dict[str, object]:
    metas = glob.glob(os.path.join(session_dir, "**", "*.meta"), recursive=True)
    if not metas:
        return {}
    ap = [m for m in metas if m.lower().endswith(".ap.meta")]
    kv = _parse_meta_file(sorted(ap or metas)[0])
    if not kv:
        return {}
    out: Dict[str, object] = {"Auto: acquisition": "SpikeGLX (Neuropixels)"}
    sr = kv.get("imSampRate") or kv.get("niSampRate") or kv.get("apSampRate")
    if sr:
        try:
            out["Auto: sample rate (Hz)"] = f"{float(sr):,.0f}"
        except Exception:
            out["Auto: sample rate (Hz)"] = sr
    if kv.get("fileTimeSecs"):
        try:
            secs = float(kv["fileTimeSecs"])
            out["Auto: duration"] = _fmt_duration(secs)
            out["Auto: duration (s)"] = round(secs, 1)
        except Exception:
            pass
    if kv.get("nSavedChans"):
        out["Auto: channels"] = kv["nSavedChans"]
    if kv.get("imDatPrb_type") or kv.get("imProbeOpt"):
        out["Auto: probe"] = kv.get("imDatPrb_type") or kv.get("imProbeOpt")
    if kv.get("fileCreateTime"):
        out["Auto: recorded at"] = kv["fileCreateTime"]
    if kv.get("imDatPrb_sn") or kv.get("imProbeSN"):
        out["Auto: probe serial"] = kv.get("imDatPrb_sn") or kv.get("imProbeSN")
    return out


# ── video (OpenCV, optional) ────────────────────────────────────────────────

def _scrape_video(video_paths: List[str]) -> Dict[str, object]:
    if not video_paths:
        return {}
    out: Dict[str, object] = {"Auto: video files": len(video_paths)}
    try:
        import cv2  # noqa: PLC0415
    except Exception:
        return out
    path = sorted(video_paths)[0]
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return out
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
        cap.release()
    except Exception:
        return out
    out["Auto: video"] = os.path.basename(path)
    if w and h:
        out["Auto: video resolution"] = f"{int(w)}×{int(h)}"
    if fps:
        out["Auto: video fps"] = round(float(fps), 2)
    if n:
        out["Auto: video frames"] = int(n)
    if fps and n:
        out["Auto: video duration"] = _fmt_duration(n / fps)
    return out


# ── public API ───────────────────────────────────────────────────────────

def scrape_session(session_dir: str, deep: bool = True) -> Dict[str, object]:
    """Return auto-detected metadata for *session_dir* (keys prefixed ``Auto: ``).

    *deep* enables the heavier probes (video via OpenCV). Set it False for fast,
    inventory-only scraping when scanning many sessions.
    """
    if not session_dir or not os.path.isdir(session_dir):
        return {}

    total = 0
    exts: Dict[str, int] = {}
    mtimes: List[float] = []
    names: List[str] = []
    video_paths: List[str] = []
    file_count = 0

    for root, _dirs, files in os.walk(session_dir):
        for fname in files:
            file_count += 1
            names.append(fname)
            p = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            exts[ext] = exts.get(ext, 0) + 1
            if ext in VIDEO_EXTS:
                video_paths.append(p)
            try:
                total += os.path.getsize(p)
                mtimes.append(os.path.getmtime(p))
            except Exception:
                pass

    out: Dict[str, object] = {}
    out["Auto: file count"] = file_count
    out["Auto: total size"] = _human_size(total)
    if exts:
        top = sorted(exts.items(), key=lambda kv: -kv[1])
        out["Auto: file types"] = ", ".join(
            f"{e or '(none)'} ×{c}" for e, c in top[:8]
        )
    if mtimes:
        lo, hi = min(mtimes), max(mtimes)
        out["Auto: modified"] = _fmt_time(hi) if abs(hi - lo) < 1 else f"{_fmt_time(lo)} → {_fmt_time(hi)}"

    modality = classify_modality(exts, names)
    if modality:
        out["Auto: modality"] = modality

    out.update(_scrape_spikeglx(session_dir))
    if deep:
        out.update(_scrape_video(video_paths))
    return out


def merge_auto(meta: Dict[str, object], scraped: Dict[str, object]) -> Dict[str, object]:
    """Return *meta* with *scraped* fields added where missing (user values win)."""
    out = dict(meta or {})
    for k, v in (scraped or {}).items():
        if k not in out or str(out.get(k, "")).strip() == "":
            out[k] = v
    return out
