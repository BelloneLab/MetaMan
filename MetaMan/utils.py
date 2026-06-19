import threading
from typing import Callable
from PySide6.QtCore import QObject, Signal

# ── cooperative cancellation ───────────────────────────────────────────────
# Long copy/backup jobs run on daemon threads. A CancelToken lets the UI (or a
# graceful app shutdown) ask a running job to stop at the next file boundary so
# it never leaves a half-written file behind.

_ACTIVE_TOKENS: set = set()
_ACTIVE_LOCK = threading.Lock()


class CancelToken:
    """A thread-safe cancel flag registered globally so app shutdown can signal
    every in-flight job at once."""

    def __init__(self):
        self._ev = threading.Event()
        with _ACTIVE_LOCK:
            _ACTIVE_TOKENS.add(self)

    def cancel(self):
        self._ev.set()

    def is_cancelled(self) -> bool:
        return self._ev.is_set()

    def done(self):
        with _ACTIVE_LOCK:
            _ACTIVE_TOKENS.discard(self)


def cancel_all_jobs():
    """Signal every active CancelToken (used on graceful shutdown)."""
    with _ACTIVE_LOCK:
        tokens = list(_ACTIVE_TOKENS)
    for t in tokens:
        t.cancel()


def active_job_count() -> int:
    with _ACTIVE_LOCK:
        return len(_ACTIVE_TOKENS)


class LogEmitter(QObject):
    append_line = Signal(str)

    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.append_line.connect(self._append)

    def _append(self, msg: str):
        self.text_widget.append(msg.rstrip())

    def log(self, msg: str):
        self.append_line.emit(msg)

def format_speed(bytes_per_sec: float) -> str:
    units = ["B/s","KB/s","MB/s","GB/s","TB/s"]
    i = 0
    while bytes_per_sec >= 1024 and i < len(units)-1:
        bytes_per_sec /= 1024.0
        i += 1
    return f"{bytes_per_sec:.2f} {units[i]}"

def run_in_thread(fn: Callable, on_error: Callable[[Exception], None]=None):
    def _runner():
        try:
            fn()
        except Exception as e:
            if on_error:
                on_error(e)
    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return t
