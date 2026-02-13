import threading
from typing import Callable
from PySide6.QtCore import QObject, Signal

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
