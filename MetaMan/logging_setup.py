"""Central logging configuration.

MetaMan moves and deletes scientific data, so failures must never vanish.
:func:`setup_logging` installs a rotating file handler (so problems are
recoverable after the fact) plus a console handler, and is safe to call more
than once.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False


def log_path() -> str:
    return str(Path.home() / ".metaman" / "metaman.log")


def setup_logging(level: int = logging.INFO) -> str:
    """Configure root logging once. Returns the log file path."""
    global _CONFIGURED
    path = log_path()
    if _CONFIGURED:
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    try:
        fh = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        # A read-only home should never stop the app from launching.
        pass

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    _CONFIGURED = True
    logging.getLogger(__name__).info("Logging initialised -> %s", path)
    return path
