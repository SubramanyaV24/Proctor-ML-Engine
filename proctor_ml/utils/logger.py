"""
Session-scoped file logger.

Each session gets its own log file:  logs/session_<id>_<date>.log
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict

from proctor_ml.config import LOGS_DIR

_session_loggers: Dict[str, logging.Logger] = {}


def get_session_logger(session_id: str) -> logging.Logger:
    """Return (creating if necessary) a logger scoped to *session_id*."""
    if session_id in _session_loggers:
        return _session_loggers[session_id]

    os.makedirs(LOGS_DIR, exist_ok=True)

    logger = logging.getLogger(f"session.{session_id}")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        _session_loggers[session_id] = logger
        return logger

    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(LOGS_DIR, f"session_{session_id}_{date_str}.log")

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    _session_loggers[session_id] = logger
    return logger
