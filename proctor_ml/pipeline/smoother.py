"""
Temporal smoothing helpers.

RollingSmoother  — exponential moving average for continuous values (yaw, pitch).
BooleanSmoother  — majority-vote window for boolean signals (phone, multiple faces).
SessionState     — per-session collection of smoothers (keyed by session_id).
"""
from __future__ import annotations

import collections
import logging
import time
from typing import Dict, Optional

from proctor_ml.config import (
    MULTI_FACE_THRESHOLD,
    MULTI_FACE_WINDOW,
    MULTI_PEOPLE_THRESHOLD,
    MULTI_PEOPLE_WINDOW,
)


class RollingSmoother:
    """Exponential moving average for continuous values."""

    def __init__(self, window: int = 5, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self.smoothed: Optional[float] = None

    def update(self, val: Optional[float]) -> Optional[float]:
        if val is None:
            return self.smoothed
        if self.smoothed is None:
            self.smoothed = val
        else:
            self.smoothed = self.alpha * val + (1 - self.alpha) * self.smoothed
        return self.smoothed


class BooleanSmoother:
    """Fire True when at least *threshold* of the last *window* values are True."""

    def __init__(self, window: int = 5, threshold: int = 3) -> None:
        self.deque: collections.deque = collections.deque(maxlen=window)
        self.threshold = threshold

    def update(
        self, val: bool, logger: Optional[logging.Logger] = None
    ) -> bool:
        self.deque.append(val)
        result = sum(self.deque) >= self.threshold
        if logger and logger.isEnabledFor(logging.DEBUG):
            msg = (
                f"BooleanSmoother window={list(self.deque)} "
                f"threshold={self.threshold} → {result}"
            )
            logger.debug(msg)
        return result


class DurationTracker:
    """Tracks continuous duration of a boolean condition."""
    
    def __init__(self) -> None:
        self.start_time: Optional[int] = None
        self.duration_ms: int = 0
        
    def update(self, is_active: bool) -> int:
        now = int(time.time() * 1000)
        if is_active:
            if self.start_time is None:
                self.start_time = now
            self.duration_ms = now - self.start_time
        else:
            self.start_time = None
            self.duration_ms = 0
        return self.duration_ms


class SessionState:
    """All smoothers for a single inference session."""

    def __init__(self) -> None:
        # alpha=0.4: slower convergence than 0.6 — a single bad landmark frame
        # shifts the smoothed angle less, reducing transient false positives.
        self.gaze_yaw = RollingSmoother(window=5, alpha=0.4)
        self.gaze_pitch = RollingSmoother(window=5, alpha=0.4)
        # MULTI_FACE_WINDOW=5, MULTI_FACE_THRESHOLD=4: requires a second face
        # to appear in ~4 of 5 consecutive frames before raising the alert.
        # Single-frame reflection artefacts are filtered out.
        self.multiple_people = BooleanSmoother(window=MULTI_PEOPLE_WINDOW, threshold=MULTI_PEOPLE_THRESHOLD)
        self.phone_detected = BooleanSmoother(window=3, threshold=2)
        self.multi_speaker = BooleanSmoother(window=3, threshold=2)
        self.speech_confidence = RollingSmoother(window=3, alpha=0.6)
        self.multiple_faces = BooleanSmoother(window=MULTI_FACE_WINDOW, threshold=MULTI_FACE_THRESHOLD)
        
        # Duration Trackers
        self.look_away_duration = DurationTracker()
        self.phone_duration = DurationTracker()
        self.multiple_people_duration = DurationTracker()
        self.multiple_faces_duration = DurationTracker()
        self.face_missing_duration = DurationTracker()


# Global registry — keyed by opaque session_id strings supplied by the caller
SESSION_STATES: Dict[str, SessionState] = collections.defaultdict(SessionState)
