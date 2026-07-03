"""
Temporal smoothing helpers.

RollingSmoother  — exponential moving average for continuous values (yaw, pitch).
BooleanSmoother  — majority-vote window for boolean signals (phone, multiple faces).
SessionState     — per-session collection of smoothers (keyed by session_id).
"""
from __future__ import annotations

import collections
import logging
from typing import Dict, Optional


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


class SessionState:
    """All smoothers for a single inference session."""

    def __init__(self) -> None:
        # alpha=0.6 converges fast: a 90° turn reaches ~54° on frame 1,
        # immediately crossing the 25° lookingAway threshold.
        self.gaze_yaw = RollingSmoother(window=3, alpha=0.6)
        self.gaze_pitch = RollingSmoother(window=3, alpha=0.6)
        # 2-of-3 fires in ~1 s at 2 fps while filtering single-frame glitches
        self.multiple_people = BooleanSmoother(window=3, threshold=2)
        self.phone_detected = BooleanSmoother(window=3, threshold=2)
        self.multi_speaker = BooleanSmoother(window=3, threshold=2)
        self.speech_confidence = RollingSmoother(window=3, alpha=0.6)
        self.multiple_faces = BooleanSmoother(window=3, threshold=2)


# Global registry — keyed by opaque session_id strings supplied by the caller
SESSION_STATES: Dict[str, SessionState] = collections.defaultdict(SessionState)
