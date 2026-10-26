"""
Temporal smoothing helpers.

RollingSmoother           — exponential moving average for continuous values (yaw, pitch).
BooleanSmoother           — majority-vote window for boolean signals (phone, multiple faces).
DurationTracker           — tracks how long a condition has been continuously active.
ViolationCooldownTracker  — suppresses repeated violation signals so candidates have
                            time to self-correct after seeing a warning.
SessionState              — per-session collection of smoothers (keyed by session_id).
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
    PHONE_SMOOTHER_WINDOW,
    PHONE_SMOOTHER_THRESHOLD,
    NOISE_SMOOTHER_WINDOW,
    NOISE_SMOOTHER_THRESHOLD,
    GAZE_EMA_ALPHA,
    VIOLATION_COOLDOWN_FIRST_S,
    VIOLATION_COOLDOWN_REPEAT_S,
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


class ViolationCooldownTracker:
    """Suppress repeated violation signals after one has just been raised.

    The first time a violation fires, a longer cooldown (VIOLATION_COOLDOWN_FIRST_S)
    is applied so the candidate has ample time to self-correct before the next
    alert.  All subsequent violations use a shorter cooldown
    (VIOLATION_COOLDOWN_REPEAT_S), keeping them responsive while still avoiding
    rapid-fire duplicate messages.

    Usage::

        tracker = ViolationCooldownTracker()
        raw_violation: bool = <smoothed detector output>
        if tracker.allow(raw_violation):
            # emit violation to downstream / frontend
    """

    def __init__(
        self,
        first_cooldown_s: float = VIOLATION_COOLDOWN_FIRST_S,
        repeat_cooldown_s: float = VIOLATION_COOLDOWN_REPEAT_S,
    ) -> None:
        self._first_cooldown_s = first_cooldown_s
        self._repeat_cooldown_s = repeat_cooldown_s
        self._violation_count: int = 0        # total violations emitted so far
        self._last_violation_ts: float = 0.0  # epoch seconds of last emission

    @property
    def violation_count(self) -> int:
        return self._violation_count

    def allow(self, is_active: bool) -> bool:
        """Return True if the violation should be emitted right now.

        Args:
            is_active: Raw (already-smoothed) violation flag for this frame.

        Returns:
            True  → violation is active AND the cooldown period has elapsed,
                    so the caller should emit a violation event.
            False → either the condition is not active, or we are still within
                    the cooldown window (suppress the duplicate).
        """
        if not is_active:
            return False

        now = time.time()
        cooldown = (
            self._first_cooldown_s
            if self._violation_count == 0
            else self._repeat_cooldown_s
        )

        if now - self._last_violation_ts < cooldown:
            # Still within the cooldown window — suppress
            return False

        # Cooldown has elapsed: emit the violation
        self._violation_count += 1
        self._last_violation_ts = now
        return True

    def reset(self) -> None:
        """Reset state (e.g. when the session ends or the condition clears)."""
        self._violation_count = 0
        self._last_violation_ts = 0.0


class SessionState:
    """All smoothers for a single inference session."""

    def __init__(self) -> None:
        # --- Gaze / head pose ---
        # GAZE_EMA_ALPHA (default 0.6, was 0.4) for faster EMA convergence
        # so "looking away" is identified in fewer frames.
        self.gaze_yaw = RollingSmoother(window=5, alpha=GAZE_EMA_ALPHA)
        self.gaze_pitch = RollingSmoother(window=5, alpha=GAZE_EMA_ALPHA)

        # --- Multiple faces / people (unchanged — these need higher certainty) ---
        # MULTI_FACE_WINDOW=5, MULTI_FACE_THRESHOLD=4: requires a second face
        # to appear in ~4 of 5 consecutive frames before raising the alert.
        # Single-frame reflection artefacts are filtered out.
        self.multiple_people = BooleanSmoother(window=MULTI_PEOPLE_WINDOW, threshold=MULTI_PEOPLE_THRESHOLD)
        self.multiple_faces = BooleanSmoother(window=MULTI_FACE_WINDOW, threshold=MULTI_FACE_THRESHOLD)

        # --- Phone detection (faster: 2-of-2, was 2-of-3) ---
        self.phone_detected = BooleanSmoother(
            window=PHONE_SMOOTHER_WINDOW, threshold=PHONE_SMOOTHER_THRESHOLD
        )

        # --- Audio / noise (faster: 2-of-2, was 2-of-3) ---
        self.multi_speaker = BooleanSmoother(
            window=NOISE_SMOOTHER_WINDOW, threshold=NOISE_SMOOTHER_THRESHOLD
        )
        self.speech_confidence = RollingSmoother(window=3, alpha=0.6)

        # --- Duration trackers ---
        self.look_away_duration = DurationTracker()
        self.phone_duration = DurationTracker()
        self.multiple_people_duration = DurationTracker()
        self.multiple_faces_duration = DurationTracker()
        self.face_missing_duration = DurationTracker()

        # --- Post-violation cooldown trackers ---
        # These gate whether a smoothed violation should be forwarded to the
        # caller.  First violation: 8 s cooldown; subsequent: 4 s cooldown.
        self.phone_cooldown = ViolationCooldownTracker()
        self.look_away_cooldown = ViolationCooldownTracker()
        self.face_missing_cooldown = ViolationCooldownTracker()
        self.noise_cooldown = ViolationCooldownTracker()


# Global registry — keyed by opaque session_id strings supplied by the caller
SESSION_STATES: Dict[str, SessionState] = collections.defaultdict(SessionState)
