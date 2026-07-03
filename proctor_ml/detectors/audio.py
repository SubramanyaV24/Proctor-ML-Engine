"""
Audio Voice Activity Detection (VAD).

Primary:  Silero VAD  (neural, high accuracy)
Fallback: WebRTC VAD  (frame-based, low latency)

Returns (multi_speaker: bool, speech_confidence: float).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from proctor_ml.models.loader import Models


def detect_speech(audio_bytes: bytes, models: Models) -> Tuple[bool, float]:
    """Run VAD on raw 16-bit mono PCM bytes at 16 kHz.

    Returns:
        multi_speaker   – True when multiple speakers / long continuous speech detected.
        speech_confidence – 0.0–1.0 fraction of the clip that is speech.
    """
    # ---- Silero (neural) ------------------------------------------------
    if models.silero_vad is not None and models.silero_get_timestamps is not None:
        import torch
        waveform = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        waveform_t = torch.from_numpy(waveform)
        speech_ts = models.silero_get_timestamps(
            waveform_t, models.silero_vad, sampling_rate=16000
        )
        if not speech_ts:
            return False, 0.0

        speech_samples = sum(seg["end"] - seg["start"] for seg in speech_ts)
        confidence = min(1.0, speech_samples / max(1, len(audio_bytes)) * 2)

        gaps_ms = [
            (speech_ts[i]["start"] - speech_ts[i - 1]["end"]) * 1000 / 16000
            for i in range(1, len(speech_ts))
        ]
        multi = len(speech_ts) > 1 and any(g < 300 for g in gaps_ms)
        return multi, float(confidence)

    # ---- WebRTC VAD (fallback) ------------------------------------------
    if models.webrtc_vad is None:
        return False, 0.0

    frame_len = 960  # 16 kHz × 30 ms × 2 bytes (16-bit)
    speech_frames = 0
    total_frames = 0
    max_contiguous = cur_contiguous = 0

    # Use memoryview for zero-copy frame slicing
    audio_view = memoryview(audio_bytes)
    for i in range(0, len(audio_bytes) - frame_len + 1, frame_len):
        frame = bytes(audio_view[i : i + frame_len])
        total_frames += 1
        try:
            is_speech = models.webrtc_vad.is_speech(frame, 16000)
        except Exception:
            is_speech = False
        if is_speech:
            speech_frames += 1
            cur_contiguous += 1
            max_contiguous = max(max_contiguous, cur_contiguous)
        else:
            cur_contiguous = 0

    if total_frames == 0:
        return False, 0.0

    confidence = speech_frames / total_frames
    multi = max_contiguous >= 75  # ~2.25 s of continuous speech
    return multi, float(confidence)
