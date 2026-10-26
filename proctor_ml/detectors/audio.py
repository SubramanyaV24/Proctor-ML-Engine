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


def _verify_with_whisper(
    waveform: np.ndarray, 
    multi: bool, 
    confidence: float, 
    models: Models, 
    speech_ts: list = None
) -> Tuple[bool, float]:
    """If VAD detects speech, use Faster-Whisper to confirm it contains actual words.
    
    To improve performance under high concurrency, this skips silence (using 
    Silero timestamps if available) and limits transcription to the first 
    2 seconds of actual speech. We only need to know IF words were spoken 
    (to rule out fan noise/keyboard), not transcribe the whole clip.
    """
    if not multi and confidence <= 0.0:
        return multi, confidence
        
    if models.whisper is None:
        return multi, confidence

    # 1. Strip silence by concatenating only the VAD-detected speech segments
    if speech_ts is not None and len(speech_ts) > 0:
        segments = []
        for seg in speech_ts:
            start_idx = seg["start"]
            end_idx = seg["end"]
            segments.append(waveform[start_idx:end_idx])
        if segments:
            verify_waveform = np.concatenate(segments)
        else:
            verify_waveform = waveform
    else:
        # Fallback (WebRTC path without explicit timestamps)
        verify_waveform = waveform

    # 2. Truncate to max 2 seconds (16,000 Hz * 2s = 32,000 samples)
    # Transcribing 10-30s clips was causing severe blocking on the ML Engine CPU.
    MAX_SAMPLES = 32000
    if len(verify_waveform) > MAX_SAMPLES:
        verify_waveform = verify_waveform[:MAX_SAMPLES]

    # If the speech segment is extremely short, skip whisper entirely to save CPU
    if len(verify_waveform) < 4000: # 0.25 seconds
        return multi, confidence

    try:
        segments, _ = models.whisper.transcribe(
            verify_waveform, 
            beam_size=1, 
            language="en", 
            vad_filter=False
        )
        text = "".join([segment.text for segment in segments]).strip()
        
        # If no meaningful text was decoded, it was likely background noise
        if not text:
            return False, 0.0
            
    except Exception as exc:
        print(f"[Audio] Whisper verification error: {exc}")

    return multi, confidence


def detect_speech(audio_bytes: bytes, models: Models) -> Tuple[bool, float]:
    """Run VAD on raw 16-bit mono PCM bytes at 16 kHz.

    Returns:
        multi_speaker   – True when multiple speakers / long continuous speech detected.
        speech_confidence – 0.0–1.0 fraction of the clip that is speech.
    """
    # Create waveform once for both VAD and Whisper
    waveform = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # ---- Silero (neural) ------------------------------------------------
    if models.silero_vad is not None and models.silero_get_timestamps is not None:
        import torch
        waveform_t = torch.from_numpy(waveform)
        speech_ts = models.silero_get_timestamps(
            waveform_t, 
            models.silero_vad, 
            sampling_rate=16000,
            threshold=0.3,
            speech_pad_ms=400
        )
        if not speech_ts:
            return False, 0.0

        speech_samples = sum(seg["end"] - seg["start"] for seg in speech_ts)
        confidence = min(1.0, float(speech_samples) / max(1, len(waveform)) * 2)

        gaps_ms = [
            (speech_ts[i]["start"] - speech_ts[i - 1]["end"]) * 1000 / 16000
            for i in range(1, len(speech_ts))
        ]
        multi = len(speech_ts) > 1 and any(g < 300 for g in gaps_ms)
        
        return _verify_with_whisper(waveform, multi, float(confidence), models, speech_ts)

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
    
    return _verify_with_whisper(waveform, multi, float(confidence), models)
