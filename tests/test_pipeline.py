"""
Smoke tests for the inference pipeline.
Run with:  pytest tests/
"""
import base64

import cv2
import numpy as np
import pytest


def _make_gray_frame(h: int = 480, w: int = 640) -> str:
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf).decode()
    return f"data:image/jpeg;base64,{b64}"


@pytest.fixture(scope="module")
def models():
    from proctor_ml.models.loader import load_all_models
    return load_all_models()


def test_analyze_image_returns_dict(models):
    from proctor_ml.pipeline.analyze import analyze_image

    result = analyze_image(_make_gray_frame(), session_id="test", models=models)

    assert isinstance(result, dict)
    assert "faceDetected" in result
    assert "processingTimeMs" in result
    assert result["error"] is None


def test_analyze_image_bad_input(models):
    from proctor_ml.pipeline.analyze import analyze_image

    result = analyze_image("not_valid_base64!!", session_id="test_bad", models=models)
    assert result["error"] is not None


def test_analyze_audio_silence(models):
    from proctor_ml.pipeline.analyze import analyze_audio

    # 1 second of silence: 16000 samples × 2 bytes = 32000 bytes
    silence = bytes(32000)
    b64 = base64.b64encode(silence).decode()
    result = analyze_audio(b64, session_id="test_audio", models=models)

    assert "multi_speaker" in result
    assert "speech_confidence" in result
    assert result["multi_speaker"] is False
