# Proctor-ML-Engine

A **generic AI inference service** for proctoring.  
Its only responsibility: accept media data → run ML models → return detections.

It knows nothing about candidates, exams, trust scores, Frappe, or any application layer.

---

## What it does

| Input | Models | Output |
|-------|--------|--------|
| Camera frame (base64) | MediaPipe → YuNet → Haar (face) | Face count, bounding boxes |
| Camera frame | 6DRepNet → geometric fallback (gaze) | Head pose (yaw/pitch/roll), lookingAway |
| Camera frame | YOLO11m + MobileNetV3 | Phones, laptops, books; person count |
| Audio PCM (base64) | Silero VAD → WebRTC VAD | multi_speaker, speech_confidence |

All detections use temporal smoothing (rolling / majority-vote) to suppress single-frame noise.

---

## Project structure

```
Proctor-ML-Engine/
├── proctor_ml/
│   ├── api/            # FastAPI routes (routes.py)
│   ├── pipeline/       # Orchestration (analyze.py) + smoothing (smoother.py)
│   ├── detectors/      # face.py, gaze.py, object.py, audio.py
│   ├── verifiers/      # Secondary classification (phone_verifier.py)
│   ├── models/         # loader.py — downloads & initialises all models
│   ├── preprocess/     # image.py, audio.py — decode helpers
│   ├── utils/          # logger.py
│   ├── config.py       # All env-var driven configuration
│   └── main.py         # App factory + uvicorn entry point
├── model_weights/      # Downloaded at runtime (gitignored)
├── logs/               # Session logs (gitignored)
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure (optional)
cp .env.example .env
# edit .env as needed

# 4. Start the server
uvicorn proctor_ml.main:app --host 0.0.0.0 --port 8000
```

Model weights are downloaded automatically on first startup into `./model_weights/`.

---

## Docker

```bash
docker-compose up --build
```

Model weights are persisted in a local `./model_weights` volume so they survive restarts.

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness + model status |
| GET | `/models` | Detailed model load status |
| POST | `/analyze` | Vision inference on one frame |
| POST | `/audio-analyze` | VAD on one audio clip |
| POST | `/run_all` | Vision + audio in one call |
| GET | `/benchmark` | 10-run latency benchmark |

### `/analyze` request
```json
{
  "image_base64": "data:image/jpeg;base64,/9j/...",
  "session_id": "abc123"
}
```

### `/run_all` request
```json
{
  "image_base64": "data:image/jpeg;base64,...",
  "audio_base64": "base64pcm...",
  "session_id": "abc123"
}
```

---

## Configuration

All values are environment variables (see `.env.example`).  
Key ones:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_WEIGHTS_DIR` | `./model_weights` | Where weights are stored |
| `GAZE_YAW_THRESHOLD` | `25.0` | Yaw degrees before lookingAway fires |
| `GAZE_PITCH_THRESHOLD` | `20.0` | Pitch degrees before lookingAway fires |
| `PHONE_DETECTION_THRESHOLD` | `0.60` | YOLO confidence threshold for phones |
| `PHONE_VERIFIER_ENABLED` | `True` | Enable MobileNetV3 secondary verification |
| `PHONE_VERIFIER_THRESHOLD` | `0.50` | Secondary model confidence required |

---

## Two-Stage Mobile Phone Verification

To significantly reduce false positives (e.g. YOLO confusing headphones or wall objects with phones), the engine uses a two-stage detection pipeline for phones:
1. **YOLO Candidate**: YOLO11m flags bounding boxes it believes are `cell phone`.
2. **Geometric Checks**: Candidate bounding box aspect ratio and size are verified.
3. **MobileNetV3 Verification**: A lightweight ImageNet-pretrained model evaluates the cropped bounding box (on CPU, <10ms). It maps the ImageNet cellular telephone class to a `PHONE` category. If the verification confidence falls below `PHONE_VERIFIER_THRESHOLD`, the candidate is rejected (with the highest predicted class logged as the reason).
4. **Boolean Smoother**: Only verified phones reach the temporal smoother to trigger a final `phoneDetected` alert.

Custom verifiers can be added by implementing the `VerificationResult` interface in `proctor_ml/verifiers/`.
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |

---

## Running tests

```bash
pytest tests/ -v
```
