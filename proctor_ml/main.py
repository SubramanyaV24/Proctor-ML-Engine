"""
Proctor-ML-Engine entry point.

Exposes `app` at module level so uvicorn can import it directly:

    uvicorn proctor_ml.main:app --host 0.0.0.0 --port 8000

Models are loaded synchronously before the app object is created so they are
available immediately when the first request arrives (no cold-start on request 1).
"""
from __future__ import annotations

import logging

import uvicorn

from proctor_ml.api.routes import create_app
from proctor_ml.models.loader import load_all_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Load models once at import time — safe because uvicorn forks after import.
_models = load_all_models()

# Module-level `app` — this is what uvicorn and the systemctl shim both import.
app = create_app(_models)


def run() -> None:
    uvicorn.run("proctor_ml.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
