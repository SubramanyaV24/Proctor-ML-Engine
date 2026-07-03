"""
FastAPI routes for the Proctor ML Engine.

Rules are owned entirely by Frappe — this server does not have or need a
rules.json. Rule evaluation happens via /evaluate → Frappe FRAPPE_EVAL_URL.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from proctor_ml.pipeline.analyze import analyze_audio, analyze_image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — all env-driven
# ---------------------------------------------------------------------------
FRAPPE_BASE_URL = os.getenv("FRAPPE_BASE_URL", "http://10.80.4.71")
FRAPPE_EVAL_URL = os.getenv(
    "FRAPPE_EVAL_URL",
    f"{FRAPPE_BASE_URL}/api/method/ai_apps.api.proctor_api.evaluate_event.evaluate_event",
)
# Accept requests from any origin — nginx is the actual perimeter.
# Set ALLOWED_ORIGINS=https://exam.example.com,https://other.com to restrict.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS: list[str] = _raw_origins.split(",") if _raw_origins != "*" else ["*"]

PROCTOR_API_TOKEN = os.getenv("PROCTOR_API_TOKEN", "")

MAX_BYTES = 5 * 1024 * 1024  # 5 MB

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class EventPayload(BaseModel):
    event_type: str = Field(..., description="The type of event, e.g. GAZE_AWAY")
    payload: Dict[str, Any] = Field(..., description="Arbitrary event data required by the rule")
    evidence: Optional[str] = Field(None, description="Base64-encoded image evidence of the violation")
    timestamp_ms: int | None = Field(None, description="Optional epoch ms; server fills if omitted")
    session_id: str | None = Field(None, description="Optional session_id to associate with Frappe events")
    session: str | None = Field(None, description="Optional session parameter alias")

    class Config:
        extra = "allow"


class ViolationResponse(BaseModel):
    rule_id: str
    severity: str
    penalty: float
    timestamp_ms: int
    details: Dict[str, Any]


class SessionStartRequest(BaseModel):
    session_id: Optional[str] = Field(None)
    user_id: str = Field(...)
    entity_id: Optional[str] = Field(None)
    entity_type: Optional[str] = Field(None)
    user_name: Optional[str] = Field(None)
    user_email: Optional[str] = Field(None)
    client_ip: Optional[str] = Field(None)
    metadata: dict = Field(default_factory=dict)


class SessionStartResponse(BaseModel):
    session_id: str
    created_at: datetime.datetime
    expires_at: datetime.datetime
    user_id: Optional[str] = None
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None
    user_name: Optional[str] = None
    user_email: Optional[str] = None


class RunAllRequest(BaseModel):
    image_base64: str = Field(...)
    audio_base64: Optional[str] = Field(None)
    audio: Optional[str] = Field(None)
    session_id: Optional[str] = Field("default")


class RunAllResponse(BaseModel):
    analysis: dict
    head_pose: dict | None = None
    audio: dict | None = None


class AudioAnalyzeRequest(BaseModel):
    audio_base64: str
    session_id: Optional[str] = "default"


class AudioAnalyzeResponse(BaseModel):
    multi_speaker: bool
    speech_confidence: float


# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------
_sessions: dict[str, dict[str, Any]] = {}


async def cleanup_expired_sessions() -> None:
    from proctor_ml.pipeline.smoother import SESSION_STATES
    while True:
        await asyncio.sleep(60)
        now = datetime.datetime.utcnow()
        expired = [sid for sid, info in list(_sessions.items()) if info["expires_at"] <= now]
        for sid in expired:
            del _sessions[sid]
            SESSION_STATES.pop(sid, None)
            logger.info(f"Cleaned up expired session: {sid}")


# ---------------------------------------------------------------------------
# Event forwarding to Frappe
# Shared persistent AsyncClient — reuses TCP connections across all requests.
# ---------------------------------------------------------------------------
async def _process_event(event: EventPayload, http_client: httpx.AsyncClient) -> List[ViolationResponse]:
    forward_payload = {
        "event_type": event.event_type,
        "payload": event.payload,
        "evidence": event.evidence,
        "timestamp_ms": event.timestamp_ms or int(time.time() * 1000),
        "session_id": event.session_id or event.session,
    }
    logger.info(
        f"Processing event: {event.event_type} for session "
        f"{forward_payload['session_id']} (has_evidence: {bool(event.evidence)})"
    )
    try:
        resp = await http_client.post(FRAPPE_EVAL_URL, json=forward_payload)
        resp.raise_for_status()
        violations = resp.json()
        logger.info(f"Received {len(violations)} violations from Frappe")
        return [ViolationResponse(**v) for v in violations]
    except Exception as exc:
        logger.error(f"Error calling Frappe endpoint: {exc}", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/models")
async def list_models(request: Request):
    m = request.app.state.models
    return {
        "yunet":     m.yunet is not None,
        "head_pose": m.head_pose is not None,
        "yolo":      m.yolo is not None,
        "audio_vad": (m.silero_vad is not None or m.webrtc_vad is not None),
    }


@router.post("/run_all", response_model=RunAllResponse)
async def run_all(req: RunAllRequest, request: Request):
    logger.info(
        f"[RunAll] image_len={len(req.image_base64)} "
        f"audio_len={len(req.audio_base64 or req.audio or '')}"
    )

    if len(req.image_base64) > MAX_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Image payload too large")

    audio_payload = req.audio_base64 or req.audio
    if audio_payload and len(audio_payload) > MAX_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Audio payload too large")

    models = request.app.state.models
    session_id = req.session_id or "default"

    # Image and audio run in parallel, off the event loop
    tasks = [asyncio.to_thread(analyze_image, req.image_base64, session_id, models)]
    if audio_payload:
        tasks.append(asyncio.to_thread(analyze_audio, audio_payload, session_id, models))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    if isinstance(results[0], Exception):
        logger.error(f"[RunAll] Image analysis failed: {results[0]}")
        analysis: dict = {"error": str(results[0])}
    else:
        analysis = results[0]

    head_pose = analysis.get("headPose") if not analysis.get("error") else None

    audio_res = None
    if audio_payload and len(results) > 1:
        if isinstance(results[1], Exception):
            logger.error(f"[RunAll] Audio analysis failed: {results[1]}")
        else:
            audio_res = results[1]

    return RunAllResponse(analysis=analysis, head_pose=head_pose, audio=audio_res)


@router.post("/audio-analyze", response_model=AudioAnalyzeResponse)
async def audio_analyze_endpoint(req: AudioAnalyzeRequest, request: Request):
    if len(req.audio_base64) > MAX_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Audio payload too large")
    result = await asyncio.to_thread(
        analyze_audio,
        req.audio_base64,
        req.session_id or "default",
        request.app.state.models,
    )
    return AudioAnalyzeResponse(**result)


@router.post("/evaluate", response_model=List[ViolationResponse])
async def evaluate_event(event: EventPayload, request: Request):
    return await _process_event(event, request.app.state.http_client)


@router.post("/api/session/{session_id}/evaluate_event")
async def evaluate_event_with_session(session_id: str, event: EventPayload, request: Request):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    event.session_id = session_id
    violations = await _process_event(event, request.app.state.http_client)
    return {"session_id": session_id, "violations": violations}


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    http_client: httpx.AsyncClient = websocket.app.state.http_client
    try:
        while True:
            message_str = await websocket.receive_text()
            try:
                event = EventPayload(**json.loads(message_str))
                violations = await _process_event(event, http_client)
                await websocket.send_json([v.model_dump() for v in violations])
            except Exception as exc:
                logger.error(f"[WebSocket] Error processing message: {exc}")
                try:
                    await websocket.send_json({"error": str(exc)})
                except RuntimeError:
                    pass
    except WebSocketDisconnect:
        logger.info("[WebSocket] Connection closed by client")
    except Exception as exc:
        logger.error(f"[WebSocket] Connection error: {exc}")


@router.post("/api/session/start", response_model=SessionStartResponse, status_code=201)
async def start_session(payload: SessionStartRequest, request: Request):
    client_ip = payload.client_ip or (request.client.host if request.client else None)
    if not client_ip:
        raise HTTPException(status_code=400, detail="Unable to determine client IP")

    now = datetime.datetime.utcnow()

    if payload.session_id and payload.session_id in _sessions:
        info = _sessions[payload.session_id]
        if info["expires_at"] > now:
            return SessionStartResponse(
                session_id=payload.session_id,
                created_at=info["created_at"],
                expires_at=info["expires_at"],
                user_id=info["user_id"],
                entity_id=info["entity_id"],
                entity_type=info["entity_type"],
                user_name=info["user_name"],
                user_email=info["user_email"],
            )

    session_id = payload.session_id or str(uuid.uuid4())
    expires_at = now + datetime.timedelta(minutes=15)

    _sessions[session_id] = {
        "expires_at": expires_at,
        "created_at": now,
        "user_id": payload.user_id,
        "entity_id": payload.entity_id,
        "entity_type": payload.entity_type,
        "user_name": payload.user_name,
        "user_email": payload.user_email,
    }

    return SessionStartResponse(
        session_id=session_id,
        created_at=now,
        expires_at=expires_at,
        user_id=payload.user_id,
        entity_id=payload.entity_id,
        entity_type=payload.entity_type,
        user_name=payload.user_name,
        user_email=payload.user_email,
    )


@router.get("/api/session/{session_id}/trust-score", status_code=200)
async def get_trust_score(session_id: str):
    raise HTTPException(status_code=404, detail="Trust score service endpoint removed")


@router.post("/api/session/{session_id}/end", status_code=200)
async def end_session(session_id: str, request: Request):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del _sessions[session_id]
    return {"status": "ended", "session_id": session_id, "frappe_sync": "skipped"}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(models) -> FastAPI:
    """Build and return the FastAPI application with models pre-loaded."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http_client = httpx.AsyncClient(
            timeout=10.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
        logger.info(f"ML Service Engine started. Frappe endpoint: {FRAPPE_EVAL_URL}")
        cleanup_task = asyncio.create_task(cleanup_expired_sessions())

        yield

        cleanup_task.cancel()
        await app.state.http_client.aclose()
        logger.info("ML Service Engine shut down cleanly.")

    application = FastAPI(
        title="Proctor ML Engine API",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    application.state.models = models
    application.include_router(router)

    return application
