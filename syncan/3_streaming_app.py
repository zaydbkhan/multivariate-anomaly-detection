"""
SynCAN Anomaly Detection Server

FastAPI REST endpoint for 20-channel CAN bus anomaly scoring.
Uses a single global TranAD model trained on SynCAN normal data.

Usage:
    uv run uvicorn syncan.3_streaming_app:app --host 0.0.0.0 --port 8000
    # or
    uv run python syncan/3_streaming_app.py
"""

import json
import logging
import os
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas import AnomalySegment, AttributedDimension, TimestepResult
from src.scorer import build_segment_summaries, find_anomaly_segments, score_batch
from src.syncan_registry import SynCANRegistry
from src.utils import auto_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "syncan" / "best"))
DATA_DIR = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data" / "syncan" / "processed"))
DEVICE_STR = os.getenv("DEVICE", "cpu")

N_FEATURES = 20
ROLLING_BUFFER_BATCHES = 20

registry = SynCANRegistry(base_dir=MODEL_DIR)
device = auto_device(DEVICE_STR)

_norm_params: np.ndarray | None = None
_scorer_state: dict | None = None
_signal_labels: list[str] | None = None
_raw_data_buffer: deque | None = None


def _load_resources() -> None:
    global _norm_params, _scorer_state, _signal_labels, _raw_data_buffer
    _, cfg = registry.get_model(device)

    if _raw_data_buffer is None:
        _raw_data_buffer = deque(maxlen=ROLLING_BUFFER_BATCHES * cfg.window_size)

    if _norm_params is None:
        p = Path(DATA_DIR) / "norm_params.npy"
        if not p.exists():
            raise FileNotFoundError(f"norm_params not found at {p}")
        _norm_params = np.load(p)

    if _scorer_state is None:
        _scorer_state = registry.get_scorer_state()

    if _signal_labels is None:
        p = Path(DATA_DIR) / "signal_columns.npy"
        if p.exists():
            _signal_labels = np.load(p, allow_pickle=True).tolist()
        else:
            raise FileNotFoundError(f"signal_columns.npy not found at {p}")


# -- Pydantic models --


class SyncANScoreRequest(BaseModel):
    data: list[list[float]] = Field(
        ...,
        description=(
            "Time series matrix: outer list is timesteps, inner list has "
            f"exactly {N_FEATURES} float values. Data should be raw/unnormalized."
        ),
    )
    include_per_timestep: bool = Field(
        default=False,
        description="Include per-timestep scores in response",
    )
    include_attribution: bool = Field(
        default=True,
        description="Include segment-level feature attribution",
    )
    scoring_mode: str = Field(
        default="averaged",
        description="Scoring mode: 'phase2_only' or 'averaged'",
    )
    timestamp: str | None = Field(default=None)
    filename: str | None = Field(default=None)

    @field_validator("data")
    @classmethod
    def validate_minimum_timesteps(cls, v):
        if len(v) < 10:
            raise ValueError(
                f"At least 10 timesteps required, got {len(v)}"
            )
        return v

    @field_validator("data")
    @classmethod
    def validate_consistent_features(cls, v):
        if not v:
            return v
        n_features = len(v[0])
        for i, row in enumerate(v):
            if len(row) != n_features:
                raise ValueError(
                    f"Inconsistent feature count: row 0 has {n_features} features, "
                    f"row {i} has {len(row)}"
                )
        return v

    @field_validator("scoring_mode")
    @classmethod
    def validate_scoring_mode(cls, v):
        if v not in ("phase2_only", "averaged"):
            raise ValueError(
                f"scoring_mode must be 'phase2_only' or 'averaged', got '{v}'"
            )
        return v


class SyncANScoreResponse(BaseModel):
    n_timesteps: int
    n_features: int
    threshold: float
    n_anomalies: int
    anomaly_ratio: float
    anomaly_segments: list[AnomalySegment] = Field(default_factory=list)
    per_timestep: list[TimestepResult] | None = None
    dimension_means: list[float] = Field(default_factory=list)
    scoring_mode: str
    threshold_method: str
    timestamp: str | None = None
    filename: str | None = None


class SyncANHealthResponse(BaseModel):
    status: str
    detector: str = "tranad-syncan"
    n_features: int
    model_loaded: bool


# -- Lifespan --


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting SynCAN Anomaly Detection Server")
    logger.info("Device: %s | Model dir: %s | Data dir: %s", device, MODEL_DIR, DATA_DIR)
    try:
        _load_resources()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning("Model not yet loaded: %s", e)
    yield
    registry.clear_cache()


app = FastAPI(
    title="SynCAN Anomaly Detection",
    description="20-channel CAN bus anomaly detection using TranAD",
    version="0.1.0",
    lifespan=lifespan,
)


# -- Endpoints --


@app.get("/health", response_model=SyncANHealthResponse)
async def health():
    model_loaded = _scorer_state is not None
    try:
        _, cfg = registry.get_model(device)
        nf = cfg.n_features
    except FileNotFoundError:
        nf = N_FEATURES
    return SyncANHealthResponse(
        status="ready" if model_loaded else "starting",
        n_features=nf,
        model_loaded=model_loaded,
    )


@app.post("/score", response_model=SyncANScoreResponse)
async def score(request: SyncANScoreRequest):
    t_start = time.monotonic()

    try:
        _load_resources()
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    model, cfg = registry.get_model(device)
    threshold = _scorer_state.get("threshold", 0.5) if _scorer_state else 0.5
    baselines = np.array(_scorer_state.get("feature_baselines", [])) if _scorer_state else np.array([])
    threshold_method = _scorer_state.get("method", "unknown") if _scorer_state else "unknown"

    data = np.array(request.data, dtype=np.float64)
    if data.shape[1] != cfg.n_features:
        raise HTTPException(
            status_code=422,
            detail=f"Feature count mismatch: model expects {cfg.n_features}, got {data.shape[1]}",
        )

    min_vals = _norm_params[0]
    max_vals = _norm_params[1]
    normalized = (data - min_vals) / (max_vals - min_vals + 1e-4)

    scores = score_batch(
        model, normalized,
        window_size=cfg.window_size,
        device=str(device),
        scoring_mode=request.scoring_mode,
        batch_size=2000,
    )

    scores_1d = np.mean(scores, axis=1)
    predictions = (scores_1d > threshold).astype(int)
    n_anomalies = int(predictions.sum())

    assert _raw_data_buffer is not None
    history = np.array(_raw_data_buffer) if len(_raw_data_buffer) > 0 else None
    _raw_data_buffer.extend(data)

    segments: list[AnomalySegment] = []
    if request.include_attribution and baselines.size > 0 and n_anomalies > 0:
        raw_summaries = build_segment_summaries(
            scores, predictions, baselines,
            feature_labels=_signal_labels,
            normalized_data=data, history_data=history,
        )
        for s in raw_summaries:
            segments.append(
                AnomalySegment(
                    segment_start=s["segment_start"],
                    segment_end=s["segment_end"],
                    segment_length=s["segment_length"],
                    peak_score=s["peak_score"],
                    peak_timestamp=s["peak_timestamp"],
                    mean_score=s["mean_score"],
                    attributed_dimensions=[
                        AttributedDimension(**d)
                        for d in s["attributed_dimensions"]
                    ],
                )
            )
    elif n_anomalies > 0:
        seg_boundaries = find_anomaly_segments(predictions)
        for start, end in seg_boundaries:
            seg_1d = scores_1d[start : end + 1]
            peak_offset = int(np.argmax(seg_1d))
            segments.append(
                AnomalySegment(
                    segment_start=start,
                    segment_end=end,
                    segment_length=end - start + 1,
                    peak_score=round(float(seg_1d[peak_offset]), 6),
                    peak_timestamp=start + peak_offset,
                    mean_score=round(float(np.mean(seg_1d)), 6),
                    attributed_dimensions=[],
                )
            )

    mean_src = history if history is not None else data
    dimension_means = np.mean(mean_src, axis=0).round(6).tolist()

    per_timestep = None
    if request.include_per_timestep:
        per_timestep = [
            TimestepResult(
                index=i,
                score=round(float(scores_1d[i]), 6),
                is_anomaly=bool(predictions[i]),
            )
            for i in range(len(scores_1d))
        ]

    elapsed_ms = (time.monotonic() - t_start) * 1000
    logger.info(
        "Scored: %d timesteps, %d anomalies, %d segments, %.1fms",
        len(data), n_anomalies, len(segments), elapsed_ms,
    )

    return SyncANScoreResponse(
        n_timesteps=len(data),
        n_features=cfg.n_features,
        threshold=threshold,
        n_anomalies=n_anomalies,
        anomaly_ratio=round(n_anomalies / len(data), 6),
        anomaly_segments=segments,
        dimension_means=dimension_means,
        per_timestep=per_timestep,
        scoring_mode=request.scoring_mode,
        threshold_method=threshold_method,
        timestamp=request.timestamp,
        filename=request.filename,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting SynCAN server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
