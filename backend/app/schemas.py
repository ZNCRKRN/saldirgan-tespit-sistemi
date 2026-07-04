"""Pydantic şemaları (API istek/yanıt gövdeleri)."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ── Kamera ─────────────────────────────────────────────────────────
class CameraBase(BaseModel):
    name: str
    location: str = ""
    source: str = "demo"
    is_active: bool = True


class CameraCreate(CameraBase):
    pass


class CameraUpdate(BaseModel):
    name: str | None = None
    location: str | None = None
    source: str | None = None
    is_active: bool | None = None


class CameraOut(CameraBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime


# ── Olay (Event) ───────────────────────────────────────────────────
class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    camera_id: int | None
    label: str
    threat_score: float
    action: str
    person_count: int
    snapshot_path: str | None
    timestamp: datetime


# ── Uyarı (Alert) ──────────────────────────────────────────────────
class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    event_id: int
    severity: str
    message: str
    acknowledged: bool
    acknowledged_by: str | None
    created_at: datetime
    event: EventOut | None = None


class AlertAck(BaseModel):
    acknowledged_by: str = "operatör"


# ── Raporlama ──────────────────────────────────────────────────────
class StatsSummary(BaseModel):
    total_events: int
    total_alerts: int
    attacker_events: int
    unacknowledged_alerts: int
    avg_threat_score: float
    active_cameras: int


class TimeBucket(BaseModel):
    label: str
    attacker: int
    normal: int
    suspicious: int


class SeverityCount(BaseModel):
    severity: str
    count: int


class ReportResponse(BaseModel):
    summary: StatsSummary
    timeline: list[TimeBucket]
    severity_breakdown: list[SeverityCount]
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Sistem / Model durumu ──────────────────────────────────────────
class ModelStatus(BaseModel):
    pipeline: str
    person_detector: str
    pose_estimator: str
    behavior_classifier: str
    using_real_model: bool
    model_kind: str = "pose-heuristic"
    device: str = "cpu"
    frame_window: int | None = None
    val_accuracy: float | None = None
    threat_threshold: float
    sequence_length: int
