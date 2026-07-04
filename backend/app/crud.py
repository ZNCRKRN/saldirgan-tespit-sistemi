"""Veritabanı işlemleri (olay/uyarı oluşturma, sorgular)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from . import models
from .ml.types import FrameResult


def severity_for(score: float) -> str:
    if score >= 0.85:
        return "critical"
    if score >= 0.70:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def record_frame_result(
    db: Session,
    result: FrameResult,
    camera_id: int | None,
    snapshot_path: str | None = None,
) -> models.Alert | None:
    """Bir karede saldırgan tespiti varsa Event + Alert kaydı oluşturur.

    Yalnızca saldırgan/şüpheli durumlar kalıcı yazılır (gürültüyü azaltmak için).
    Saldırgan ise bir uyarı döndürür (WebSocket ile yayınlanır).
    """
    if not result.persons:
        return None

    top = max(result.persons, key=lambda p: p.threat_score)
    if top.label == "normal":
        return None

    event = models.Event(
        camera_id=camera_id,
        label=top.label,
        threat_score=top.threat_score,
        action=top.action,
        person_count=len(result.persons),
        snapshot_path=snapshot_path,
    )
    db.add(event)
    db.flush()  # event.id almak için

    alert: models.Alert | None = None
    if top.label == "attacker":
        alert = models.Alert(
            event_id=event.id,
            severity=severity_for(top.threat_score),
            message=(
                f"Saldırgan davranış tespit edildi "
                f"(eylem: {top.action or 'bilinmiyor'}, "
                f"skor: {top.threat_score:.0%})."
            ),
        )
        db.add(alert)

    db.commit()
    if alert is not None:
        db.refresh(alert)
    return alert


def summary_stats(db: Session) -> dict:
    total_events = db.query(func.count(models.Event.id)).scalar() or 0
    total_alerts = db.query(func.count(models.Alert.id)).scalar() or 0
    attacker_events = (
        db.query(func.count(models.Event.id))
        .filter(models.Event.label == "attacker")
        .scalar()
        or 0
    )
    unack = (
        db.query(func.count(models.Alert.id))
        .filter(models.Alert.acknowledged.is_(False))
        .scalar()
        or 0
    )
    avg_threat = db.query(func.avg(models.Event.threat_score)).scalar() or 0.0
    active_cams = (
        db.query(func.count(models.Camera.id))
        .filter(models.Camera.is_active.is_(True))
        .scalar()
        or 0
    )
    return {
        "total_events": total_events,
        "total_alerts": total_alerts,
        "attacker_events": attacker_events,
        "unacknowledged_alerts": unack,
        "avg_threat_score": round(float(avg_threat), 3),
        "active_cameras": active_cams,
    }


def timeline(db: Session, hours: int = 24) -> list[dict]:
    """Son `hours` saati saatlik kovalara böler ve etiket dağılımını verir."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    events = (
        db.query(models.Event)
        .filter(models.Event.timestamp >= start)
        .all()
    )
    buckets: dict[str, dict[str, int]] = {}
    for h in range(hours, -1, -1):
        t = now - timedelta(hours=h)
        key = t.strftime("%H:00")
        buckets[key] = {"attacker": 0, "normal": 0, "suspicious": 0}

    for e in events:
        ts = e.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        key = ts.strftime("%H:00")
        if key in buckets and e.label in buckets[key]:
            buckets[key][e.label] += 1

    return [{"label": k, **v} for k, v in buckets.items()]


def severity_breakdown(db: Session) -> list[dict]:
    rows = (
        db.query(models.Alert.severity, func.count(models.Alert.id))
        .group_by(models.Alert.severity)
        .all()
    )
    return [{"severity": s, "count": c} for s, c in rows]


def recent_alerts(db: Session, limit: int = 50, only_unack: bool = False):
    q = (
        db.query(models.Alert)
        .options(joinedload(models.Alert.event))
        .order_by(models.Alert.created_at.desc())
    )
    if only_unack:
        q = q.filter(models.Alert.acknowledged.is_(False))
    return q.limit(limit).all()
