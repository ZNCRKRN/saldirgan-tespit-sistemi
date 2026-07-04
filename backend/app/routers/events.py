"""Olay (Event) ve Uyarı (Alert) endpoint'leri."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload

from .. import crud, models, schemas
from ..database import get_db

router = APIRouter(prefix="/api", tags=["events"])


@router.get("/events", response_model=list[schemas.EventOut])
def list_events(
    label: str | None = Query(None, description="normal | suspicious | attacker"),
    camera_id: int | None = None,
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
):
    q = db.query(models.Event).order_by(models.Event.timestamp.desc())
    if label:
        q = q.filter(models.Event.label == label)
    if camera_id:
        q = q.filter(models.Event.camera_id == camera_id)
    return q.limit(limit).all()


@router.get("/alerts", response_model=list[schemas.AlertOut])
def list_alerts(
    only_unack: bool = False,
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db),
):
    return crud.recent_alerts(db, limit=limit, only_unack=only_unack)


@router.post("/alerts/{alert_id}/ack", response_model=schemas.AlertOut)
def acknowledge_alert(
    alert_id: int,
    payload: schemas.AlertAck | None = None,
    db: Session = Depends(get_db),
):
    alert = (
        db.query(models.Alert)
        .options(joinedload(models.Alert.event))
        .get(alert_id)
    )
    if not alert:
        raise HTTPException(404, "Uyarı bulunamadı")
    alert.acknowledged = True
    alert.acknowledged_by = payload.acknowledged_by if payload else "operatör"
    db.commit()
    db.refresh(alert)
    return alert


@router.delete("/events/{event_id}", status_code=204)
def delete_event(event_id: int, db: Session = Depends(get_db)):
    ev = db.get(models.Event, event_id)
    if not ev:
        raise HTTPException(404, "Olay bulunamadı")
    db.delete(ev)
    db.commit()
