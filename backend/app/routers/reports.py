"""Raporlama endpoint'leri (form bölüm 2.6 "Veri Kaydı ve Raporlama")."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from .. import crud, schemas
from ..database import get_db
from ..ml.pipeline import get_pipeline

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("/summary", response_model=schemas.StatsSummary)
def summary(db: Session = Depends(get_db)):
    return crud.summary_stats(db)


@router.get("", response_model=schemas.ReportResponse)
def full_report(
    hours: int = Query(24, ge=1, le=168), db: Session = Depends(get_db)
):
    return {
        "summary": crud.summary_stats(db),
        "timeline": crud.timeline(db, hours=hours),
        "severity_breakdown": crud.severity_breakdown(db),
    }


@router.get("/model-status", response_model=schemas.ModelStatus)
def model_status():
    return get_pipeline().status()
