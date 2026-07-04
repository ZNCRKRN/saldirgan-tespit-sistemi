"""Veritabanı ORM modelleri.

Kamera, tespit olayı (Event) ve uyarı (Alert) kayıtlarını tutar.
İş paketi 2.6 "Veri Kaydı ve Raporlama" gereksinimini karşılar.
"""
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    location: Mapped[str] = mapped_column(String(120), default="")
    # "0" => yerel webcam indexi, bir RTSP/HTTP URL, "demo" => simülasyon
    source: Mapped[str] = mapped_column(String(255), default="demo")
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    events: Mapped[list["Event"]] = relationship(
        back_populates="camera", cascade="all, delete-orphan"
    )


class Event(Base):
    """Modelin tespit ettiği tek bir davranış olayı (saldırgan veya normal)."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[int | None] = mapped_column(
        ForeignKey("cameras.id"), nullable=True
    )
    # "attacker" (saldırgan) | "normal" | "suspicious" (şüpheli)
    label: Mapped[str] = mapped_column(String(40), default="normal", index=True)
    # Saldırganlık skoru 0-1
    threat_score: Mapped[float] = mapped_column(Float, default=0.0)
    # Modelin tahmin türü, ör. "running", "fighting", "loitering"
    action: Mapped[str] = mapped_column(String(60), default="")
    person_count: Mapped[int] = mapped_column(Integer, default=0)
    snapshot_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, index=True
    )

    camera: Mapped["Camera | None"] = relationship(back_populates="events")
    alert: Mapped["Alert | None"] = relationship(
        back_populates="event", cascade="all, delete-orphan", uselist=False
    )


class Alert(Base):
    """Saldırgan tespiti sonrası tetiklenen uyarı sinyali (İş paketi 2.5)."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), nullable=False)
    # "low" | "medium" | "high" | "critical"
    severity: Mapped[str] = mapped_column(String(20), default="medium", index=True)
    message: Mapped[str] = mapped_column(Text, default="")
    acknowledged: Mapped[bool] = mapped_column(default=False, index=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, index=True
    )

    event: Mapped["Event"] = relationship(back_populates="alert")
