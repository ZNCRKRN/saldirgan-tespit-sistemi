"""SQLAlchemy veritabanı kurulumu (SQLite)."""
from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings

connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)

engine = create_engine(settings.database_url, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: istek başına DB oturumu."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Tabloları oluştur ve örnek veri ekle."""
    from . import models  # noqa: F401  (modellerin metadata'ya kaydı için)

    Base.metadata.create_all(bind=engine)
    _migrate(engine)
    _seed(SessionLocal())


def _migrate(eng) -> None:
    """Hafif şema migrasyonu: eski veritabanlarına yeni kolonları ekle."""
    with eng.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(cameras)"))]
        if "zone" not in cols:
            conn.execute(
                text("ALTER TABLE cameras ADD COLUMN zone VARCHAR(100) DEFAULT ''")
            )
            conn.commit()


def _seed(db: Session) -> None:
    """İlk çalıştırmada örnek kamera kaydı oluştur (yerel webcam)."""
    from .models import Camera

    if db.query(Camera).count() == 0:
        db.add(Camera(name="Giriş Holü - Açı 1", location="Zemin Kat", source="0"))
        db.commit()
    db.close()
