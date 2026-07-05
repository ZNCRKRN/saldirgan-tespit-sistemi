"""FastAPI uygulama giriş noktası.

Kapalı Alanlarda Derin Öğrenme Tabanlı Saldırgan Tespiti — Backend.
"""
import sys
from contextlib import asynccontextmanager

# Windows konsolu (cp1254) Unicode karakterleri (→, ç, ş…) basamadığında
# başlangıç çıktısı çökmesin diye stdout/stderr'i UTF-8'e sabitle.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .database import init_db
from .ml.pipeline import get_pipeline
from .routers import cameras, events, reports, stream


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    pipe = get_pipeline()  # modeli/yedeği başlangıçta yükle
    status = pipe.status()
    print("=" * 60)
    print(f"  {settings.app_name} v{settings.app_version}")
    print(f"  Pipeline       : {status['pipeline']}")
    print(f"  Kişi tespiti   : {status['person_detector']}")
    print(f"  Poz çıkarımı   : {status['pose_estimator']}")
    print(f"  Davranış modeli: {status['behavior_classifier']}")
    print(f"  Gerçek model?  : {'EVET' if status['using_real_model'] else 'HAYIR (yedek)'}")
    print("=" * 60)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Çok açılı video ile derin öğrenme tabanlı saldırgan tespit sistemi.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tespit snapshot'larını statik olarak sun
app.mount(
    "/snapshots",
    StaticFiles(directory=str(settings.snapshot_dir)),
    name="snapshots",
)

app.include_router(cameras.router)
app.include_router(events.router)
app.include_router(reports.router)
app.include_router(stream.router)


@app.get("/api/health", tags=["system"])
def health():
    return {"status": "ok", "app": settings.app_name, "version": settings.app_version}


# ── Derlenmiş arayüzü (frontend/dist) aynı sunucudan sun ────────────
# Böylece sistemi indiren kişi Node/npm kurmadan tek adresten kullanır:
# http://localhost:8000  → arayüz + API + WebSocket bir arada.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

if (_FRONTEND_DIST / "index.html").exists():

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        """SPA fallback: dosya varsa onu, yoksa index.html'i döndür.

        /api, /ws, /snapshots yolları yukarıdaki router'larda eşleştiği için
        buraya düşmez; burası yalnızca arayüz sayfalarını karşılar.
        """
        candidate = _FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_FRONTEND_DIST / "index.html")
