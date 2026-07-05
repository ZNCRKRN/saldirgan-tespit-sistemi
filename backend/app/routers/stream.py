"""Canlı akış (WebSocket) ve video yükleyip analiz etme endpoint'leri.

WebSocket her karede şunu yayınlar:
  { "image": <jpeg-base64>, "result": {...}, "alert": {...}|null }
"""
from __future__ import annotations

import asyncio
import base64
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from .. import crud
from ..config import settings
from ..database import SessionLocal, get_db
from ..ml.demo_source import DemoScene
from ..ml.pipeline import get_pipeline

router = APIRouter(tags=["stream"])


def _encode_jpeg(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode("ascii") if ok else ""


def _open_capture(source: str):
    """Kamera kaynağını aç. '0' -> webcam, dosya yolu veya RTSP/HTTP URL.

    Webcam için Windows'ta DSHOW backend'i kullanılır (MSMF sık başarısız
    oluyor) ve birkaç kez denenir: tarayıcı yenilendiğinde eski bağlantı
    cihazı ~1-2 sn rehin tutabiliyor.
    """
    if not source.isdigit():
        return cv2.VideoCapture(source)

    idx = int(source)
    for attempt in range(3):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            # Tamponu küçült: en güncel kare gelsin (gecikme birikmesin)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
        time.sleep(0.8)  # cihazın serbest kalmasını bekle
    return cv2.VideoCapture(idx)  # son çare: varsayılan backend


@router.websocket("/ws/stream/{camera_id}")
async def stream(websocket: WebSocket, camera_id: int):
    await websocket.accept()
    pipeline = get_pipeline()
    db: Session = SessionLocal()

    cap = None
    demo = None
    try:
        from ..models import Camera

        cam = db.get(Camera, camera_id)
        source = cam.source if cam else "demo"

        demo = DemoScene() if source == "demo" else None
        # Açılış (webcam'de yeniden denemeli) olay döngüsünü kilitlemesin
        cap = None if demo else await asyncio.to_thread(_open_capture, source)
        if cap is not None and not cap.isOpened():
            await websocket.send_json({"error": f"Kaynak açılamadı: {source}"})
            await websocket.close()
            return

        frame_interval = 1.0 / max(settings.target_fps, 1)
        snapshot_cooldown = 0.0  # aynı olayı saniyede bir kez kaydet

        while True:
            tick = time.time()

            if demo is not None:
                # Demo sahnesi sentetik olduğundan gerçek (görüntü-tabanlı)
                # şiddet modeli yerine kişi-bazlı sezgisel sınıflandırıcı çalışır.
                frame, persons = demo.next()
                result = pipeline.process(frame, persons=persons)
            else:
                ok, frame = cap.read()
                if not ok:
                    # Dosya bittiyse başa sar (canlı his). Tamponu da sıfırla:
                    # son kareler + ilk kareler yan yana gelirse modele ani
                    # içerik sıçraması "şiddet" gibi görünebiliyor.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pipeline.reset_stream(str(camera_id))
                    await asyncio.sleep(0.05)
                    continue
                # Gerçek şiddet modelini ayrı bir iş parçacığında çalıştır
                # (CPU çıkarımı olay döngüsünü kilitlemesin).
                scene = await asyncio.to_thread(
                    pipeline.score_scene, frame, str(camera_id)
                )
                if pipeline.clip is not None and scene is None:
                    # Isınma (tampon doluyor): sezgisel yedeğe DÜŞME — o,
                    # sentetik iskelet titremesinden sahte "saldırgan"
                    # alarmı üretebiliyor. Skoru 0 kabul et, bekle.
                    result = await asyncio.to_thread(
                        pipeline.process, frame,
                        warming=True, stream_id=str(camera_id),
                    )
                else:
                    result = await asyncio.to_thread(
                        pipeline.process, frame,
                        scene_score=scene, stream_id=str(camera_id),
                    )

            annotated = pipeline.annotate(frame, result)

            # Saldırgan/şüpheli ise kalıcı kayıt + uyarı
            alert_payload = None
            if result.persons and any(p.label != "normal" for p in result.persons):
                snap_path = None
                if (
                    settings.save_snapshots
                    and result.has_attacker
                    and tick > snapshot_cooldown
                ):
                    fname = f"{uuid.uuid4().hex}.jpg"
                    fpath = settings.snapshot_dir / fname
                    cv2.imwrite(str(fpath), annotated)
                    snap_path = f"/snapshots/{fname}"
                    snapshot_cooldown = tick + 1.0

                alert = crud.record_frame_result(
                    db, result, camera_id=camera_id, snapshot_path=snap_path
                )
                if alert is not None:
                    alert_payload = {
                        "id": alert.id,
                        "severity": alert.severity,
                        "message": alert.message,
                        "created_at": alert.created_at.isoformat(),
                    }

            await websocket.send_json(
                {
                    "image": _encode_jpeg(annotated),
                    "result": result.to_dict(),
                    "alert": alert_payload,
                }
            )

            elapsed = time.time() - tick
            await asyncio.sleep(max(0.0, frame_interval - elapsed))

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:  # noqa: BLE001
            pass
    finally:
        if cap is not None:
            cap.release()
        pipeline.reset_stream(str(camera_id))
        db.close()


@router.post("/api/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    camera_id: int | None = None,
    db: Session = Depends(get_db),
):
    """Yüklenen bir videoyu kare kare analiz eder ve özet döndürür.

    İş paketi 4: "test-doğrulama senaryoları" için toplu video analizi.
    """
    if not file.filename:
        raise HTTPException(400, "Dosya adı yok")

    dest = settings.upload_dir / f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    with dest.open("wb") as f:
        f.write(await file.read())

    pipeline = get_pipeline()
    cap = cv2.VideoCapture(str(dest))
    if not cap.isOpened():
        cap.release()
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Video açılamadı (format desteklenmiyor olabilir)")

    max_threat = 0.0
    attacker_windows = 0
    alerts_created = 0

    if pipeline.clip is not None:
        # ── Gerçek model: videoyu pencerelere bölüp her birini sınıflandır ──
        # Eğitim dağılımıyla uyum: model ~5 sn'ye eşit yayılmış 20 kare
        # gördü; örnekleme adımı 20 karenin ~5 sn'yi kapsamasına göre seçilir.
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(fps / 4))
        frames = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frames.append(frame)
            idx += 1
        cap.release()

        n = pipeline.clip.num_frames
        if len(frames) <= n:
            windows = [frames] if frames else []
        else:
            num_win = min(12, max(1, len(frames) // n))
            starts = np.linspace(0, len(frames) - n, num_win, dtype=int)
            windows = [frames[s:s + n] for s in starts]

        for w in windows:
            score = pipeline.clip.infer(w)
            # Canlı akışla aynı hareket-enerjisi kapısı: düşük hareketli
            # pencerelerde (sohbet, el sıkışma) skor bastırılır.
            if settings.motion_gate:
                m = pipeline._motion_energy(w)
                floor = max(0.1, settings.motion_floor)
                if m < floor:
                    score *= (m / floor) ** 2
            max_threat = max(max_threat, score)
            mid = w[len(w) // 2]
            result = pipeline.process(mid, scene_score=score)
            if result.has_attacker:
                attacker_windows += 1
            alert = crud.record_frame_result(db, result, camera_id=camera_id)
            if alert is not None:
                alerts_created += 1

        processed = len(windows)
        verdict_unit = "pencere"
        dest.unlink(missing_ok=True)  # geçici yüklemeyi temizle
    else:
        # ── Model yoksa: eski kare-bazlı sezgisel analiz ──
        processed = 0
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % 2 != 0:
                continue
            processed += 1
            result = pipeline.process(frame)
            max_threat = max(max_threat, result.max_threat)
            if result.has_attacker:
                attacker_windows += 1
            alert = crud.record_frame_result(db, result, camera_id=camera_id)
            if alert is not None:
                alerts_created += 1
        cap.release()
        verdict_unit = "kare"
        dest.unlink(missing_ok=True)  # geçici yüklemeyi temizle

    return {
        "filename": file.filename,
        "processed_frames": processed,
        "attacker_frames": attacker_windows,
        "max_threat_score": round(max_threat, 3),
        "alerts_created": alerts_created,
        "verdict": (
            f"SALDIRGAN/ŞİDDET TESPİT EDİLDİ ({attacker_windows} {verdict_unit})"
            if attacker_windows else "Temiz"
        ),
    }
