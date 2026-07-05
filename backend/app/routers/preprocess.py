"""Ön işleme önizleme endpoint'i (form 2.3 kanıtı).

Yüklenen tek bir karede ön işleme aşamalarını (gürültü azaltma, histogram
eşitleme, CLAHE aydınlatma düzeltme) yan yana bir montaj görüntüsü olarak
döndürür. Arka plan ayrıştırma + bölütleme zamansal olduğundan canlı akışta
gösterilir (bkz. pipeline.annotate); bu endpoint tek-kare tekniklerini
kanıtlar. Sınıflandırıcıdan bağımsızdır.
"""
from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Response, UploadFile

from ..ml.preprocess import FramePreprocessor

router = APIRouter(prefix="/api/preprocess", tags=["preprocess"])

_pre = FramePreprocessor()


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Panelin üstüne başlık şeridi ekle."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (25, 25, 25), -1)
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


@router.post("/preview")
async def preview(file: UploadFile = File(...)):
    """Bir görüntünün ön işleme aşamalarını 2×2 montaj JPEG olarak döndürür."""
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Görüntü çözülemedi (JPEG/PNG bekleniyor)")

    # Standart boyuta getir (montaj tutarlı olsun)
    h, w = frame.shape[:2]
    scale = 400.0 / max(w, 1)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    original = _label(frame, "1. Orijinal")
    denoised = _label(_pre.reduce_noise(frame), "2. Gurultu Azaltma")
    hist_eq = _label(_pre.equalize_histogram(frame), "3. Histogram Esitleme")
    clahe = _label(_pre.equalize_lighting(frame), "4. CLAHE Aydinlatma")

    top = np.hstack([original, denoised])
    bottom = np.hstack([hist_eq, clahe])
    montage = np.vstack([top, bottom])

    ok, buf = cv2.imencode(".jpg", montage, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(500, "Montaj üretilemedi")
    return Response(content=buf.tobytes(), media_type="image/jpeg")
