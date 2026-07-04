"""Demo sahne üretici.

Gerçek kamera/video olmadan sistemi uçtan uca göstermek için sentetik bir
sahne üretir: hareketli kişi siluetleri ve bunlara karşılık gelen
PersonDetection kutuları. Ara sıra "saldırgan" bir kişi (ani/hızlı hareket)
enjekte ederek uyarı akışını tetikler.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .types import BBox, PersonDetection

_W, _H = 640, 360


class DemoScene:
    def __init__(self, seed: int = 0) -> None:
        self.t = 0
        self.seed = seed
        # Her kişi: faz, hız, yarıçap, saldırganlık tetik aralığı
        self._actors = [
            {"phase": 0.0, "speed": 0.9, "y": 0.55, "amp": 0.30, "aggr_at": 70},
            {"phase": 2.1, "speed": 0.6, "y": 0.62, "amp": 0.22, "aggr_at": -1},
            {"phase": 4.0, "speed": 1.3, "y": 0.50, "amp": 0.18, "aggr_at": 130},
        ]

    def next(self) -> tuple[np.ndarray, list[PersonDetection]]:
        self.t += 1
        frame = self._background()
        persons: list[PersonDetection] = []

        for i, a in enumerate(self._actors):
            # Yatay salınım hareketi
            base = 0.5 + a["amp"] * math.sin(self.t * 0.03 * a["speed"] + a["phase"])
            # Saldırgan anlarında ani sıçrama (yüksek hız => yüksek skor)
            aggressive = a["aggr_at"] > 0 and (self.t % 200) in range(
                a["aggr_at"], a["aggr_at"] + 40
            )
            jitter = 0.11 * math.sin(self.t * 1.1) if aggressive else 0.0
            cx = int((base + jitter) * _W)
            cy = int(a["y"] * _H + (16 * math.sin(self.t * 0.7) if aggressive else 0))

            bw, bh = 46, 120
            x, y = cx - bw // 2, cy - bh // 2
            # Silueti çiz
            color = (60, 60, 60)
            cv2.rectangle(frame, (x, y + 30), (x + bw, y + bh), color, -1)
            cv2.circle(frame, (cx, y + 18), 16, color, -1)

            persons.append(
                PersonDetection(bbox=BBox(x, y, bw, bh), score=0.9)
            )
        return frame, persons

    @staticmethod
    def _background() -> np.ndarray:
        img = np.full((_H, _W, 3), 28, dtype=np.uint8)
        # Zemin ızgarası (kapalı alan hissi)
        for gx in range(0, _W, 64):
            cv2.line(img, (gx, 0), (gx, _H), (40, 40, 40), 1)
        for gy in range(0, _H, 64):
            cv2.line(img, (0, gy), (_W, gy), (40, 40, 40), 1)
        return img
