"""Aşama 2 — Poz/İskelet Çıkarımı (OpenPose).

Form bölüm 2.4.1.2: OpenPose ile her kişinin vücut eklem noktaları
(keypoints) çıkarılır ve iskelet kurulur. Bu noktalar LSTM girişine
özellik vektörü olur.

Yedek olarak, bbox geometrisinden türetilen sentetik bir iskelet üretilir
(gerçek OpenPose/MediaPipe ağırlıkları gelene kadar pipeline'ı çalıştırır).
Gerçek poz tahmini `MediaPipePoseEstimator` veya `OpenPoseEstimator`
içine takılır.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .types import Keypoint, PersonDetection


class BasePoseEstimator(ABC):
    name: str = "base"

    @abstractmethod
    def estimate(self, frame: np.ndarray, person: PersonDetection) -> list[Keypoint]:
        ...


class SyntheticPoseEstimator(BasePoseEstimator):
    """Bbox'tan oransal olarak 17 COCO keypoint üreten yedek.

    Gerçek poz vermez; iskelet görselleştirmesi ve özellik akışının uçtan
    uca çalışmasını sağlar. Eğitilmiş model gelince bu sınıf değiştirilir.
    """

    name = "Sentetik-İskelet (yedek)"

    # bbox içinde (oran_x, oran_y) olarak yaklaşık COCO keypoint yerleşimi
    _LAYOUT = [
        (0.50, 0.10), (0.45, 0.08), (0.55, 0.08), (0.40, 0.10), (0.60, 0.10),
        (0.35, 0.28), (0.65, 0.28), (0.28, 0.45), (0.72, 0.45),
        (0.25, 0.60), (0.75, 0.60), (0.40, 0.58), (0.60, 0.58),
        (0.38, 0.78), (0.62, 0.78), (0.38, 0.97), (0.62, 0.97),
    ]

    def estimate(self, frame: np.ndarray, person: PersonDetection) -> list[Keypoint]:
        b = person.bbox
        return [
            Keypoint(x=b.x + rx * b.w, y=b.y + ry * b.h, confidence=0.5)
            for (rx, ry) in self._LAYOUT
        ]


class MediaPipePoseEstimator(BasePoseEstimator):
    """MediaPipe ile gerçek poz çıkarımı için takılabilir iskelet (opsiyonel)."""

    name = "MediaPipe-Pose"

    def __init__(self) -> None:
        self.pose = None
        # import mediapipe as mp
        # self.pose = mp.solutions.pose.Pose(static_image_mode=False)

    def estimate(self, frame: np.ndarray, person: PersonDetection) -> list[Keypoint]:
        if self.pose is None:
            return []
        # Kişi bölgesini kırp, MediaPipe çalıştır, 17 keypoint'e eşle.
        return []


class OpenPoseEstimator(BasePoseEstimator):
    """Gerçek OpenPose (Caffe/CMU) için takılabilir iskelet."""

    name = "OpenPose"

    def __init__(self, model_dir: str | None = None) -> None:
        self.net = None
        # OpenPose Python API veya cv2.dnn ile prototxt+caffemodel yüklenir.

    def estimate(self, frame: np.ndarray, person: PersonDetection) -> list[Keypoint]:
        if self.net is None:
            return []
        return []


def build_pose_estimator() -> BasePoseEstimator:
    """Mevcut en iyi poz tahmincisini seç; yoksa sentetik yedeğe düş."""
    mp_est = MediaPipePoseEstimator()
    if mp_est.pose is not None:
        return mp_est
    return SyntheticPoseEstimator()
