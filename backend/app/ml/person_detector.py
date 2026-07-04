"""Aşama 1 — Kişi Tespiti (R-CNN).

Form bölüm 2.4.1.1: R-CNN ile güvenlik kamerası görüntülerindeki kişiler
tespit edilir. Burada soyut bir arayüz ve OpenCV HOG tabanlı bir yedek
(mock) sağlanır. Gerçek R-CNN modeli `RCNNPersonDetector` içine takılır.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from .types import BBox, Keypoint, PersonDetection


class BasePersonDetector(ABC):
    name: str = "base"
    # True ise detect() PersonDetection.keypoints alanını da doldurur
    # (ayrı poz tahmini aşamasına gerek kalmaz).
    provides_keypoints: bool = False

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        ...


class HOGPersonDetector(BasePersonDetector):
    """OpenCV HOG + SVM yaya tespiti.

    Gerçek R-CNN ağırlıkları gelene kadar çalışan, harici bağımlılık
    gerektirmeyen bir yedektir.
    """

    name = "OpenCV-HOG (yedek)"

    def __init__(self) -> None:
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        # Hız için kareyi küçült, sonra koordinatları ölçekle.
        h, w = frame.shape[:2]
        scale = 640.0 / max(w, 1)
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
        inv = 1.0 / scale if scale < 1 else 1.0

        rects, weights = self.hog.detectMultiScale(
            small, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        out: list[PersonDetection] = []
        for (x, y, bw, bh), score in zip(rects, weights):
            out.append(
                PersonDetection(
                    bbox=BBox(int(x * inv), int(y * inv), int(bw * inv), int(bh * inv)),
                    score=float(min(max(score, 0.0), 1.0)),
                )
            )
        return out


class KeypointRCNNDetector(BasePersonDetector):
    """torchvision Keypoint R-CNN: GERÇEK kişi tespiti + GERÇEK 17-nokta
    COCO iskeleti tek modelde (form aşama 1 R-CNN + aşama 2 poz).

    GPU varsa ~50 ms/kare; `detect_stride` ile seyreltilerek canlı akışta
    kullanılır. Her tespit için keypoints alanı da doldurulur.
    """

    name = "Keypoint R-CNN (torchvision)"
    provides_keypoints = True

    def __init__(self, score_thr: float = 0.75, max_side: int = 640) -> None:
        import torch
        from torchvision.models.detection import (
            KeypointRCNN_ResNet50_FPN_Weights,
            keypointrcnn_resnet50_fpn,
        )

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        ).to(self.device).eval()
        self.score_thr = score_thr
        self.max_side = max_side
        self.name = f"{KeypointRCNNDetector.name} [{self.device.type.upper()}]"

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        h, w = frame.shape[:2]
        scale = self.max_side / max(h, w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
        inv = 1.0 / scale if scale < 1 else 1.0

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        x = self.torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255).to(self.device)
        with self.torch.no_grad():
            out = self.model([x])[0]

        persons: list[PersonDetection] = []
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        kps = out["keypoints"].cpu().numpy()          # (N, 17, 3)
        kp_scores = out["keypoints_scores"].cpu().numpy()  # (N, 17)

        for i in range(len(boxes)):
            if labels[i] != 1 or scores[i] < self.score_thr:  # 1 == person
                continue
            x1, y1, x2, y2 = (boxes[i] * inv).astype(int)
            keypoints = [
                Keypoint(
                    x=float(kps[i, j, 0] * inv),
                    y=float(kps[i, j, 1] * inv),
                    # skor logit'ini 0-1 güvene çevir; ~2 üzeri "görünür"
                    confidence=float(1.0 / (1.0 + np.exp(-kp_scores[i, j]))),
                )
                for j in range(kps.shape[1])
            ]
            persons.append(
                PersonDetection(
                    bbox=BBox(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    score=float(scores[i]),
                    keypoints=keypoints,
                )
            )
        return persons


class RCNNPersonDetector(BasePersonDetector):
    """Gerçek R-CNN dedektörü için takılabilir iskelet.

    Eğitilmiş model (torchvision Faster R-CNN, Detectron2, vb.) bu sınıfa
    yüklenir. Yüklenemezse pipeline otomatik olarak HOG yedeğine düşer.
    """

    name = "R-CNN"

    def __init__(self, weights_path: str | None = None) -> None:
        self.model = None
        self.weights_path = weights_path
        # ── Gerçek model yükleme örneği (torchvision) ──────────────
        # import torch
        # from torchvision.models.detection import fasterrcnn_resnet50_fpn
        # self.model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        # if weights_path:
        #     self.model.load_state_dict(torch.load(weights_path))
        # self.model.eval()

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        if self.model is None:
            return []
        # ── Gerçek çıkarım örneği ──────────────────────────────────
        # tensor = torch.from_numpy(frame[..., ::-1].copy()).permute(2,0,1).float()/255
        # with torch.no_grad():
        #     pred = self.model([tensor])[0]
        # COCO sınıf 1 == "person"; skor eşiği uygula ve BBox üret.
        return []


def build_person_detector() -> BasePersonDetector:
    """Gerçek Keypoint R-CNN'i kur; olmazsa HOG yedeğine düş."""
    try:
        det = KeypointRCNNDetector()
        print(f"[ML] Kişi+poz dedektörü yüklendi: {det.name}")
        return det
    except Exception as exc:  # noqa: BLE001
        print(f"[ML] Keypoint R-CNN yüklenemedi ({exc}); HOG yedeğine geçildi")
    return HOGPersonDetector()
