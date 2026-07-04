"""Pipeline aşamaları arasında dolaşan veri tipleri."""
from __future__ import annotations

from dataclasses import dataclass, field

# OpenPose/BODY_25 benzeri 17 noktalı (COCO) iskelet bağlantıları.
# (baş, omuzlar, dirsekler, eller, kalçalar, dizler, ayaklar)
SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # baş-göz-kulak
    (5, 6),                                   # omuzlar
    (5, 7), (7, 9), (6, 8), (8, 10),          # kollar
    (5, 11), (6, 12), (11, 12),               # gövde
    (11, 13), (13, 15), (12, 14), (14, 16),   # bacaklar
]

KEYPOINT_NAMES = [
    "burun", "sol_goz", "sag_goz", "sol_kulak", "sag_kulak",
    "sol_omuz", "sag_omuz", "sol_dirsek", "sag_dirsek",
    "sol_bilek", "sag_bilek", "sol_kalca", "sag_kalca",
    "sol_diz", "sag_diz", "sol_ayak", "sag_ayak",
]


@dataclass
class BBox:
    """Piksel cinsinden sınırlayıcı kutu."""

    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float = 1.0


@dataclass
class PersonDetection:
    """Tek bir kişiye ait, pipeline boyunca zenginleştirilen tespit."""

    bbox: BBox
    score: float = 1.0                       # kişi tespit güveni (R-CNN)
    track_id: int | None = None              # zaman serisi takip kimliği
    keypoints: list[Keypoint] = field(default_factory=list)  # OpenPose
    label: str = "normal"                    # normal | suspicious | attacker
    threat_score: float = 0.0                # LSTM+Attention skoru 0-1
    action: str = ""                         # tahmin edilen eylem

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.as_list(),
            "score": round(self.score, 3),
            "track_id": self.track_id,
            "label": self.label,
            "threat_score": round(self.threat_score, 3),
            "action": self.action,
            "keypoints": [
                [round(k.x, 1), round(k.y, 1), round(k.confidence, 2)]
                for k in self.keypoints
            ],
        }


@dataclass
class FrameResult:
    """Tek bir karenin tüm pipeline sonucu."""

    frame_index: int
    persons: list[PersonDetection] = field(default_factory=list)
    max_threat: float = 0.0
    has_attacker: bool = False
    scene_threat: float = 0.0          # sahne düzeyi şiddet skoru (gerçek model)
    model_name: str = ""               # skoru üreten model adı (UI rozeti için)
    warming: bool = False              # model kare tamponu doluyor (ısınma)

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "persons": [p.to_dict() for p in self.persons],
            "max_threat": round(self.max_threat, 3),
            "has_attacker": self.has_attacker,
            "person_count": len(self.persons),
            "scene_threat": round(self.scene_threat, 3),
            "model_name": self.model_name,
            "warming": self.warming,
        }
