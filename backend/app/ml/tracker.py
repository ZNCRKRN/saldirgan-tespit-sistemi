"""Basit centroid tabanlı kişi takibi.

LSTM/Attention'ın zaman serisi analizi yapabilmesi için her kişiye kalıcı
bir `track_id` atanır ve kişi başına özellik geçmişi (sequence buffer)
tutulur. Form bölüm 2.4.1.3: hareketlerin zaman içindeki değişimi.
"""
from __future__ import annotations

import math
from collections import deque

from .types import PersonDetection


class _Track:
    __slots__ = ("id", "centroid", "missed", "history")

    def __init__(self, track_id: int, centroid: tuple[float, float], maxlen: int):
        self.id = track_id
        self.centroid = centroid
        self.missed = 0
        # Kişi başına özellik geçmişi (LSTM penceresi)
        self.history: deque[list[float]] = deque(maxlen=maxlen)


class CentroidTracker:
    def __init__(self, max_distance: float = 80.0, max_missed: int = 15,
                 sequence_length: int = 16) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.sequence_length = sequence_length
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}

    @staticmethod
    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update(self, persons: list[PersonDetection]) -> None:
        """Tespitlere track_id atar (in-place)."""
        unmatched = set(self._tracks.keys())

        for person in persons:
            c = person.bbox.center
            best_id, best_d = None, self.max_distance
            for tid in unmatched:
                d = self._dist(c, self._tracks[tid].centroid)
                if d < best_d:
                    best_id, best_d = tid, d

            if best_id is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _Track(tid, c, self.sequence_length)
            else:
                tid = best_id
                self._tracks[tid].centroid = c
                self._tracks[tid].missed = 0
                unmatched.discard(tid)

            person.track_id = tid

        # Eşleşmeyen track'leri yaşlandır / temizle
        for tid in list(unmatched):
            self._tracks[tid].missed += 1
            if self._tracks[tid].missed > self.max_missed:
                del self._tracks[tid]

    def push_features(self, track_id: int, features: list[float]) -> list[list[float]]:
        """Bir track'in özellik geçmişine yeni kare ekle ve pencereyi döndür."""
        track = self._tracks.get(track_id)
        if track is None:
            return [features]
        track.history.append(features)
        return list(track.history)
