"""Aşama 3 — Davranış Sınıflandırma (LSTM + Attention).

Form bölüm 2.4.1.3 / 2.4.1.4: Kişi başına biriken keypoint zaman serisi,
LSTM + Attention ağına verilir ve saldırganlık skoru (0-1) üretilir.

>>> EĞİTİLMİŞ MODELİN BURAYA TAKILIR <<<
Modelinizi `backend/models/` klasörüne koyun:
  - Keras/TensorFlow:  models/behavior_model.h5  (veya SavedModel klasörü)
  - PyTorch:           models/behavior_model.pt

Sistem başlangıçta bu dosyaları arar; bulursa `KerasBehaviorClassifier` /
`TorchBehaviorClassifier` devreye girer, bulamazsa hareket-temelli sezgisel
`HeuristicBehaviorClassifier` yedeği çalışır (sistem yine de uçtan uca çalışır).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..config import settings

# Saldırganlıkla ilişkilendirilen örnek eylem etiketleri.
ACTIONS = ["normal", "yürüme", "koşma", "itişme", "yumruk", "tekme", "düşme"]
AGGRESSIVE_ACTIONS = {"koşma", "itişme", "yumruk", "tekme"}


class BehaviorResult:
    __slots__ = ("threat_score", "action", "label")

    def __init__(self, threat_score: float, action: str, label: str):
        self.threat_score = threat_score
        self.action = action
        self.label = label


class BaseBehaviorClassifier(ABC):
    name: str = "base"
    is_real_model: bool = False

    @abstractmethod
    def classify(self, sequence: list[list[float]]) -> BehaviorResult:
        """sequence: (T, F) keypoint/özellik zaman serisi."""

    def _label_for(self, score: float) -> str:
        if score >= settings.threat_threshold:
            return "attacker"
        if score >= settings.threat_threshold * 0.6:
            return "suspicious"
        return "normal"


class HeuristicBehaviorClassifier(BaseBehaviorClassifier):
    """Yedek sınıflandırıcı: keypoint hareket hızı/ivmesinden skor üretir.

    Eğitilmiş model olmadan makul bir demo davranışı sağlar: ani ve büyük
    hareketler (koşma/itişme/yumruk) yüksek skor üretir.
    """

    name = "Sezgisel (yedek)"
    is_real_model = False

    def classify(self, sequence: list[list[float]]) -> BehaviorResult:
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2:
            return BehaviorResult(0.0, "normal", "normal")

        # Ardışık kareler arası ortalama hareket büyüklüğü (hız).
        # Özellikler [0,1] normalize olduğundan katsayılar buna göre seçildi.
        deltas = np.diff(arr, axis=0)
        speed = float(np.mean(np.abs(deltas)))
        # İvme (hareketin ani değişimi) — saldırgan eylemlerde yüksek
        accel = float(np.mean(np.abs(np.diff(deltas, axis=0)))) if arr.shape[0] > 2 else 0.0

        raw = speed * 28.0 + accel * 55.0
        score = float(1.0 / (1.0 + np.exp(-(raw - 2.0))))  # sigmoid eşikleme

        if score >= 0.75:
            action = "yumruk" if accel > speed else "koşma"
        elif score >= 0.5:
            action = "itişme"
        elif score >= 0.3:
            action = "yürüme"
        else:
            action = "normal"

        return BehaviorResult(score, action, self._label_for(score))


class KerasBehaviorClassifier(BaseBehaviorClassifier):
    """Eğitilmiş Keras/TensorFlow LSTM+Attention modeli için sarmalayıcı."""

    name = "LSTM+Attention (Keras)"
    is_real_model = True

    def __init__(self, model_path: Path) -> None:
        import tensorflow as tf  # yerel import: yalnızca model varsa gerekli

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.seq_len = settings.sequence_length

    def classify(self, sequence: list[list[float]]) -> BehaviorResult:
        x = self._prepare(sequence)
        preds = self.model.predict(x, verbose=0)[0]
        return self._interpret(preds)

    def _prepare(self, sequence: list[list[float]]) -> np.ndarray:
        arr = np.asarray(sequence, dtype=np.float32)
        # Pencereyi sabit uzunluğa getir (baştan pad / sondan kırp)
        if arr.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - arr.shape[0], arr.shape[1]), np.float32)
            arr = np.vstack([pad, arr])
        else:
            arr = arr[-self.seq_len:]
        return arr[np.newaxis, ...]  # (1, T, F)

    def _interpret(self, preds: np.ndarray) -> BehaviorResult:
        preds = np.atleast_1d(preds)
        if preds.shape[-1] == 1:  # tek nöron => saldırgan olasılığı
            score = float(preds.reshape(-1)[0])
            action = "saldırgan" if score >= settings.threat_threshold else "normal"
        else:  # çok sınıflı softmax
            idx = int(np.argmax(preds))
            action = ACTIONS[idx] if idx < len(ACTIONS) else f"sınıf_{idx}"
            score = float(np.sum([preds[i] for i, a in enumerate(ACTIONS)
                                  if i < len(preds) and a in AGGRESSIVE_ACTIONS]))
        return BehaviorResult(score, action, self._label_for(score))


class TorchBehaviorClassifier(BaseBehaviorClassifier):
    """Eğitilmiş PyTorch LSTM+Attention modeli için sarmalayıcı."""

    name = "LSTM+Attention (PyTorch)"
    is_real_model = True

    def __init__(self, model_path: Path) -> None:
        import torch

        self.torch = torch
        self.model = torch.load(model_path, map_location="cpu")
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.seq_len = settings.sequence_length

    def classify(self, sequence: list[list[float]]) -> BehaviorResult:
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - arr.shape[0], arr.shape[1]), np.float32)
            arr = np.vstack([pad, arr])
        else:
            arr = arr[-self.seq_len:]
        x = self.torch.from_numpy(arr[np.newaxis, ...])
        with self.torch.no_grad():
            out = self.model(x)
        preds = out.detach().cpu().numpy().reshape(-1)
        score = float(preds[0]) if preds.size == 1 else float(np.max(preds))
        action = "saldırgan" if score >= settings.threat_threshold else "normal"
        return BehaviorResult(score, action, self._label_for(score))


def build_behavior_classifier() -> BaseBehaviorClassifier:
    """models/ klasöründe eğitilmiş model ararsa onu, yoksa sezgiseli kur."""
    md = settings.model_dir
    keras_candidates = [md / "behavior_model.h5", md / "behavior_model.keras",
                        md / "behavior_model"]
    torch_candidates = [md / "behavior_model.pt", md / "behavior_model.pth"]

    for p in keras_candidates:
        if p.exists():
            try:
                return KerasBehaviorClassifier(p)
            except Exception as exc:  # noqa: BLE001
                print(f"[ML] Keras modeli yüklenemedi ({p.name}): {exc}")

    for p in torch_candidates:
        if p.exists():
            try:
                return TorchBehaviorClassifier(p)
            except Exception as exc:  # noqa: BLE001
                print(f"[ML] PyTorch modeli yüklenemedi ({p.name}): {exc}")

    return HeuristicBehaviorClassifier()
