"""Şiddet / Saldırganlık klip sınıflandırıcısı — gerçek eğitilmiş model.

Mimari (kullanıcının Kaggle eğitimi):
    ResNet50 (ImageNet, feature extractor)
      → BiLSTM (2048 → 256, 2 kat, çift yönlü)
      → Temporal Attention
      → Sınıflandırıcı (512 → 128 → 1, Sigmoid)

Giriş : 20 ardışık RGB kare, 224×224, ImageNet normalize  → (1, 20, 3, 224, 224)
Çıkış : tek nöron sigmoid → şiddet/saldırganlık olasılığı [0, 1]
Etiket: 1 = Violence (saldırgan), 0 = NonViolence (normal)

Veri seti: "Real Life Violence Situations". Bu model SAHNE düzeyinde çalışır
(kişi-bazlı iskelet değil), bu yüzden boru hattında tek bir sahne skoru üretir
ve tespit edilen tüm kişilere yansıtılır.

Model dosyası `backend/models/best_model.pth` (checkpoint dict) olarak beklenir.
Bulunamazsa `build_violence_classifier()` None döner ve sistem sezgisel yedeğe
geri düşer.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import settings

NUM_FRAMES = 20
IMG_SIZE = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _build_arch(torch, hidden_size: int, num_layers: int, dropout: float):
    """Eğitimdeki `ViolenceDetectionModel` mimarisini birebir kurar."""
    import torch.nn as nn
    from torchvision import models

    class AttentionLayer(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.Tanh(),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, lstm_output):
            scores = self.attention(lstm_output)
            weights = torch.softmax(scores, dim=1)
            context = torch.sum(weights * lstm_output, dim=1)
            return context, weights

    class ViolenceDetectionModel(nn.Module):
        def __init__(self, hidden=256, layers=2, drop=0.3):
            super().__init__()
            resnet = models.resnet50(weights=None)  # ağırlıklar state_dict'ten gelecek
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_size = 2048
            self.lstm = nn.LSTM(
                input_size=self.feature_size,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=drop if layers > 1 else 0,
                bidirectional=True,
            )
            self.attention = AttentionLayer(hidden * 2)
            self.classifier = nn.Sequential(
                nn.Dropout(drop),
                nn.Linear(hidden * 2, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(drop),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            b, f, C, H, W = x.shape
            x = x.view(b * f, C, H, W)
            feat = self.feature_extractor(x).view(b, f, self.feature_size)
            out, _ = self.lstm(feat)
            ctx, w = self.attention(out)
            return self.classifier(ctx).squeeze(1), w

    return ViolenceDetectionModel(hidden_size, num_layers, dropout)


def _build_arch_v2(torch, hidden_size: int, num_layers: int, dropout: float):
    """Yeni (anti-overfitting) eğitimdeki mimari — feature_reducer +
    SpatialDropout + LayerNorm + dar attention. `kaggle_training_notebook (1).py`
    ile birebir aynı yapı."""
    import torch.nn as nn
    from torchvision import models

    class SpatialDropout1D(nn.Module):
        def __init__(self, p=0.2):
            super().__init__()
            self.p = p

        def forward(self, x):
            if not self.training or self.p == 0:
                return x
            mask = torch.bernoulli(
                torch.ones(x.shape[0], 1, x.shape[2], device=x.device) * (1 - self.p)
            )
            return x * mask / (1 - self.p)

    class AttentionLayer(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(hidden, hidden // 4),
                nn.Tanh(),
                nn.Dropout(0.3),
                nn.Linear(hidden // 4, 1),
            )

        def forward(self, lstm_output):
            scores = self.attention(lstm_output)
            weights = torch.softmax(scores, dim=1)
            context = torch.sum(weights * lstm_output, dim=1)
            return context, weights

    class ViolenceDetectionModelV2(nn.Module):
        def __init__(self, hidden=128, layers=1, drop=0.5):
            super().__init__()
            resnet = models.resnet50(weights=None)  # ağırlıklar state_dict'ten
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_size = 2048
            self.feature_reducer = nn.Sequential(
                nn.Linear(self.feature_size, hidden * 2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden * 2),
            )
            self.spatial_dropout = SpatialDropout1D(p=0.3)
            self.lstm = nn.LSTM(
                input_size=hidden * 2,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=0,
                bidirectional=True,
            )
            self.layer_norm = nn.LayerNorm(hidden * 2)
            self.attention = AttentionLayer(hidden * 2)
            self.classifier = nn.Sequential(
                nn.Dropout(drop),
                nn.Linear(hidden * 2, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(drop * 0.6),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            b, f, C, H, W = x.shape
            x = x.view(b * f, C, H, W)
            feat = self.feature_extractor(x).view(b * f, self.feature_size)
            feat = self.feature_reducer(feat).view(b, f, -1)
            feat = self.spatial_dropout(feat)
            out, _ = self.lstm(feat)
            out = self.layer_norm(out)
            ctx, w = self.attention(out)
            return self.classifier(ctx).squeeze(1), w

    return ViolenceDetectionModelV2(hidden_size, num_layers, dropout)


class ViolenceClipClassifier:
    """Eğitilmiş ResNet50+BiLSTM+Attention şiddet modelini saran çıkarım sınıfı."""

    name = "ResNet50+BiLSTM+Attention (PyTorch)"
    is_real_model = True

    def __init__(self, model_path: Path) -> None:
        import torch

        self.torch = torch
        # GPU varsa kullan (RTX vb.) — CPU'da ~0.5sn/klip, GPU'da ~10-30x hızlı.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
            cfg = ckpt.get("model_config", {}) or {}
            self.val_acc = ckpt.get("val_acc")
            self.test_acc = ckpt.get("test_accuracy")
            # Yeni eğitimin ek metrikleri (varsa) — UI'de gösterilir
            self.test_f1 = ckpt.get("test_f1")
            self.test_auc = ckpt.get("test_auc")
            self.classification_report = ckpt.get("classification_report")
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt  # düz state_dict
            cfg, self.val_acc, self.test_acc = {}, None, None
            self.test_f1 = self.test_auc = self.classification_report = None
        else:  # tam pickle'lanmış model
            state, cfg, self.val_acc, self.test_acc = None, {}, None, None
            self.test_f1 = self.test_auc = self.classification_report = None
            self.model = ckpt

        self.num_frames = int(cfg.get("num_frames", NUM_FRAMES))
        self.img_size = int(cfg.get("img_size", IMG_SIZE))

        if state is not None:
            # Mimari sürümünü state_dict anahtarlarından tanı:
            # v2 (yeni anti-overfitting eğitim) feature_reducer içerir.
            if any(k.startswith("feature_reducer.") for k in state):
                model = _build_arch_v2(
                    torch,
                    int(cfg.get("hidden_size", 128)),
                    int(cfg.get("num_layers", 1)),
                    float(cfg.get("dropout", 0.5)),
                )
            else:
                model = _build_arch(
                    torch,
                    int(cfg.get("hidden_size", 256)),
                    int(cfg.get("num_layers", 2)),
                    float(cfg.get("dropout", 0.3)),
                )
            model.load_state_dict(state, strict=True)
            self.model = model

        self.model.to(self.device).eval()
        self.name = f"{ViolenceClipClassifier.name} [{self.device.type.upper()}]"

    # ── Ön işleme ──────────────────────────────────────────────────
    def _sample_indices(self, n: int) -> list[int]:
        """n kareden `num_frames` adet eşit aralıklı indeks (eğitimdeki gibi)."""
        if n <= 0:
            return []
        if n >= self.num_frames:
            return list(np.linspace(0, n - 1, self.num_frames, dtype=int))
        # az kare varsa sonuncuyu tekrarla
        return list(range(n)) + [n - 1] * (self.num_frames - n)

    def _to_tensor(self, frames: list[np.ndarray]):
        idxs = self._sample_indices(len(frames))
        chans = []
        for i in idxs:
            f = frames[i]
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            f = cv2.resize(f, (self.img_size, self.img_size)).astype(np.float32) / 255.0
            f = (f - _MEAN) / _STD
            chans.append(f.transpose(2, 0, 1))  # (3,H,W)
        x = np.stack(chans)[np.newaxis, ...]  # (1,T,3,H,W)
        return self.torch.from_numpy(x).float()

    # ── Çıkarım ────────────────────────────────────────────────────
    def infer(self, frames: list[np.ndarray]) -> float:
        """Bir kare listesinden tek bir şiddet olasılığı [0,1] üretir."""
        if not frames:
            return 0.0
        x = self._to_tensor(frames).to(self.device)
        with self.torch.no_grad():
            prob, _ = self.model(x)
        return float(prob.detach().cpu().numpy().reshape(-1)[0])


def build_violence_classifier() -> ViolenceClipClassifier | None:
    """models/ (ve proje kökü) içinde eğitilmiş şiddet modeli ararsa kurar."""
    md = settings.model_dir
    candidates = [
        md / "best_model.pth",
        md / "best_model.pth.zip",
        md / "violence_model.pth",
        md / "violence_detection_full_model.pth",
        md / "violence_detection_model_weights.pth",
        # proje kökü (kullanıcı buraya bırakmış olabilir)
        md.parent.parent / "best_model.pth.zip",
        md.parent.parent / "best_model.pth",
    ]
    for p in candidates:
        if p.exists():
            try:
                clf = ViolenceClipClassifier(p)
                print(f"[ML] Şiddet modeli yüklendi: {p.name} ({clf.name})")
                return clf
            except Exception as exc:  # noqa: BLE001
                print(f"[ML] Şiddet modeli yüklenemedi ({p.name}): {exc}")
    return None
