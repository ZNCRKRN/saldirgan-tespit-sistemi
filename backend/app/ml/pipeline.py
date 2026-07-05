"""Tespit boru hattı (pipeline) orkestratörü.

Akış (form bölüm 2.1 "Projenin Genel Mimarisi"):
  kare → R-CNN (kişi) → OpenPose (iskelet) → takip → LSTM+Attention (skor)
        → etiketleme → görselleştirme.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime

import cv2
import numpy as np

from ..config import settings
from .behavior_classifier import build_behavior_classifier
from .person_detector import build_person_detector
from .pose_estimator import build_pose_estimator
from .tracker import CentroidTracker
from .types import BBox, FrameResult, PersonDetection, SKELETON_EDGES
from .violence_classifier import build_violence_classifier

# Etiket -> BGR renk (görselleştirme)
_COLORS = {
    "normal": (90, 200, 90),
    "suspicious": (40, 180, 240),
    "attacker": (40, 40, 230),
}


class DetectionPipeline:
    def __init__(self) -> None:
        self.person_detector = build_person_detector()
        self.pose_estimator = build_pose_estimator()
        self.classifier = build_behavior_classifier()
        # Gerçek eğitilmiş şiddet modeli (sahne düzeyi). Yoksa None → sezgisel.
        self.clip = build_violence_classifier()
        self.tracker = CentroidTracker(sequence_length=settings.sequence_length)
        self._frame_index = 0
        # Akış (kamera) başına kare tamponu ve son sahne skoru önbelleği
        self._buffers: dict[str, deque] = {}
        self._counters: dict[str, int] = {}
        self._last_scene: dict[str, float] = {}
        # Son ham çıkarım skorları (ardışık doğrulama / tek-tepe filtresi)
        self._raw_scores: dict[str, deque] = {}
        # Kişi tespiti seyreltme: akış başına son tespitler + kare sayacı
        self._last_persons: dict[str, list[PersonDetection]] = {}
        self._det_counters: dict[str, int] = {}

    # ── Durum / tanılama ───────────────────────────────────────────
    def status(self) -> dict:
        if self.clip is not None:
            behavior_name = self.clip.name
            using_real = True
            window = self.clip.num_frames
            device = self.clip.device.type
            val_acc = self.clip.val_acc
            test_acc = getattr(self.clip, "test_acc", None)
            test_f1 = getattr(self.clip, "test_f1", None)
            test_auc = getattr(self.clip, "test_auc", None)
            class_report = getattr(self.clip, "classification_report", None)
        else:
            behavior_name = self.classifier.name
            using_real = self.classifier.is_real_model
            window = settings.sequence_length
            device = "cpu"
            val_acc = None
            test_acc = test_f1 = test_auc = class_report = None
        real_pose = getattr(self.person_detector, "provides_keypoints", False)
        return {
            "pipeline": "Keypoint R-CNN → "
            + ("ResNet50+LSTM+Attention (sahne)" if self.clip else "LSTM+Attention")
            if real_pose
            else "R-CNN → OpenPose → "
            + ("ResNet50+LSTM+Attention (sahne)" if self.clip else "LSTM+Attention"),
            "person_detector": self.person_detector.name,
            "pose_estimator": (
                "Keypoint R-CNN — 17 nokta COCO (gerçek)"
                if real_pose else self.pose_estimator.name
            ),
            "behavior_classifier": behavior_name,
            "using_real_model": using_real,
            "model_kind": "scene-clip" if self.clip else "pose-heuristic",
            "device": device,
            "frame_window": window,
            "val_accuracy": round(val_acc, 4) if val_acc is not None else None,
            # Bağımsız TEST kümesi metrikleri (yeni checkpoint'te kayıtlı)
            "test_accuracy": round(test_acc, 4) if test_acc is not None else None,
            "test_f1": round(test_f1, 4) if test_f1 is not None else None,
            "test_auc": round(test_auc, 4) if test_auc is not None else None,
            "class_report": class_report,
            "threat_threshold": settings.threat_threshold,
            "sequence_length": settings.sequence_length,
        }

    # ── Sahne (klip) skorlama — gerçek model ───────────────────────
    @staticmethod
    def _scene_label(score: float) -> str:
        if score >= settings.threat_threshold:
            return "attacker"
        if score >= settings.threat_threshold * 0.6:
            return "suspicious"
        return "normal"

    @staticmethod
    def _scene_action(score: float) -> str:
        if score >= settings.threat_threshold:
            return "şiddet/saldırı"
        if score >= settings.threat_threshold * 0.6:
            return "şüpheli hareket"
        return "normal"

    def score_scene(self, frame: np.ndarray, stream_id: str = "default",
                    stride: int | None = None) -> float | None:
        """Kareyi tampona ekler; periyodik olarak şiddet modelini çalıştırıp
        sahne skorunu (0-1) döndürür. Model yoksa None döner.

        CPU maliyeti yüksek olduğundan her `stride` karede bir gerçek çıkarım
        yapılır; aradaki karelerde son skor tekrar kullanılır.
        """
        if self.clip is None:
            return None
        n = self.clip.num_frames
        # Eğitim dağılımıyla uyum: model ~5 sn'ye eşit yayılmış 20 kare ile
        # eğitildi. Tamponu n*fs kare tutup modele her fs. kareyi veririz;
        # böylece pencere canlıda da ~5 sn'yi kapsar.
        fs = max(1, getattr(settings, "clip_frame_stride", 3))
        buf = self._buffers.setdefault(stream_id, deque(maxlen=n * fs))
        buf.append(frame.copy())
        cnt = self._counters.get(stream_id, 0) + 1
        self._counters[stream_id] = cnt
        stride = stride or getattr(settings, "clip_stride", 10)

        have_window = len(buf) >= n * fs
        fresh_due = (stream_id not in self._last_scene) or (cnt % stride == 0)
        if have_window and fresh_due:
            k = max(2, getattr(settings, "clip_consecutive", 3))
            window = list(buf)[::fs]
            raw = self.clip.infer(window)
            # Hareket kapısı: pencere içi bölgesel hareket düşükse (el ele
            # tutuşma, sohbet, selamlaşma) modelin skoru bastırılır. Gerçek
            # kavga sürekli yüksek hareket ürettiğinden etkilenmez.
            if getattr(settings, "motion_gate", True):
                motion = self._motion_energy(window)
                floor = max(0.1, getattr(settings, "motion_floor", 6.0))
                if motion < floor:
                    raw *= (motion / floor) ** 2
            raws = self._raw_scores.setdefault(stream_id, deque(maxlen=k))
            raws.append(raw)
            # Anlık tepeler (el sallama, kameraya yaklaşma...) yanlış alarm
            # üretmesin: karar skoru son K çıkarımın MİNİMUMU. Alarm ancak
            # K ardışık pencere de eşiği geçerse tetiklenir — gerçek şiddet
            # sürekli olduğundan bu pencereleri kolayca doldurur.
            if len(raws) >= k:
                decision = min(raws)
            else:
                # Yeterli geçmiş yokken tek başına alarm tetiklenemesin
                decision = min(raw, settings.threat_threshold * 0.99)
            self._last_scene[stream_id] = decision
        # Pencere henüz dolmadıysa None (ısınma); sonra son skoru ver.
        return self._last_scene.get(stream_id) if have_window else None

    @staticmethod
    def enhance_frame(frame: np.ndarray) -> np.ndarray:
        """Düşük ışık iyileştirme (form 2.3): kare karanlıksa CLAHE ile
        aydınlatma düzeltmesi + hafif Gauss yumuşatmasıyla gürültü azaltma.

        Normal aydınlıkta kare OLDUĞU GİBİ döner (modelin eğitim dağılımı
        korunur); yalnızca ortalama parlaklık eşiğin altındaysa müdahale edilir.
        """
        if not getattr(settings, "low_light_enhance", True):
            return frame
        gray_mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        if gray_mean >= getattr(settings, "low_light_threshold", 60):
            return frame
        # LAB uzayında yalnızca parlaklık (L) kanalına CLAHE uygula:
        # renkler bozulmadan kontrast/aydınlık dengelenir.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        out = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        # Karanlıkta yükselen sensör gürültüsünü hafif yumuşat
        return cv2.GaussianBlur(out, (3, 3), 0)

    @staticmethod
    def _motion_energy(frames: list[np.ndarray]) -> float:
        """Pencerenin 'sürekli bölgesel hareket' ölçüsü (0-255 ölçeğinde).

        Kareler küçültülüp griye çevrilir; ardışık kare farkları 8x8 hücrede
        ortalanır ve her kare çiftinin EN hareketli hücresi alınır (uzaktaki/
        küçük bir kavga da kendi hücresini aydınlatır). Pencere genelinde
        MEDYAN kullanılır: tek anlık hamle (el uzatma) medyanı yükseltemez,
        sürekli hareket (kavga) yükseltir.
        """
        if len(frames) < 2:
            return 0.0
        grays = [
            cv2.cvtColor(cv2.resize(f, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            for f in frames
        ]
        peaks = []
        for a, b in zip(grays, grays[1:]):
            diff = np.abs(b - a)
            # 8x8 hücre ortalamaları (64/8=8 piksellik bloklar)
            cells = diff.reshape(8, 8, 8, 8).mean(axis=(1, 3))
            peaks.append(float(cells.max()))
        return float(np.median(peaks))

    def reset_stream(self, stream_id: str) -> None:
        """Bir akış kapandığında tamponunu temizle."""
        self._buffers.pop(stream_id, None)
        self._counters.pop(stream_id, None)
        self._last_scene.pop(stream_id, None)
        self._raw_scores.pop(stream_id, None)
        self._last_persons.pop(stream_id, None)
        self._det_counters.pop(stream_id, None)

    def _detect_persons(self, frame: np.ndarray,
                        stream_id: str | None) -> list[PersonDetection]:
        """Kişi tespiti — ağır dedektörü `detect_stride` karede bir çalıştır,
        aradaki karelerde son tespitleri yeniden kullan."""
        ds = max(1, getattr(settings, "detect_stride", 2))
        heavy = getattr(self.person_detector, "provides_keypoints", False)
        if stream_id is None or ds == 1 or not heavy:
            return self.person_detector.detect(frame)
        cnt = self._det_counters.get(stream_id, 0) + 1
        self._det_counters[stream_id] = cnt
        if stream_id not in self._last_persons or cnt % ds == 1:
            self._last_persons[stream_id] = self.person_detector.detect(frame)
        return self._last_persons[stream_id]

    # ── Ana işlem ──────────────────────────────────────────────────
    def process(
        self,
        frame: np.ndarray,
        persons: list[PersonDetection] | None = None,
        scene_score: float | None = None,
        warming: bool = False,
        stream_id: str | None = None,
    ) -> FrameResult:
        """Bir kareyi işler.

        `scene_score` verilirse (gerçek şiddet modeli aktif) skor sahne
        düzeyinde kabul edilir ve tespit edilen tüm kişilere yansıtılır.
        Verilmezse kişi-bazlı sezgisel/LSTM sınıflandırıcı çalışır.
        `warming=True` ise gerçek model kare tamponu henüz dolmamıştır:
        skor 0 kabul edilir, alarm üretilmez (sezgisele DÜŞÜLMEZ).
        """
        self._frame_index += 1
        result = FrameResult(frame_index=self._frame_index, warming=warming)
        if warming and scene_score is None:
            scene_score = 0.0
        real = scene_score is not None

        # 1) Kişi tespiti (R-CNN). `persons` dışarıdan verilirse (ör. demo
        #    modu) tespit aşaması atlanır.
        if persons is None:
            persons = self._detect_persons(frame, stream_id)

        # 2) Takip kimliği ata (zaman serisi için)
        self.tracker.update(persons)

        scene_label = self._scene_label(scene_score) if real else None
        scene_action = self._scene_action(scene_score) if real else None

        # Kişiler-arası şiddet en az 2 kişi gerektirir: dedektör güvenilirken
        # sahnede tek kişi varsa (kameraya el sallama, selamlama, tek başına
        # hızlı hareket) 'saldırgan' etiketini 'şüpheli'ye düşür.
        detector_reliable = getattr(self.person_detector, "provides_keypoints", False)
        if (real and scene_label == "attacker" and detector_reliable
                and len(persons) < getattr(settings, "min_persons_for_alarm", 2)):
            scene_score = min(scene_score, settings.threat_threshold * 0.99)
            scene_label = "suspicious"
            scene_action = "şüpheli hareket (tek kişi)"

        for person in persons:
            # 3) Poz/iskelet: dedektör GERÇEK keypoint verdiyse onu koru;
            #    vermediyse (HOG/demo) yedek tahminciyle doldur.
            if not person.keypoints:
                person.keypoints = self.pose_estimator.estimate(frame, person)

            # 4) Keypoint -> özellik vektörü, kişi geçmişine ekle
            feats = self._features(person, frame.shape)
            sequence = self.tracker.push_features(person.track_id, feats)

            if real:
                # 5a) Gerçek model: sahne skorunu kişiye yansıt
                person.threat_score = scene_score
                person.action = scene_action
                person.label = scene_label
            else:
                # 5b) Sezgisel/LSTM kişi-bazlı sınıflandırma
                behavior = self.classifier.classify(sequence)
                person.threat_score = behavior.threat_score
                person.action = behavior.action
                person.label = behavior.label

            result.max_threat = max(result.max_threat, person.threat_score)
            if person.label == "attacker":
                result.has_attacker = True

        # Gerçek model sahneyi şüpheli/saldırgan buldu ama hiç kişi
        # tespit edilemediyse, olayın kaydı/uyarısı düşmesin diye tam-kare
        # temsili bir tespit üret. İSTİSNA: kişi dedektörü gerçekse
        # (Keypoint R-CNN) ve sahnede insan yoksa kavga da olamaz —
        # temsili tespit üretmek kesin yanlış alarm olur, üretme.
        block_no_person = detector_reliable and getattr(
            settings, "require_person_for_alarm", True
        )
        if (real and not persons and not block_no_person
                and scene_score >= settings.threat_threshold * 0.6):
            h, w = frame.shape[:2]
            persons = [PersonDetection(
                bbox=BBox(2, 2, w - 4, h - 4), score=1.0, track_id=0,
                threat_score=scene_score, action=scene_action, label=scene_label,
            )]
            result.max_threat = scene_score
            if scene_label == "attacker":
                result.has_attacker = True

        result.persons = persons
        if real:
            result.scene_threat = scene_score
            result.model_name = self.clip.name if self.clip else ""
        return result

    def annotate(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Tespit sonuçlarını kare üzerine çiz (kutu + iskelet + etiket)."""
        out = frame.copy()

        # ── Sahne durumu çubuğu (gerçek model aktifken sürekli görünür) ──
        # Kullanıcı sistemin "canlı" olduğunu her an görsün diye skor,
        # tespit olmasa bile ekranın üstünde gösterilir.
        if result.model_name or result.warming:
            score = max(0.0, min(1.0, result.scene_threat))
            if result.warming:
                text, color = "MODEL ISINIYOR...", (160, 160, 160)
            else:
                lbl = self._scene_label(score)
                names = {"normal": "NORMAL", "suspicious": "SUPHELI",
                         "attacker": "SALDIRGAN"}
                text = f"SAHNE: {names[lbl]} %{score * 100:.0f}"
                color = _COLORS[lbl]
            bar_w, bar_h, x0, y0 = 230, 26, 10, 10
            cv2.rectangle(out, (x0, y0), (x0 + bar_w, y0 + bar_h), (25, 25, 25), -1)
            fill = int(bar_w * score)
            if fill > 0 and not result.warming:
                cv2.rectangle(out, (x0, y0), (x0 + fill, y0 + bar_h), color, -1)
            cv2.rectangle(out, (x0, y0), (x0 + bar_w, y0 + bar_h), (210, 210, 210), 1)
            cv2.putText(out, text, (x0 + 6, y0 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for p in result.persons:
            color = _COLORS.get(p.label, (200, 200, 200))
            b = p.bbox
            cv2.rectangle(out, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 2)

            # İskelet çizimi — yalnızca güvenle GÖRÜNÜR eklemler çizilir;
            # düşük güvenli noktalar (perde arkası eklem vb.) atlanır ki
            # iskelet "abuk subuk" görünmesin.
            kps = p.keypoints
            if kps:
                vis = 0.5  # görünürlük eşiği
                for i, j in SKELETON_EDGES:
                    if (i < len(kps) and j < len(kps)
                            and kps[i].confidence >= vis
                            and kps[j].confidence >= vis):
                        cv2.line(out, (int(kps[i].x), int(kps[i].y)),
                                 (int(kps[j].x), int(kps[j].y)), color, 2)
                for k in kps:
                    if k.confidence >= vis:
                        cv2.circle(out, (int(k.x), int(k.y)), 3, color, -1)

            tag = f"#{p.track_id} {p.label} {p.threat_score:.0%}"
            cv2.rectangle(out, (b.x, b.y - 18), (b.x + 8 + 7 * len(tag), b.y), color, -1)
            cv2.putText(out, tag, (b.x + 4, b.y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Zaman damgası (form 2.2.3) — sağ alt köşe, kayıt/analiz kanıtı ──
        ts = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        h, w = out.shape[:2]
        (tw, th), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (w - tw - 14, h - th - 14), (w - 4, h - 4), (25, 25, 25), -1)
        cv2.putText(out, ts, (w - tw - 9, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    # ── Yardımcılar ────────────────────────────────────────────────
    @staticmethod
    def _features(person: PersonDetection, shape: tuple) -> list[float]:
        """Keypoint'leri kare boyutuna göre normalize edilmiş düz vektöre çevir."""
        h, w = shape[:2]
        if not person.keypoints:
            # Keypoint yoksa bbox merkezini özellik olarak kullan
            cx, cy = person.bbox.center
            return [cx / w, cy / h]
        feats: list[float] = []
        for k in person.keypoints:
            feats.extend([k.x / max(w, 1), k.y / max(h, 1)])
        return feats


# Tüm uygulamanın paylaştığı tekil pipeline örneği
_pipeline: DetectionPipeline | None = None


def get_pipeline() -> DetectionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DetectionPipeline()
    return _pipeline
