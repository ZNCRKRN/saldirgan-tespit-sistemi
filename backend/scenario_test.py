"""İş Paketi 4 — Test/Doğrulama Senaryo Koşucusu.

Başvuru formundaki İP4 ölçütü: "En az 50 farklı doğrulama senaryosu için
test çalışması ve en az %95 doğruluk". Bu script, eğitilmiş şiddet modelini
(ResNet50+BiLSTM+Attention) yerel RLVS test videoları üzerinde çalıştırıp
`docs/TEST_RAPORU.md` dosyasını OTOMATİK üretir. Sabit tohum (42) ile
örnekleme yaptığından sonuç birebir tekrarlanabilir — jüri aynı komutla
aynı raporu üretebilir.

İki düzeyde ölçüm yapılır:
  1. MODEL düzeyi   — ham çıkarım skoru, karar eşiği 0.5 (akademik ölçüt).
  2. SİSTEM düzeyi  — canlı akıştaki TÜM yanlış-alarm filtreleriyle
                      (hareket kapısı, min-2-kişi, %80 alarm eşiği). Bu,
                      `/api/analyze` endpoint'iyle AYNI mantıktır.

Kullanım
--------
    cd backend
    python scenario_test.py --data "C:/veri/RLVS/test" --per-class 30

`--data` klasörü şu yapıda beklenir (RLVS'in standart bölünmesi):
    <data>/Violence/*.mp4        (şiddet videoları)
    <data>/NonViolence/*.mp4     (normal videolar)

Alt klasör adları esnektir: "Violence/Fight/violent" ve
"NonViolence/NonFight/normal" varyasyonları otomatik tanınır. Model dosyası
`backend/models/` içinde aranır (bkz. violence_classifier.build_violence_classifier).
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Uygulama paketini içe aktarabilmek için backend/ dizinini yola ekle
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.ml.pipeline import get_pipeline  # noqa: E402

SEED = 42
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
# Alt klasör adı eşleştirme (küçük harfe indirgenir)
VIOLENCE_DIRS = {"violence", "fight", "fights", "violent", "1"}
NORMAL_DIRS = {"nonviolence", "non-violence", "nonfight", "normal", "0"}


# ── Veri keşfi ─────────────────────────────────────────────────────────
def _collect(data_dir: Path, names: set[str]) -> list[Path]:
    """`data_dir` altında adı `names` kümesindeki bir alt klasörde bulunan
    tüm videoları toplar (bir düzey iç içe de taranır)."""
    found: list[Path] = []
    for sub in data_dir.iterdir():
        if sub.is_dir() and sub.name.lower() in names:
            for f in sub.rglob("*"):
                if f.suffix.lower() in VIDEO_EXTS:
                    found.append(f)
    return found


def _sample(paths: list[Path], k: int, rng: random.Random) -> list[Path]:
    if len(paths) <= k:
        return sorted(paths)
    return sorted(rng.sample(paths, k))


# ── Tek videoyu skorla (canlı /api/analyze mantığıyla birebir) ─────────
def _score_video(pipeline, path: Path) -> tuple[float, float, bool]:
    """Videoyu pencerelere böler.

    Döndürür: (model_skoru, sistem_skoru, sistem_alarmı)
      - model_skoru : pencerelerin ham çıkarım skorlarının MAKSİMUMU
      - sistem_skoru: filtre-sonrası karar skoru (hareket kapısı vb.)
      - sistem_alarmı: herhangi bir pencere 'attacker' eşiğini geçti mi
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return 0.0, 0.0, False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / 4))  # ~5 sn'ye yayılmış 20 kare (eğitim dağılımı)
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(pipeline.enhance_frame(frame))
        idx += 1
    cap.release()

    if not frames:
        return 0.0, 0.0, False

    n = pipeline.clip.num_frames
    if len(frames) <= n:
        windows = [frames]
    else:
        num_win = min(12, max(1, len(frames) // n))
        starts = np.linspace(0, len(frames) - n, num_win, dtype=int)
        windows = [frames[s:s + n] for s in starts]

    raw_max = 0.0
    sys_max = 0.0
    attacker = False
    for w in windows:
        raw = pipeline.clip.infer(w)
        raw_max = max(raw_max, raw)
        score = raw
        # Canlı akışla aynı hareket-enerjisi kapısı
        if settings.motion_gate:
            m = pipeline._motion_energy(w)
            floor = max(0.1, settings.motion_floor)
            if m < floor:
                score *= (m / floor) ** 2
        mid = w[len(w) // 2]
        result = pipeline.process(mid, scene_score=score)
        sys_max = max(sys_max, result.max_threat)
        if result.has_attacker:
            attacker = True
    return raw_max, sys_max, attacker


# ── Metrik hesabı ──────────────────────────────────────────────────────
def _metrics(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    total = len(y_true) or 1
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "acc": acc, "prec": prec, "rec": rec, "f1": f1}


# ── Rapor üretimi (docs/TEST_RAPORU.md ile aynı biçim) ─────────────────
def _write_report(rows: list[dict], model_m: dict, sys_m: dict,
                  status: dict, out_path: Path, n_each: int) -> None:
    total = len(rows)
    thr_model = 0.5
    device = "GPU (CUDA)" if status.get("device") == "cuda" else "CPU"
    test_acc = status.get("test_accuracy")
    test_f1 = status.get("test_f1")
    test_auc = status.get("test_auc")
    model_line = (
        f"ResNet50 + BiLSTM + Attention "
        + (f"(bağımsız test kümesinde %{test_acc * 100:.1f} doğruluk"
           f"{f', F1 {test_f1:.3f}' if test_f1 else ''}"
           f"{f', AUC {test_auc:.3f}' if test_auc else ''})"
           if test_acc is not None else "(checkpoint metriği yok)")
    )

    def rowline(r: dict) -> str:
        return (f"| {r['name']} | {r['truth']} | {r['model_score']:.3f} | "
                f"{r['model_pred']} | {r['sys_pred']} | {r['sys_score']:.3f} |")

    scenario_rows = "\n".join(rowline(r) for r in rows)
    now = datetime.now()

    md = f"""# Test-Doğrulama Raporu (İş Paketi 4)

**Başarı ölçütü (başvuru formu):** En az 50 farklı doğrulama senaryosu için
test çalışması ve en az %95 doğruluk.

- **Senaryo sayısı:** {total} ({n_each} şiddet + {n_each} normal — yerel RLVS test
  bölünmesinden, tohum={SEED} ile tekrarlanabilir örnekleme)
- **Test verisi:** "Real Life Violence Situations" yerel test bölünmesi.
- **Model:** {model_line}
- **Donanım:** {device}
- **Karar eşiği:** model düzeyi {thr_model}, sistem düzeyi
  %{round(settings.threat_threshold * 100)} (canlı alarm eşiği)

## 1. Model Düzeyi Sonuçlar (eşik {thr_model})

| Metrik | Değer |
|---|---|
| **Doğruluk (Accuracy)** | **%{model_m['acc'] * 100:.1f}** |
| Kesinlik (Precision) | %{model_m['prec'] * 100:.1f} |
| Duyarlılık (Recall) | %{model_m['rec'] * 100:.1f} |
| F1 | %{model_m['f1'] * 100:.1f} |

**Karışıklık matrisi:** TP={model_m['tp']} · TN={model_m['tn']} · \
FP={model_m['fp']} · FN={model_m['fn']}

Formdaki **≥%95 doğruluk** ölçütü: \
{'✅ SAĞLANDI' if model_m['acc'] >= 0.95 else '⚠️ SAĞLANMADI'}

## 2. Sistem Düzeyi (uçtan uca, tüm yanlış-alarm filtreleriyle)

Videolar canlı sistemdeki TÜM koruma katmanlarıyla (hareket kapısı,
min-2-kişi kuralı, %{round(settings.threat_threshold * 100)} alarm eşiği)
işlenmiştir. Bu katmanlar kasıtlı olarak alarm üretmeyi zorlaştırır (yanlış
alarm maliyeti gerçek sahada yüksektir):

| Metrik | Değer |
|---|---|
| Doğruluk | %{sys_m['acc'] * 100:.1f} |
| Kesinlik (alarm güvenilirliği) | %{sys_m['prec'] * 100:.1f} |
| Duyarlılık (yakalama oranı) | %{sys_m['rec'] * 100:.1f} |

TP={sys_m['tp']} · TN={sys_m['tn']} · FP={sys_m['fp']} · FN={sys_m['fn']}

> Yorum: Sistem düzeyinde kesinlik önceliklidir — normal videolarda yanlış
> alarm minimumda tutulurken sürekli/çok kişili gerçek şiddet yakalanır.
> Model düzeyi doğruluk akademik ölçüt, sistem düzeyi ise operasyonel
> davranıştır.

## 3. Senaryo Detayları

| Video | Gerçek | Model Skoru | Model Kararı | Sistem Kararı | Sistem Skoru |
|---|---|---|---|---|---|
{scenario_rows}

## 4. Tekrar Üretme

Aynı sonuçlar tek komutla üretilir (tohum sabit: {SEED}):

```bash
cd backend
python scenario_test.py --data "<RLVS_test_klasoru>" --per-class {n_each}
```

`<RLVS_test_klasoru>` içinde `Violence/` ve `NonViolence/` alt klasörleri
bulunmalıdır (Kaggle "Real Life Violence Situations" standart yapısı).

---
*Bu rapor `scenario_test.py` ile otomatik üretilmiştir
({now.strftime('%d.%m.%Y %H:%M')}). Aynı tohumla ({SEED}) birebir tekrarlanabilir.*
"""
    out_path.write_text(md, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="İP4 senaryo test koşucusu")
    ap.add_argument("--data", required=True,
                    help="RLVS test klasörü (Violence/ ve NonViolence/ içerir)")
    ap.add_argument("--per-class", type=int, default=30,
                    help="Sınıf başına senaryo sayısı (varsayılan 30 → toplam 60)")
    ap.add_argument("--out", default=None,
                    help="Rapor çıktısı (varsayılan: ../docs/TEST_RAPORU.md)")
    args = ap.parse_args()

    data_dir = Path(args.data).expanduser().resolve()
    if not data_dir.is_dir():
        print(f"HATA: veri klasörü bulunamadı: {data_dir}")
        return 2

    pipeline = get_pipeline()
    if pipeline.clip is None:
        print("HATA: eğitilmiş şiddet modeli yüklenemedi. Model dosyasını "
              "backend/models/ içine koyun (bkz. models/README.md).")
        return 3

    v_paths = _collect(data_dir, VIOLENCE_DIRS)
    n_paths = _collect(data_dir, NORMAL_DIRS)
    if not v_paths or not n_paths:
        print("HATA: Violence/ ve NonViolence/ alt klasörlerinde video "
              f"bulunamadı. (şiddet={len(v_paths)}, normal={len(n_paths)})")
        return 4

    rng = random.Random(SEED)
    k = args.per_class
    v_sel = _sample(v_paths, k, rng)
    n_sel = _sample(n_paths, k, rng)
    n_each = min(len(v_sel), len(n_sel))
    v_sel, n_sel = v_sel[:n_each], n_sel[:n_each]

    print(f"[İP4] {2 * n_each} senaryo işleniyor "
          f"({n_each} şiddet + {n_each} normal)...")

    rows: list[dict] = []
    y_true, model_pred, sys_pred = [], [], []
    thr_model = 0.5
    for truth, paths in ((1, v_sel), (0, n_sel)):
        for i, p in enumerate(paths, 1):
            raw, sys_score, attacker = _score_video(pipeline, p)
            m_pred = 1 if raw >= thr_model else 0
            s_pred = 1 if attacker else 0
            y_true.append(truth)
            model_pred.append(m_pred)
            sys_pred.append(s_pred)
            rows.append({
                "name": p.name,
                "truth": "Şiddet" if truth == 1 else "Normal",
                "model_score": raw,
                "model_pred": "Şiddet" if m_pred else "Normal",
                "sys_pred": "ALARM" if s_pred else "Temiz",
                "sys_score": sys_score,
            })
            tag = "Şiddet" if truth == 1 else "Normal"
            print(f"  [{tag} {i}/{len(paths)}] {p.name}: "
                  f"model={raw:.3f} sistem={sys_score:.3f} "
                  f"{'ALARM' if attacker else 'temiz'}")

    model_m = _metrics(y_true, model_pred)
    sys_m = _metrics(y_true, sys_pred)

    out_path = (Path(args.out).resolve() if args.out
                else BACKEND_DIR.parent / "docs" / "TEST_RAPORU.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(rows, model_m, sys_m, pipeline.status(), out_path, n_each)

    print("\n" + "=" * 60)
    print(f"  MODEL  düzeyi doğruluk : %{model_m['acc'] * 100:.1f} "
          f"(≥%95 {'✅' if model_m['acc'] >= 0.95 else '⚠️'})")
    print(f"  SİSTEM düzeyi doğruluk : %{sys_m['acc'] * 100:.1f}")
    print(f"  Rapor yazıldı          : {out_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
