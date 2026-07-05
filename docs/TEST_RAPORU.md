# Test-Doğrulama Raporu (İş Paketi 4)

**Başarı ölçütü (başvuru formu):** En az 50 farklı doğrulama senaryosu için
test çalışması ve en az %95 doğruluk.

- **Senaryo sayısı:** 60 (30 şiddet + 30 normal — yerel RLVS test
  bölünmesinden, tohum=42 ile tekrarlanabilir örnekleme)
- **Test verisi:** "Real Life Violence Situations" yerel test bölünmesi.
  Not: Bu bölünme yerel olarak ayrılmıştır; Kaggle eğitimindeki bölünmeyle
  birebir örtüşmeyebilir. Kesin bağımsız ölçüm, modelin kendi ayrık test
  kümesindeki sonuçtur (%98.0, 300 video). Bu çalışmanın amacı İP4'teki
  50+ SENARYO doğrulamasıdır ve sistemin uçtan uca davranışını da kapsar.
- **Model:** ResNet50 + BiLSTM + Attention v2 (bağımsız test kümesinde
  %98.0 doğruluk, F1 0.980, AUC 0.994 — 300 video)
- **Donanım:** NVIDIA RTX 4060, ~41 ms/klip

## 1. Model Düzeyi Sonuçlar (eşik 0.5)

| Metrik | Değer |
|---|---|
| **Doğruluk (Accuracy)** | **%100.0** |
| Kesinlik (Precision) | %100.0 |
| Duyarlılık (Recall) | %100.0 |
| F1 | %100.0 |

**Karışıklık matrisi:** TP=30 · TN=30 · FP=0 · FN=0

Formdaki **≥%95 doğruluk** ölçütü: ✅ SAĞLANDI

## 2. Sistem Düzeyi (uçtan uca, tüm yanlış-alarm filtreleriyle)

Videolar `/api/analyze` üzerinden, canlı sistemdeki TÜM koruma katmanlarıyla
(hareket kapısı, min-2-kişi kuralı, %80 alarm eşiği) işlenmiştir. Bu katmanlar
kasıtlı olarak alarm üretmeyi zorlaştırır (yanlış alarm maliyeti gerçek
sahada yüksektir):

| Metrik | Değer |
|---|---|
| Doğruluk | %85.0 |
| Kesinlik (alarm güvenilirliği) | %100.0 |
| Duyarlılık (yakalama oranı) | %70.0 |

TP=21 · TN=30 · FP=0 · FN=9

> Yorum: Sistem düzeyinde kesinlik önceliklidir — normal videolarda alarm
> (0/30) minimumda tutulurken, sürekli/çok kişili gerçek
> şiddet yakalanır. Model düzeyi doğruluk akademik ölçüt, sistem düzeyi ise
> operasyonel davranıştır.

## 3. Senaryo Detayları

| Video | Gerçek | Model Skoru | Model Kararı | Sistem Kararı | Sistem Skoru |
|---|---|---|---|---|---|
| V_966.mp4 | Şiddet | 0.955 | Şiddet | ALARM | 0.953 |
| V_844.mp4 | Şiddet | 0.948 | Şiddet | ALARM | 0.947 |
| V_824.mp4 | Şiddet | 0.940 | Şiddet | ALARM | 0.938 |
| V_99.mp4 | Şiddet | 0.938 | Şiddet | ALARM | 0.939 |
| V_882.mp4 | Şiddet | 0.942 | Şiddet | Temiz | 0.792 |
| V_875.mp4 | Şiddet | 0.952 | Şiddet | ALARM | 0.951 |
| V_870.mp4 | Şiddet | 0.939 | Şiddet | ALARM | 0.946 |
| V_850.mp4 | Şiddet | 0.941 | Şiddet | ALARM | 0.946 |
| V_989.mp4 | Şiddet | 0.899 | Şiddet | Temiz | 0.792 |
| V_842.mp4 | Şiddet | 0.944 | Şiddet | Temiz | 0.000 |
| V_975.mp4 | Şiddet | 0.939 | Şiddet | ALARM | 0.942 |
| V_944.mp4 | Şiddet | 0.951 | Şiddet | Temiz | 0.792 |
| V_839.mp4 | Şiddet | 0.903 | Şiddet | ALARM | 0.933 |
| V_955.mp4 | Şiddet | 0.945 | Şiddet | Temiz | 0.792 |
| V_916.mp4 | Şiddet | 0.954 | Şiddet | ALARM | 0.953 |
| V_826.mp4 | Şiddet | 0.953 | Şiddet | ALARM | 0.952 |
| V_825.mp4 | Şiddet | 0.952 | Şiddet | Temiz | 0.792 |
| V_84.mp4 | Şiddet | 0.959 | Şiddet | ALARM | 0.961 |
| V_869.mp4 | Şiddet | 0.956 | Şiddet | Temiz | 0.000 |
| V_872.mp4 | Şiddet | 0.938 | Şiddet | ALARM | 0.931 |
| V_935.mp4 | Şiddet | 0.947 | Şiddet | ALARM | 0.952 |
| V_958.mp4 | Şiddet | 0.947 | Şiddet | ALARM | 0.943 |
| V_997.mp4 | Şiddet | 0.953 | Şiddet | Temiz | 0.792 |
| V_948.mp4 | Şiddet | 0.946 | Şiddet | Temiz | 0.792 |
| V_864.mp4 | Şiddet | 0.941 | Şiddet | ALARM | 0.941 |
| V_969.mp4 | Şiddet | 0.951 | Şiddet | ALARM | 0.954 |
| V_991.mp4 | Şiddet | 0.944 | Şiddet | ALARM | 0.945 |
| V_915.mp4 | Şiddet | 0.953 | Şiddet | ALARM | 0.942 |
| V_87.mp4 | Şiddet | 0.961 | Şiddet | ALARM | 0.960 |
| V_921.mp4 | Şiddet | 0.955 | Şiddet | ALARM | 0.952 |
| NV_962.mp4 | Normal | 0.068 | Normal | Temiz | 0.000 |
| NV_848.mp4 | Normal | 0.069 | Normal | Temiz | 0.069 |
| NV_785.mp4 | Normal | 0.057 | Normal | Temiz | 0.059 |
| NV_82.mp4 | Normal | 0.066 | Normal | Temiz | 0.000 |
| NV_988.mp4 | Normal | 0.064 | Normal | Temiz | 0.061 |
| NV_924.mp4 | Normal | 0.066 | Normal | Temiz | 0.065 |
| NV_862.mp4 | Normal | 0.065 | Normal | Temiz | 0.069 |
| NV_998.mp4 | Normal | 0.068 | Normal | Temiz | 0.068 |
| NV_819.mp4 | Normal | 0.058 | Normal | Temiz | 0.057 |
| NV_833.mp4 | Normal | 0.062 | Normal | Temiz | 0.057 |
| NV_861.mp4 | Normal | 0.064 | Normal | Temiz | 0.066 |
| NV_807.mp4 | Normal | 0.072 | Normal | Temiz | 0.071 |
| NV_804.mp4 | Normal | 0.060 | Normal | Temiz | 0.061 |
| NV_914.mp4 | Normal | 0.059 | Normal | Temiz | 0.063 |
| NV_805.mp4 | Normal | 0.065 | Normal | Temiz | 0.063 |
| NV_9.mp4 | Normal | 0.063 | Normal | Temiz | 0.056 |
| NV_87.mp4 | Normal | 0.056 | Normal | Temiz | 0.055 |
| NV_966.mp4 | Normal | 0.062 | Normal | Temiz | 0.060 |
| NV_844.mp4 | Normal | 0.092 | Normal | Temiz | 0.000 |
| NV_794.mp4 | Normal | 0.058 | Normal | Temiz | 0.058 |
| NV_932.mp4 | Normal | 0.054 | Normal | Temiz | 0.054 |
| NV_950.mp4 | Normal | 0.059 | Normal | Temiz | 0.059 |
| NV_811.mp4 | Normal | 0.068 | Normal | Temiz | 0.063 |
| NV_913.mp4 | Normal | 0.058 | Normal | Temiz | 0.000 |
| NV_801.mp4 | Normal | 0.061 | Normal | Temiz | 0.062 |
| NV_954.mp4 | Normal | 0.052 | Normal | Temiz | 0.058 |
| NV_851.mp4 | Normal | 0.069 | Normal | Temiz | 0.071 |
| NV_971.mp4 | Normal | 0.053 | Normal | Temiz | 0.056 |
| NV_97.mp4 | Normal | 0.058 | Normal | Temiz | 0.058 |
| NV_90.mp4 | Normal | 0.060 | Normal | Temiz | 0.060 |

---
*Bu rapor `scenario_test.py` ile otomatik üretilmiştir
(05.07.2026 22:18). Aynı tohumla (42) birebir tekrarlanabilir.*
