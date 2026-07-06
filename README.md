# Kapalı Alanlarda Derin Öğrenme Tabanlı Saldırgan Tespiti

TÜBİTAK 2209-A araştırma projesi: güvenlik kamerası görüntülerinden
**derin öğrenme ile gerçek zamanlı saldırgan/şiddet tespiti** yapan
uçtan uca sistem.

- **Şiddet tespiti:** ResNet50 + BiLSTM + Attention (PyTorch) — [RWF-2000](https://www.kaggle.com/datasets/vulamnguyen/rwf2000), [Real Life Violence Situations](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) ve [CCTV Aggressive Poses / Fight Detection](https://www.kaggle.com/code/ocwerfrancis/cctv-aggressive-poses-fight-detection) veri setleri birleştirilerek toplam **4000+ video ile eğitildi**, doğrulama doğruluğu **%97.7**
- **Kişi tespiti + iskelet:** Keypoint R-CNN (torchvision) — 17 noktalı
  COCO insan iskeleti
- **Arayüz:** canlı izleme, uyarılar, olay geçmişi, raporlar, video analizi

---

## 🚀 Hızlı Başlangıç (önerilen — tek tık)

**Gereksinim:** [Python 3.10–3.12](https://www.python.org/downloads/)
(kurulumda *"Add Python to PATH"* işaretli olmalı). Başka hiçbir şey
kurmanıza gerek yok — arayüz derlenmiş halde pakete dahildir.

1. Projeyi indirin:
   **Code → Download ZIP** (veya `git clone https://github.com/ZNCRKRN/saldirgan-tespit-sistemi.git`)
   ve bir klasöre çıkarın.
2. **`baslat.bat`** dosyasına çift tıklayın.
   - İlk çalıştırmada paketler kurulur ve eğitilmiş model (~233 MB)
     otomatik indirilir — internet hızına göre 5–10 dk sürer, **tek seferlik**.
   - Sonraki açılışlar ~30 saniyedir.
3. Tarayıcı otomatik açılır: **http://localhost:8000**

> 💡 **GPU'nuz varsa (NVIDIA):** varsayılan kurulum CPU'da çalışır (tam
> işlevsel, canlı akış daha yavaş skorlanır). ~19 kat hız için kurulum
> bittikten sonra bir kez şunu çalıştırın:
> ```
> .venv\Scripts\pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
> ```

### İlk kullanım — 2 dakikada deneme

1. Açılışta **"Canlı Webcam"** kamerası hazırdır → **Canlı İzleme**
   sayfasında bilgisayarınızın kamerası açılır. İlk ~5 sn "MODEL ISINIYOR…"
   yazar, sonra sol üstte canlı sahne skoru (`SAHNE: NORMAL %2` gibi) akar.
2. **Video Analizi** sayfasından herhangi bir video (mp4/avi) yükleyin —
   sistem pencere pencere analiz edip şiddet içerip içermediğini raporlar.
3. **Kameralar** sayfasından RTSP/IP kamera veya video dosyası
   ekleyebilirsiniz (sayfadaki "Nasıl bağlanır?" rehberine bakın).

---

## Elle Kurulum (geliştirici)

```bash
# Backend
cd backend
python -m venv .venv
.venv\Scripts\activate              # Windows
pip install -r requirements.txt
# Model: https://github.com/ZNCRKRN/saldirgan-tespit-sistemi/releases
# adresinden best_model.pth indirin → backend/models/ içine koyun
python run.py                        # http://localhost:8000 (arayüz + API)

# Frontend'i değiştirecekseniz (opsiyonel — hazır derlenmişi pakette var)
cd frontend
npm install
npm run dev                          # http://localhost:5173 (canlı geliştirme)
npm run build                        # dist/ güncellenir
```

---

## Mimari

```
Kare akışı ──> Keypoint R-CNN ──────────> kişi kutuları + 17-nokta iskelet
        │                                          │
        └──> 20 karelik pencere (~5 sn)            ▼
             ResNet50+BiLSTM+Attention ──> sahne şiddet skoru (0-1)
                                                   │
             yanlış-alarm filtreleri  <────────────┘
             (ısınma, 4-pencere doğrulama,
              hareket-enerjisi kapısı, kişi şartı)
                                                   │
                                                   ▼
             etiketleme ──> uyarı + veritabanı + snapshot ──> arayüz (WS)
```

```
├── backend/                 FastAPI + ML pipeline (Python)
│   ├── app/
│   │   ├── main.py          Uygulama + arayüz sunumu (SPA) + CORS
│   │   ├── config.py        Eşikler ve tüm ayarlar (açıklamalı)
│   │   ├── database.py      SQLite + başlangıç kaydı
│   │   ├── routers/         cameras, events, reports, stream(ws)
│   │   └── ml/
│   │       ├── pipeline.py            Orkestratör + filtreler + çizim
│   │       ├── violence_classifier.py Eğitilmiş şiddet modeli sarmalayıcı
│   │       ├── person_detector.py     Keypoint R-CNN (yedek: HOG)
│   │       └── tracker.py             Kişi takibi
│   └── models/best_model.pth          ← eğitilmiş ağırlıklar (Releases'ten)
└── frontend/                React + Vite + Tailwind + Recharts
    └── dist/                Derlenmiş arayüz (pakete dahil)
```

---

## Çok Açılı Füzyon

Aynı alanı gören kameralara **Kameralar** sayfasından aynı "Bölge" adı
verilir (ör. `giris-holu`). Bölgedeki kameraların **en yüksek güncel sahne
skoru** tüm açılara yansıtılır: saldırı bir açıdan görünmese bile diğer
açıdan yakalandığında bölgedeki tüm kameralar alarma geçer — başvuru
formundaki "çok açılı video analizi" hedefinin gerçeklenmesidir.

## Veri Güvenliği

- Tespit snapshot'ları (kişisel veri içeren görüntüler) diskte
  **AES-128 (Fernet) ile şifreli** saklanır (`.enc`); arayüze sunulurken
  bellekte çözülür. Anahtar: `backend/storage/.snapshot_key`
  (ilk açılışta üretilir, repo'ya girmez).
- Veritabanında kişisel veri tutulmaz; yalnızca olay meta verileri
  (zaman, etiket, skor) saklanır.
- Tüm görüntülere tarih/saat damgası basılır (kayıt bütünlüğü).

## Test-Doğrulama (İş Paketi 4)

60 senaryoluk (30 şiddet + 30 normal) tekrarlanabilir doğrulama çalışmasının
sonuçları: **[docs/TEST_RAPORU.md](docs/TEST_RAPORU.md)**. Arayüzdeki
Raporlar sayfasından da güncel sistem raporu HTML olarak indirilebilir.

## Yanlış Alarm Önlemleri

Gerçek sahalarda en kritik sorun yanlış alarmdır; sistem 4 katmanlı filtre uygular:

| Katman | Ne yapar |
|---|---|
| Isınma | İlk ~5 sn tampon dolarken alarm üretilmez |
| Zaman örnekleme | Modele eğitimdeki gibi ~5 sn'ye yayılmış 20 kare verilir |
| Ardışık doğrulama | Alarm için son **4 çıkarım penceresinin tamamı** eşiği (%80) geçmeli (~3 sn sürekli şiddet) |
| Hareket kapısı | Pencere içi bölgesel hareket düşükse (el sıkışma, sohbet, el ele tutuşma) skor bastırılır |
| Kişi şartı | Keypoint R-CNN sahnede insan bulamazsa şiddet alarmı üretilmez |

Eşikler `backend/app/config.py` içinde açıklamalarıyla birlikte ayarlanabilir.

---

## Özellikler

- **Canlı izleme** — WebSocket ile gerçek zamanlı kare + tespit (kutu, iskelet, skor çubuğu)
- **Uyarı sistemi** — saldırgan tespitinde anlık uyarı + operatör onayı
- **Veri kaydı** — olay/uyarı veritabanı (SQLite) + otomatik snapshot arşivi
- **Raporlama** — zaman çizelgesi, önem dağılımı, model bilgisi panelleri
- **Video analizi** — kayıtlı video yükleyip toplu test-doğrulama
- **Kamera yönetimi** — webcam / RTSP / HTTP / video dosyası + bağlantı rehberi

---

## API Özeti

| Yöntem | Yol | Açıklama |
|---|---|---|
| GET | `/api/health` | Sağlık kontrolü |
| GET/POST/PATCH/DELETE | `/api/cameras` | Kamera CRUD |
| GET | `/api/events` · `/api/alerts` | Olay geçmişi, uyarılar |
| POST | `/api/alerts/{id}/ack` | Uyarı onaylama |
| GET | `/api/reports` · `/api/reports/model-status` | Rapor + model durumu |
| POST | `/api/analyze` | Video yükle & analiz et |
| WS | `/ws/stream/{camera_id}` | Canlı akış |

Etkileşimli dokümantasyon: **http://localhost:8000/docs**

---

## Bilinen Sınırlar (dürüst değerlendirme)

- Model karma veri setleriyle (RLVS, RWF-2000, CCTV Aggressive Poses) eğitildi; farklı sabit CCTV açılarında
  **alan farkı (domain gap)** olabilir. Yanlış alarm filtreleri bunu büyük
  ölçüde telafi eder; kalıcı çözüm hedef ortam videolarıyla ince ayardır.
- Raporlanan %97.7 doğrulama kümesi başarımıdır (video düzeyinde ayrılmış,
  sızıntısız split).
- CPU'da canlı akış skorlaması seyrekleşir (~2-3 sn'de bir); video analizi
  her donanımda tam çalışır.
