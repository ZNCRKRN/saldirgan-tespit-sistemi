# Kapalı Alanlarda Derin Öğrenme Tabanlı Saldırgan Tespiti

TÜBİTAK 2209-A araştırma projesi için geliştirilen, çok açılı güvenlik
kamerası görüntülerinden **derin öğrenme ile saldırgan davranış tespiti**
yapan uçtan uca sistem.

**Pipeline:** `R-CNN (kişi tespiti) → OpenPose (iskelet/poz) → LSTM + Attention
(zaman serisi davranış analizi) → Saldırgan/Normal sınıflandırma → Uyarı + Raporlama`

> Eğitilmiş modeliniz henüz yokken sistem **yedek (mock) modda uçtan uca
> çalışır**. Modelinizi `backend/models/` klasörüne bıraktığınızda otomatik
> olarak gerçek modele geçer (bkz. `backend/models/README.md`).

---

## Mimari

```
tübitak2209/
├── backend/                 FastAPI + ML pipeline (Python)
│   ├── app/
│   │   ├── main.py          Uygulama + CORS + statik snapshot
│   │   ├── config.py        Ayarlar (eşik, sequence_length, FPS…)
│   │   ├── database.py      SQLite + seed
│   │   ├── models.py        Camera / Event / Alert tabloları
│   │   ├── crud.py          Olay & uyarı kaydı, rapor sorguları
│   │   ├── routers/         cameras, events, reports, stream(ws)
│   │   └── ml/
│   │       ├── pipeline.py            Orkestratör + görselleştirme
│   │       ├── person_detector.py     Aşama 1 — R-CNN (yedek: HOG)
│   │       ├── pose_estimator.py      Aşama 2 — OpenPose (yedek: sentetik)
│   │       ├── behavior_classifier.py Aşama 3 — LSTM+Attention  ← MODEL BURAYA
│   │       ├── tracker.py             Kişi takibi (zaman serisi)
│   │       └── demo_source.py         Kamerasız demo sahnesi
│   └── models/              ← eğitilmiş ağırlıklar buraya
└── frontend/                React + Vite + Tailwind + Recharts
    └── src/pages/           Dashboard, LiveMonitor, Alerts, Events,
                             Reports, Cameras, VideoAnalysis
```

---

## Kurulum & Çalıştırma

### 1) Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python run.py                    # http://localhost:8000  (API docs: /docs)
```

İlk açılışta SQLite veritabanı oluşturulur ve 3 örnek kamera eklenir.

### 2) Frontend

```bash
cd frontend
npm install
npm run dev                      # http://localhost:5173
```

Vite, `/api`, `/snapshots` ve `/ws` isteklerini otomatik olarak backend'e
(`localhost:8000`) yönlendirir.

Tarayıcıda **http://localhost:5173** → *Canlı İzleme* sekmesinde demo akışı
ve saldırgan tespiti anında görünür.

---

## Eğitilmiş Modeli Bağlama

`backend/models/` klasörüne şunlardan birini koymanız yeterli:

| Çerçeve | Dosya |
|---|---|
| Keras/TensorFlow | `behavior_model.h5` veya `behavior_model.keras` |
| PyTorch | `behavior_model.pt` veya `behavior_model.pth` |

Detaylar ve beklenen giriş/çıkış formatı: **`backend/models/README.md`**.
Giriş/çıkış şekliniz farklıysa `app/ml/behavior_classifier.py` içindeki
`_prepare` / `_interpret` metotlarını uyarlayın. Gerçek R-CNN ve OpenPose'u
da `person_detector.py` / `pose_estimator.py` içindeki ilgili sınıflara
takabilirsiniz.

---

## Özellikler (forma karşılık gelir)

- **Canlı izleme** — WebSocket ile gerçek zamanlı kare + tespit (bbox, iskelet, skor)
- **Uyarı sistemi** (form 2.5) — saldırgan tespitinde anlık uyarı sinyali + onaylama
- **Veri kaydı** (form 2.6) — olay/uyarı veritabanı + snapshot arşivi
- **Raporlama** (form 2.6.3) — zaman çizelgesi, önem dağılımı, istatistik grafikleri
- **Video analizi** (İP 4) — kayıtlı video yükleyip toplu test-doğrulama
- **Kamera yönetimi** — webcam / RTSP-HTTP URL / demo kaynak

---

## API Özeti

| Yöntem | Yol | Açıklama |
|---|---|---|
| GET | `/api/health` | Sağlık kontrolü |
| GET/POST/PATCH/DELETE | `/api/cameras` | Kamera CRUD |
| GET | `/api/events` | Olay geçmişi (filtreli) |
| GET | `/api/alerts` | Uyarılar |
| POST | `/api/alerts/{id}/ack` | Uyarı onaylama |
| GET | `/api/reports` | Tam rapor (özet+zaman+önem) |
| GET | `/api/reports/model-status` | Model/pipeline durumu |
| POST | `/api/analyze` | Video yükle & analiz et |
| WS | `/ws/stream/{camera_id}` | Canlı akış |

Tam etkileşimli dokümantasyon: **http://localhost:8000/docs**
