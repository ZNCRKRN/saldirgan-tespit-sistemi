# Eğitilmiş Modelinizi Buraya Koyun

Sistem başlangıçta bu klasörde aşağıdaki dosyaları arar ve bulduğunda
otomatik olarak gerçek modeli kullanır. Hiçbiri yoksa sezgisel (heuristic)
yedek devreye girer ve sistem yine de uçtan uca çalışır.

## ✅ AKTİF — Şiddet/Saldırganlık modeli (sahne düzeyi)

Hâlihazırda yüklü model: **`best_model.pth`** — ResNet50 + BiLSTM + Attention
(Kaggle "Real Life Violence Situations" ile eğitildi). `violence_classifier.py`
bu dosyayı `best_model.pth` / `best_model.pth.zip` / `violence_model.pth` adıyla
arar.

- **Giriş:** 20 ardışık RGB kare, 224×224, ImageNet normalize → `(1, 20, 3, 224, 224)`
- **Çıkış:** tek nöron sigmoid → şiddet olasılığı `[0,1]` (1=saldırgan, 0=normal)
- **Checkpoint formatı:** `{model_state_dict, val_acc, history, optimizer_state_dict, ...}`
  veya `{..., model_config:{hidden_size,num_layers,dropout,num_frames,img_size}}`.
- Bu model **sahne düzeyinde** çalışır (kişi-bazlı iskelet değil); skor tespit
  edilen tüm kişilere yansıtılır. Canlı akışta `clip_stride` (config) karede bir
  çalışır; `/api/analyze` videoyu pencerelere bölüp her birini sınıflandırır.
- ⚠️ Demo sahnesi sentetik olduğundan onda hâlâ sezgisel sınıflandırıcı çalışır;
  gerçek model gerçek kamera/video akışlarında devreye girer.

> Gereksinim: backend ortamında `torch` + `torchvision` kurulu olmalı.

---

## (Alternatif) Pose-tabanlı davranış modeli (LSTM + Attention)

Pose/keypoint zaman serisi üzerinde çalışan bir model kullanmak isterseniz:

| Çerçeve | Dosya adı | Açıklama |
|---|---|---|
| Keras / TensorFlow | `behavior_model.h5` veya `behavior_model.keras` | `tf.keras.models.load_model` ile yüklenir |
| Keras SavedModel | `behavior_model/` (klasör) | SavedModel formatı |
| PyTorch | `behavior_model.pt` veya `behavior_model.pth` | `torch.load` ile yüklenir |

### Beklenen giriş/çıkış

- **Giriş şekli:** `(1, T, F)` — `T = sequence_length` (varsayılan 16),
  `F` = kişi başına özellik (keypoint x,y normalize). `config.py` içinden
  `sequence_length` ayarlanabilir.
- **Çıkış:**
  - Tek nöron (sigmoid) → saldırgan olasılığı `[0,1]`, veya
  - Çok sınıflı (softmax) → `behavior_classifier.py` içindeki `ACTIONS`
    sırasına göre yorumlanır; saldırgan eylemlerin toplam olasılığı skor olur.

Giriş/çıkış formatınız farklıysa `app/ml/behavior_classifier.py` içindeki
`_prepare` / `_interpret` metotlarını uyarlayın.

## (Opsiyonel) R-CNN kişi dedektörü

Gerçek R-CNN ağırlıklarınızı kullanmak için `app/ml/person_detector.py`
içindeki `RCNNPersonDetector` sınıfındaki yorum satırlarını açın. Aksi halde
OpenCV HOG yaya dedektörü yedek olarak çalışır.

## (Opsiyonel) OpenPose / MediaPipe poz çıkarımı

`app/ml/pose_estimator.py` içindeki `MediaPipePoseEstimator` veya
`OpenPoseEstimator` sınıflarını doldurun. Aksi halde sentetik iskelet yedeği
kullanılır.
