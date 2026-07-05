"""Uygulama yapılandırması.

Ayarlar ortam değişkenleri (.env) ile geçersiz kılınabilir.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ── Genel ──────────────────────────────────────────────────────
    app_name: str = "Saldırgan Tespit Sistemi"
    app_version: str = "1.0.0"
    debug: bool = True

    # ── Veritabanı ─────────────────────────────────────────────────
    database_url: str = f"sqlite:///{BASE_DIR / 'storage' / 'app.db'}"

    # ── Depolama ───────────────────────────────────────────────────
    storage_dir: Path = BASE_DIR / "storage"
    upload_dir: Path = BASE_DIR / "storage" / "uploads"
    snapshot_dir: Path = BASE_DIR / "storage" / "snapshots"
    model_dir: Path = BASE_DIR / "models"

    # ── CORS ───────────────────────────────────────────────────────
    # "*": yayınlanan arayüz (Cloudflare Pages) + tünel senaryosu için
    # tüm origin'lere izin verilir (kimlik doğrulaması/çerez kullanılmıyor).
    cors_origins: list[str] = ["*"]

    # ── ML / Tespit ────────────────────────────────────────────────
    # Davranış sınıflandırıcısının saldırgan kabul ettiği eşik (0-1).
    # 0.60 → 0.80: selamlaşma/şakalaşma gibi hızlı ama masum hareketler
    # ara skorlar üretebiliyor; gerçek kavga skorları ~0.95+ sürdürüyor.
    threat_threshold: float = 0.80
    # LSTM girişine beslenen ardışık kare (zaman penceresi) uzunluğu.
    sequence_length: int = 16
    # Tespit edilen olay snapshot'larının saklanıp saklanmayacağı.
    save_snapshots: bool = True
    # Canlı akışta hedef FPS (işlem yükünü sınırlamak için).
    target_fps: int = 12
    # Şiddet (klip) modelinin canlı akışta kaç karede bir çalışacağı.
    # CPU'da çıkarım ~0.5 sn sürdüğü için her karede çalıştırmak akışı kilitler;
    # bu aralıkta bir kez çalışıp skoru aradaki karelerde tekrar kullanılır.
    clip_stride: int = 10
    # Alarm için üst üste kaç çıkarım penceresinin eşiği geçmesi gerektiği.
    # Gerçek şiddet sürekli olduğundan pencereleri doldurur; anlık/tek-tepe
    # hareketler (el sallama, kameraya yaklaşma vb.) elenir. 3 pencere
    # ≈ 3.3 sn doğrulama gecikmesi (12 FPS, stride 10). Selamlaşma/şakalaşma
    # gibi kısa süreli hareketler 4 pencereyi dolduramaz; gerçek kavga doldurur.
    clip_consecutive: int = 4
    # Tampondaki karelerin kaçta biri modele gider. Model, eğitimde ~5 sn'lik
    # videoya EŞİT YAYILMIŞ 20 kare gördü; ardışık 20 kare (~1.7 sn) vermek
    # eğitim dağılımına uymaz ve yanlış alarma yol açar. 3 → tampon 60 kare
    # (~5 sn) tutar, modele her 3. kare gider (eğitimle aynı zaman ölçeği).
    clip_frame_stride: int = 3
    # Kişi+poz tespitinin (Keypoint R-CNN) kaç karede bir çalışacağı.
    # GPU'da ~50 ms/kare; her karede çalıştırmak şiddet modeliyle birlikte
    # 12 FPS bütçesini zorlar. Aradaki karelerde son tespitler yeniden
    # kullanılır (görsel olarak fark edilmez).
    detect_stride: int = 2
    # Hareket-enerjisi kapısı: gerçek kavga sürekli ve yüksek hareket üretir;
    # el sıkışma/el ele tutuşma/sohbet gibi sahnelerde model yanlışlıkla
    # yüksek skor verse bile pencere içi hareket düşükse skor bastırılır.
    # motion_floor: karar skorunun kısılmaya başladığı bölgesel hareket eşiği
    # (0-255 ölçeğinde ortalama piksel farkı; ~2=konuşma, ~6+=itişme/kavga).
    motion_gate: bool = True
    motion_floor: float = 6.0
    # Gerçek kişi dedektörü (Keypoint R-CNN) sahnede hiç insan bulamazsa
    # şiddet alarmı üretme (insan yoksa kavga da yoktur → kesin yanlış alarm).
    require_person_for_alarm: bool = True
    # Kişiler-arası şiddet en az bu kadar kişi gerektirir. Tek kişi varken
    # (kameraya el sallama, tek başına hızlı hareket) 'saldırgan' etiketi
    # 'şüpheli'ye düşürülür; alarm üretilmez ama olay yine izlenebilir.
    min_persons_for_alarm: int = 2
    # Düşük ışık iyileştirme (form 2.3: aydınlatma düzeltme + gürültü azaltma).
    # Kare karanlıksa (ortalama parlaklık < eşik) CLAHE + hafif yumuşatma
    # uygulanır. Normal ışıkta kare DEĞİŞTİRİLMEZ — model eğitim dağılımının
    # dışına çıkmaz, mevcut davranış bozulmaz.
    low_light_enhance: bool = True
    low_light_threshold: int = 60  # 0-255 ortalama gri; ~60 altı "karanlık"

    def ensure_dirs(self) -> None:
        for d in (self.storage_dir, self.upload_dir, self.snapshot_dir, self.model_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
