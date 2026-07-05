"""Veri güvenliği yardımcıları (form 2.6.1: "veriler şifrelenmiş ortamda").

Tespit snapshot'ları kişisel veri (insan görüntüsü) içerdiğinden diskte
AES-128 (Fernet) ile şifreli saklanır; arayüze sunulurken bellekte çözülür.
Anahtar ilk çalıştırmada üretilir ve `storage/.snapshot_key` dosyasında
tutulur — anahtar olmadan snapshot dosyaları açılamaz.
"""
from __future__ import annotations

from pathlib import Path

from .config import settings

_fernet = None


def get_fernet():
    """Fernet örneğini döndür (anahtar yoksa üret). Kütüphane yoksa None."""
    global _fernet
    if _fernet is not None:
        return _fernet
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        return None

    key_path: Path = settings.storage_dir / ".snapshot_key"
    if key_path.exists():
        key = key_path.read_bytes().strip()
    else:
        key = Fernet.generate_key()
        key_path.write_bytes(key)
    _fernet = Fernet(key)
    return _fernet


def encrypt_bytes(data: bytes) -> bytes | None:
    """Veriyi şifrele; şifreleme kullanılamıyorsa None döner."""
    f = get_fernet()
    return f.encrypt(data) if f is not None else None


def decrypt_bytes(data: bytes) -> bytes | None:
    """Şifreli veriyi çöz; anahtar/format uymazsa None döner."""
    f = get_fernet()
    if f is None:
        return None
    try:
        return f.decrypt(data)
    except Exception:  # noqa: BLE001 — bozuk/yabancı dosya
        return None
