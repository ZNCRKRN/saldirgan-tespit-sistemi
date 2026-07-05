@echo off
chcp 65001 >nul
setlocal
title Saldirgan Tespit Sistemi
cd /d "%~dp0"

echo ==============================================================
echo   SALDIRGAN TESPIT SISTEMI - KURULUM + BASLATMA (tek tik)
echo   Ilk calistirma: paketler + model indirilir (~10 dk, tek sefer)
echo   Sonraki calistirmalar: dogrudan baslar (~30 sn)
echo ==============================================================
echo.

rem ── 1) Python kontrolu ───────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [HATA] Python bulunamadi!
    echo Lutfen once Python 3.10-3.12 kurun: https://www.python.org/downloads/
    echo Kurulumda "Add Python to PATH" kutusunu isaretlemeyi unutmayin.
    pause
    exit /b 1
)

rem ── 2) Sanal ortam ───────────────────────────────────────────────
if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Sanal ortam olusturuluyor...
    python -m venv .venv
)
set PY=.venv\Scripts\python.exe

rem ── 3) Paketler (ilk seferde kurulur) ────────────────────────────
%PY% -c "import fastapi, torch, cv2" >nul 2>&1
if errorlevel 1 (
    echo [2/4] Paketler kuruluyor... ^(ilk sefer 5-10 dk surebilir^)
    %PY% -m pip install --upgrade pip -q
    %PY% -m pip install -r backend\requirements.txt
    if errorlevel 1 (
        echo [HATA] Paket kurulumu basarisiz. Internet baglantinizi kontrol edin.
        pause
        exit /b 1
    )
) else (
    echo [2/4] Paketler zaten kurulu.
)

rem ── 4) Egitilmis model (ilk seferde indirilir, ~233 MB) ──────────
if not exist "backend\models\best_model.pth" (
    echo [3/4] Egitilmis model indiriliyor ^(~233 MB^)...
    curl -L -o "backend\models\best_model.pth" "https://github.com/ZNCRKRN/saldirgan-tespit-sistemi/releases/download/v1.0/best_model.pth"
    if errorlevel 1 (
        echo [HATA] Model indirilemedi. Elle indirmek icin:
        echo https://github.com/ZNCRKRN/saldirgan-tespit-sistemi/releases
        echo dosyayi backend\models\ klasorune koyun.
        pause
        exit /b 1
    )
) else (
    echo [3/4] Egitilmis model zaten mevcut.
)

rem ── 5) Sunucuyu baslat ve tarayiciyi ac ──────────────────────────
echo [4/4] Sistem baslatiliyor... ^(model yukleniyor, ~20-30 sn^)
echo.
echo   Arayuz:  http://localhost:8000
echo   Kapatmak icin bu pencereyi kapatin veya Ctrl+C.
echo.
start "" "http://localhost:8000"
set PYTHONUTF8=1
cd backend
..\%PY% -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
