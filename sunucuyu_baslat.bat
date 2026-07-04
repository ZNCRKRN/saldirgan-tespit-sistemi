@echo off
chcp 65001 >nul
title Saldirgan Tespit - Baslatici
echo ============================================
echo   SALDIRGAN TESPIT SISTEMI - SUNUCU + TUNEL
echo ============================================
echo.

rem 1) Backend'i kendi penceresinde baslat (GPU modelleri ~20 sn yuklenir)
cd /d "%~dp0backend"
start "Backend (port 8000)" cmd /k "set PYTHONUTF8=1 && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"

rem 2) Cloudflare tuneli baslat: internete acilan ucretsiz adres verir
start "Cloudflare Tunnel" cmd /k ""C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8000"

echo Iki pencere acildi:
echo.
echo  [1] Backend      : model yukleniyor, "Uvicorn running" yazisini bekle
echo  [2] Tunnel       : icinde  https://XXXX.trycloudflare.com  adresi cikacak
echo.
echo SONRAKI ADIM:
echo   - Tunnel penceresindeki https adresini kopyala
echo   - Yayindaki siteyi ac, sol alttaki "Sunucu" ayarina yapistir
echo   - Sayfa yenilenince sistem senin bilgisayarina baglanir
echo.
pause
