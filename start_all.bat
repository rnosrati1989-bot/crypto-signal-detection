@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Crypto Signal Bot

REM ===========================================================
REM  راه‌انداز پروژه:
REM    1) جمع‌آور لیکوییدیشن (منبع OKX) در پنجره‌ی جداگانه
REM    2) ربات تلگرام در همین پنجره (بایننس + مدل 15m)
REM  فقط روی همین فایل دوبار کلیک کن. بستن پنجره‌ها = توقف.
REM ===========================================================

REM --- پروکسی بایننس (اختیاری) --------------------------------
REM بایننس برای خیلی از IPهای ایران HTTP 451 می‌دهد (مسدودسازی جغرافیایی).
REM اگر پروکسی/VPN محلی داری، خط زیر را از حالت REM خارج کن و پورت درست را بگذار:
REM set "BINANCE_PROXY=http://127.0.0.1:10809"
set "NO_PROXY=localhost,127.0.0.1"
set "no_proxy=localhost,127.0.0.1"

echo.
echo [1/2] جمع‌آور لیکوییدیشن (پنجره‌ی جداگانه، مینیمایز) ...
start "Liq Collector" /min cmd /k venv\Scripts\python.exe liquidations_collector.py
timeout /t 3 /nobreak >nul

echo.
echo [2/2] ربات تلگرام ... (Ctrl+C برای توقف)
echo      برای ذخیره‌ی لاگ:  start_all.bat ^> logs\bot.log 2^>^&1
echo.
REM  رفتار ربات: فقط سیگنال معاملاتی (LONG/SHORT) پیام می‌دهد؛ HOLD/مومنتوم ساکت است.
REM  روزی یک‌بار ساعت ۲۳ «کارنامه‌ی سیگنال‌ها» (چند درصد حد سود / حد ضرر) ارسال می‌شود.
REM  اگر هشدار مومنتوم هم می‌خواهی:  start_all.bat --momentum
venv\Scripts\python.exe telegram_bot.py --check-every 5 --daily-hour 23 %*

echo.
echo [i] ربات متوقف شد.
pause
