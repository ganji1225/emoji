@echo off
chcp 65001 >nul
title Emoji-TTS (Irodori Fork) - Web UI
echo ============================================
echo   Emoji-TTS - All-in-One Web UI
echo   Web UI: http://localhost:7863
echo   (Inference / Training / Dataset / Merge)
echo ============================================
echo.

cd /d "%~dp0"
call .venv\Scripts\activate.bat
python gradio_app.py --server-name 0.0.0.0 --server-port 7863

pause
