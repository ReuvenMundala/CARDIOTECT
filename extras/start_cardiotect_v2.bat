@echo off
title Cardiotect V2 - AI Calcium Scoring
echo ============================================
echo   CARDIOTECT V2 - Coronary Calcium Scoring
echo ============================================
echo.

REM Activate virtual environment
call ..\venv\Scripts\activate

REM Launch the V2 GUI
echo Starting Cardiotect V2...
python -u gui_v2\run_v2.py

REM Keep window open on error
if %errorlevel% neq 0 (
    echo.
    echo Error occurred! Press any key to exit...
    pause > nul
)
