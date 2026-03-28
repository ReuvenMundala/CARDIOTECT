@echo off
title Cardiotect V3 - Clinical Workstation
echo ============================================
echo   CARDIOTECT V3 - Clinical Presentation GUI
echo ============================================
echo.

REM Activate virtual environment
call ..\venv\Scripts\activate

REM Launch the V3 GUI
echo Starting Cardiotect V3...
python -u run_v3.py

REM Keep window open on error
if %errorlevel% neq 0 (
    echo.
    echo Error occurred! Press any key to exit...
    pause > nul
)
