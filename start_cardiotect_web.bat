@echo off
title Cardiotect Web GUI
echo ============================================
echo   CARDIOTECT WEB GUI (Main Application)
echo ============================================
echo.

REM Determine script directory
set "SCRIPT_DIR=%~dp0"

REM Check if virtual environment exists, if not run setup
if not exist "%SCRIPT_DIR%venv" (
    echo Virtual environment not found. Running first-time setup...
    if exist "%SCRIPT_DIR%extras\setup.bat" (
        call "%SCRIPT_DIR%extras\setup.bat"
        if %errorlevel% neq 0 (
            echo Setup failed. Please run extras\setup.bat manually.
            pause
            exit /b 1
        )
    ) else (
        echo ERROR: setup.bat not found in extras folder.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call "%SCRIPT_DIR%venv\Scripts\activate"
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

REM Launch the Web GUI
echo Starting Cardiotect Web GUI...
cd /d "%SCRIPT_DIR%extras\web_gui"
python run_web_gui.py

REM Keep window open on error
if %errorlevel% neq 0 (
    echo.
    echo Error occurred! Press any key to exit...
    pause > nul
)