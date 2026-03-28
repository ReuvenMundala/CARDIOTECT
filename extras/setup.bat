@echo off
setlocal enabledelayedexpansion
title Cardiotect Setup
echo ============================================
echo   CARDIOTECT SETUP
echo   Automated installation and configuration
echo ============================================
echo.

REM Determine root directory (parent of this script)
set "ROOT_DIR=%~dp0.."

REM 1. Check Python installation
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8+ and add to PATH.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo Python found.

REM 2. Create virtual environment if not exists in root
echo.
echo [2/4] Setting up virtual environment...
if not exist "%ROOT_DIR%\venv" (
    echo Creating virtual environment in root...
    python -m venv "%ROOT_DIR%\venv"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
call "%ROOT_DIR%\venv\Scripts\activate"
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

REM 3. Install requirements (using full path)
echo.
echo [3/4] Installing dependencies...
python -m pip install --upgrade pip >nul 2>&1
pip install -r "%~dp0requirements.txt"
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)
echo Dependencies installed successfully.

REM 4. Dataset configuration
echo.
echo [4/4] Dataset configuration...
if defined CARDIOTECT_DATASET_ROOT (
    echo Dataset root already set: %CARDIOTECT_DATASET_ROOT%
) else (
    echo Dataset root not set.
    echo.
    echo Please set the CARDIOTECT_DATASET_ROOT environment variable.
    echo Example: setx CARDIOTECT_DATASET_ROOT "C:\path\to\dataset"
    echo.
    echo Or create a .env file in the root directory with:
    echo   CARDIOTECT_DATASET_ROOT=C:\path\to\dataset
    echo.
    echo See .env.example in root for reference.
)

REM Create .env file in root if not exists
if not exist "%ROOT_DIR%\.env" (
    echo Creating .env file from template in root...
    copy "%ROOT_DIR%\.env.example" "%ROOT_DIR%\.env" >nul
    echo Please edit .env file in root to set your dataset path.
)

echo.
echo ============================================
echo   SETUP COMPLETE
echo ============================================
echo.
echo To launch the Web GUI (main application):
echo   Double-click start_cardiotect_web.bat in the root folder.
echo.
echo To launch desktop GUIs (optional):
echo   See extras\ folder for V2 and V3 GUIs.
echo.
echo For detailed instructions, see README.md in the root folder.
echo.
pause