@echo off
REM setup_env.bat - Windows Batch Setup Script
REM Digital Twin for SDN Networks

echo ==========================================
echo   Digital Twin Project - Windows Setup
echo ==========================================
echo.

REM Check if Python is installed
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python not found!
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check 'Add Python to PATH' during installation
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [OK] Found: %PYTHON_VERSION%
echo.

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo [X] requirements.txt not found!
    echo Make sure you're in the project root directory
    pause
    exit /b 1
)

REM Remove old venv if exists
if exist "venv" (
    echo Removing old virtual environment...
    rmdir /s /q venv
)

REM Create virtual environment
echo Creating Python virtual environment...
python -m venv venv

if not exist "venv" (
    echo [X] Failed to create virtual environment!
    pause
    exit /b 1
)

echo [OK] Virtual environment created
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [OK] pip upgraded
echo.

REM Install dependencies
echo Installing dependencies from requirements.txt...
echo (This may take several minutes, especially for TensorFlow)
echo.

pip install -r requirements.txt

if errorlevel 1 (
    echo [X] Some packages failed to install!
    echo Check the errors above
) else (
    echo [OK] All dependencies installed
)

echo.
echo ==========================================
echo          [OK] Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Test your setup: python test_setup.py
echo 2. Read the docs: docs\SETUP.md
echo.
echo To activate environment in the future:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate:
echo   deactivate
echo.

pause
