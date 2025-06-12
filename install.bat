@echo off
REM Windows batch installation script for exo

echo Checking for Python installation...

REM Check if Python 3.12 is available
python3.12 --version >nul 2>&1
if %errorlevel% == 0 (
    echo Python 3.12 is installed, proceeding with python3.12...
    set PYTHON_CMD=python3.12
    goto :install
)

REM Check if python3 is available
python3 --version >nul 2>&1
if %errorlevel% == 0 (
    for /f "tokens=*" %%i in ('python3 --version') do set PYTHON_VERSION=%%i
    echo The recommended version of Python to run exo with is Python 3.12, but !PYTHON_VERSION! is installed. Proceeding with !PYTHON_VERSION!
    set PYTHON_CMD=python3
    goto :install
)

REM Check if python is available
python --version >nul 2>&1
if %errorlevel% == 0 (
    for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
    echo The recommended version of Python to run exo with is Python 3.12, but !PYTHON_VERSION! is installed. Proceeding with !PYTHON_VERSION!
    set PYTHON_CMD=python
    goto :install
)

echo Python is not installed or not in PATH. Please install Python 3.12 or later.
echo Download from: https://www.python.org/downloads/
pause
exit /b 1

:install
echo Creating virtual environment...
%PYTHON_CMD% -m venv venv-windows
if %errorlevel% neq 0 (
    echo Failed to create virtual environment
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv-windows\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment
    pause
    exit /b 1
)

echo Installing exo in development mode...
pip install -e .
if %errorlevel% neq 0 (
    echo Failed to install exo
    pause
    exit /b 1
)

echo.
echo Installation complete!
echo To run exo, activate the virtual environment with: venv-windows\Scripts\activate.bat
echo Then run: exo
pause
