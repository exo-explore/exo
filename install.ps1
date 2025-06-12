# PowerShell installation script for Windows

# Check if Python 3.12 is installed
$python312 = Get-Command python3.12 -ErrorAction SilentlyContinue
$python3 = Get-Command python3 -ErrorAction SilentlyContinue
$python = Get-Command python -ErrorAction SilentlyContinue

if ($python312) {
    Write-Host "Python 3.12 is installed, proceeding with python3.12..." -ForegroundColor Green
    $pythonCmd = "python3.12"
} elseif ($python3) {
    $version = & python3 --version 2>&1
    Write-Host "The recommended version of Python to run exo with is Python 3.12, but $version is installed. Proceeding with $version" -ForegroundColor Yellow
    $pythonCmd = "python3"
} elseif ($python) {
    $version = & python --version 2>&1
    Write-Host "The recommended version of Python to run exo with is Python 3.12, but $version is installed. Proceeding with $version" -ForegroundColor Yellow
    $pythonCmd = "python"
} else {
    Write-Host "Python is not installed or not in PATH. Please install Python 3.12 or later." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Blue
& $pythonCmd -m venv venv-windows

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Blue
if (Test-Path "venv-windows\Scripts\Activate.ps1") {
    .\venv-windows\Scripts\Activate.ps1
} else {
    Write-Host "Failed to find virtual environment activation script" -ForegroundColor Red
    exit 1
}

# Install in development mode
Write-Host "Installing exo in development mode..." -ForegroundColor Blue
pip install -e .

Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "To run exo, activate the virtual environment with: .\venv-windows\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "Then run: exo" -ForegroundColor Cyan
