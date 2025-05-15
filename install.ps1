# Get Python version
$pythonVersion = & python --version 2>&1
if ($?) {
    $pythonVersion = $pythonVersion -replace '^Python ', '' # Extract version number
} else {
    Write-Error "Python is not installed or not found in PATH."
    exit 1
}

# Create a virtual environment
if (-not (Test-Path .venv)) {
    Write-Host "Creating virtual environment with Python $pythonVersion..."
    python -m venv .venv
    if ($?) {
        Write-Host "Virtual environment created successfully."
    } else {
        Write-Error "Failed to create virtual environment."
        exit 1
    }
} else {
    Write-Host "Virtual environment already exists."
}

# Activate the virtual environment
if (Test-Path .\.venv\Scripts\Activate.ps1) {
    Write-Host "Activating virtual environment..."
    & .\.venv\Scripts\Activate.ps1
} else {
    Write-Error "Activation script not found. Ensure the virtual environment was created correctly."
    exit 1
}

# Install the package in the virtual environment
$HasSetupPy = Test-Path "setup.py"
$HasToml = Test-Path "pyproject.toml"
if ($HasSetupPy -or $HasToml) {
    Write-Host "Installing package..."
    pip install .
    if ($?) {
        Write-Host "Package installed successfully."
    } else {
        Write-Error "Failed to install package."
        exit 1
    }
} else {
    Write-Error "No setup.py or pyproject.toml found in the current directory."
    exit 1
}