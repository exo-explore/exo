# PowerShell installation script for Windows with GPU auto-detection

Write-Host "=== exo Installation Script for Windows (PowerShell) ===" -ForegroundColor Cyan
Write-Host ""

# Check if Python 3.12 is installed
$python312 = Get-Command python3.12 -ErrorAction SilentlyContinue
$python3 = Get-Command python3 -ErrorAction SilentlyContinue
$python = Get-Command python -ErrorAction SilentlyContinue

if ($python312) {
    Write-Host "[OK] Python 3.12 is installed, proceeding with python3.12..." -ForegroundColor Green
    $pythonCmd = "python3.12"
} elseif ($python3) {
    $version = & python3 --version 2>&1
    Write-Host "[WARNING] The recommended version of Python to run exo with is Python 3.12, but $version is installed. Proceeding with $version" -ForegroundColor Yellow
    $pythonCmd = "python3"
} elseif ($python) {
    $version = & python --version 2>&1
    Write-Host "[WARNING] The recommended version of Python to run exo with is Python 3.12, but $version is installed. Proceeding with $version" -ForegroundColor Yellow
    $pythonCmd = "python"
} else {
    Write-Host "[ERROR] Python is not installed or not in PATH. Please install Python 3.12 or later." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Detect GPU and CUDA
Write-Host ""
Write-Host "[SCAN] Detecting hardware acceleration support..." -ForegroundColor Blue

$nvidiaGpuDetected = $false
$cudaAvailable = $false
$rtx5070TiDetected = $false

# Check for NVIDIA GPU
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    try {
        $gpuInfo = & nvidia-smi -L 2>$null
        if ($LASTEXITCODE -eq 0) {
            $nvidiaGpuDetected = $true
            Write-Host "[OK] NVIDIA GPU detected:" -ForegroundColor Green
            Write-Host $gpuInfo -ForegroundColor White
            
            # Check for RTX 5070 Ti
            if ($gpuInfo -match "5070") {
                Write-Host "[RTX 5070 Ti] Blackwell architecture GPU detected - optimizing for 16GB VRAM" -ForegroundColor Magenta
                $rtx5070TiDetected = $true
            }
            
            # Check driver version
            $driverVersion = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null
            if ($driverVersion -match "(\d+)\.") {
                $driverMajor = [int]$matches[1]
                if ($driverMajor -ge 566) {
                    Write-Host "[OK] Driver version $driverVersion supports latest GPUs including RTX 5070 Ti" -ForegroundColor Green
                } else {
                    Write-Host "[WARNING] Driver version $driverVersion may not fully support RTX 5070 Ti" -ForegroundColor Yellow
                    Write-Host "    Recommend updating to 566.03 or newer for optimal performance" -ForegroundColor Yellow
                }
            }
        }
    } catch {
        Write-Host "[INFO] nvidia-smi found but unable to detect GPU" -ForegroundColor Yellow
    }
} else {
    Write-Host "[INFO] No NVIDIA GPU detected, installing CPU-only version" -ForegroundColor Yellow
}

# Check for CUDA toolkit
if ($nvidiaGpuDetected) {
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvcc) {
        $cudaAvailable = $true
        Write-Host "[OK] CUDA toolkit detected:" -ForegroundColor Green
        $cudaVersion = & nvcc --version | Select-String "release"
        Write-Host "    $cudaVersion" -ForegroundColor White
    } else {
        Write-Host "[WARNING] CUDA toolkit not found - GPU acceleration will be limited" -ForegroundColor Yellow
        Write-Host "    Install CUDA toolkit for optimal performance" -ForegroundColor Yellow
        Write-Host "    Download from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    }
}

# Check if virtual environment already exists
Write-Host ""
if (Test-Path "venv-windows") {
    Write-Host "[INFO] Virtual environment 'venv-windows' already exists" -ForegroundColor Yellow
    $response = Read-Host "Do you want to recreate it? (y/N)"
    if ($response -match "^[Yy]") {
        Write-Host "Removing existing virtual environment..." -ForegroundColor Blue
        Remove-Item -Recurse -Force "venv-windows"
        Write-Host "Creating new virtual environment..." -ForegroundColor Blue
        & $pythonCmd -m venv venv-windows
    } else {
        Write-Host "Using existing virtual environment..." -ForegroundColor Blue
    }
} else {
    Write-Host "Creating virtual environment 'venv-windows'..." -ForegroundColor Blue
    & $pythonCmd -m venv venv-windows
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Blue
if (Test-Path "venv-windows\Scripts\Activate.ps1") {
    & ".\venv-windows\Scripts\Activate.ps1"
    
    # Verify activation worked
    $venvPython = ".\venv-windows\Scripts\python.exe"
    if (Test-Path $venvPython) {
        Write-Host "[OK] Virtual environment activated successfully" -ForegroundColor Green
        $pythonCmd = $venvPython  # Use venv python for all operations
    } else {
        Write-Host "[ERROR] Virtual environment activation may have failed" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[ERROR] Failed to find virtual environment activation script" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Install exo with automatic GPU detection
if ($nvidiaGpuDetected -and $cudaAvailable) {
    Write-Host "[INSTALL] Installing exo with NVIDIA GPU acceleration..." -ForegroundColor Green
    Write-Host "    Auto-detecting CUDA path and compiling with GPU support" -ForegroundColor White
    Write-Host "    This may take several minutes..." -ForegroundColor White
    Write-Host ""
    
    if ($rtx5070TiDetected) {
        Write-Host "[RTX 5070 Ti] The setup.py will automatically optimize for Blackwell architecture" -ForegroundColor Magenta
    }
} elseif ($nvidiaGpuDetected) {
    Write-Host "[WARNING] NVIDIA GPU detected but CUDA toolkit missing" -ForegroundColor Yellow
    Write-Host "    Installing CPU version - run fix_llamacpp_gpu.ps1 later for GPU support" -ForegroundColor Yellow
} else {
    Write-Host "[PACKAGE] Installing standard CPU version..." -ForegroundColor Blue
}

# Install dependencies with GPU support if available
Write-Host "Installing dependencies..." -ForegroundColor Blue
& $pythonCmd -m pip install --upgrade pip setuptools wheel

if ($nvidiaGpuDetected -and $cudaAvailable) {
    Write-Host "Installing llama-cpp-python with CUDA support..." -ForegroundColor Blue
    
    # Set environment variables for CUDA compilation
    $env:CMAKE_ARGS = "-DGGML_CUDA=on"
    $env:FORCE_CMAKE = "1"
    
    if ($rtx5070TiDetected) {
        Write-Host "[RTX 5070 Ti] Applying Blackwell architecture optimizations..." -ForegroundColor Magenta
        $env:CMAKE_ARGS = "-DGGML_CUDA=on -DCUDA_ARCHITECTURES=90 -DGGML_CUDA_FORCE_DMMV=ON -DGGML_CUDA_DMMV_F16=ON"
    } elseif ($gpuInfo -match "3060") {
        Write-Host "[RTX 3060] Applying Ampere architecture optimizations..." -ForegroundColor Cyan
        $env:CMAKE_ARGS = "-DGGML_CUDA=on -DCUDA_ARCHITECTURES=86"
    }
    
    Write-Host "CMAKE_ARGS: $($env:CMAKE_ARGS)" -ForegroundColor White
    
    # First attempt: Install from PyPI with CUDA compilation
    Write-Host "Attempting PyPI install with CUDA compilation..." -ForegroundColor Blue
    & $pythonCmd -m pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] llama-cpp-python with CUDA support installed from PyPI" -ForegroundColor Green
        
        # Verify CUDA support was actually compiled
        Write-Host "Verifying CUDA support..." -ForegroundColor Blue
        $testResult = & $pythonCmd -c "from llama_cpp import llama_cpp; print('CUDA support:', llama_cpp.llama_supports_gpu_offload())" 2>$null
        
        if ($testResult -match "True") {
            Write-Host "[OK] CUDA support verified successfully" -ForegroundColor Green
        } else {
            Write-Host "[WARNING] PyPI install succeeded but CUDA support not detected. Trying source compilation..." -ForegroundColor Yellow
            & $pythonCmd -m pip uninstall -y llama-cpp-python
            
            # Second attempt: Compile from source
            $gitAvailable = Get-Command git -ErrorAction SilentlyContinue
            if ($gitAvailable) {
                Write-Host "Compiling llama-cpp-python from source with CUDA..." -ForegroundColor Blue
                
                $tempDir = Join-Path $env:TEMP "llama-cpp-python"
                if (Test-Path $tempDir) {
                    Remove-Item -Recurse -Force $tempDir
                }
                
                git clone --recursive https://github.com/abetlen/llama-cpp-python.git $tempDir
                Push-Location $tempDir
                
                & $pythonCmd -m pip install -e . --verbose
                $compileResult = $LASTEXITCODE
                
                Pop-Location
                Remove-Item -Recurse -Force $tempDir
                
                if ($compileResult -eq 0) {
                    # Verify source compilation
                    $testResult = & $pythonCmd -c "from llama_cpp import llama_cpp; print('CUDA support:', llama_cpp.llama_supports_gpu_offload())" 2>$null
                    
                    if ($testResult -match "True") {
                        Write-Host "[OK] CUDA support compiled successfully from source" -ForegroundColor Green
                    } else {
                        Write-Host "[ERROR] Source compilation failed. Installing CPU version..." -ForegroundColor Red
                        & $pythonCmd -m pip uninstall -y llama-cpp-python
                        Remove-Item Env:CMAKE_ARGS -ErrorAction SilentlyContinue
                        & $pythonCmd -m pip install --upgrade llama-cpp-python
                    }
                } else {
                    Write-Host "[ERROR] Source compilation failed. Installing CPU version..." -ForegroundColor Red
                    Remove-Item Env:CMAKE_ARGS -ErrorAction SilentlyContinue
                    & $pythonCmd -m pip install --upgrade llama-cpp-python
                }
            } else {
                Write-Host "[ERROR] Git not available for source compilation. Installing CPU version..." -ForegroundColor Red
                Remove-Item Env:CMAKE_ARGS -ErrorAction SilentlyContinue
                & $pythonCmd -m pip install --upgrade llama-cpp-python
            }
        }
    } else {
        Write-Host "[ERROR] PyPI install failed. Trying source compilation..." -ForegroundColor Red
        
        # Second attempt: Compile from source
        $gitAvailable = Get-Command git -ErrorAction SilentlyContinue
        if ($gitAvailable) {
            Write-Host "Compiling llama-cpp-python from source with CUDA..." -ForegroundColor Blue
            
            $tempDir = Join-Path $env:TEMP "llama-cpp-python"
            if (Test-Path $tempDir) {
                Remove-Item -Recurse -Force $tempDir
            }
            
            git clone --recursive https://github.com/abetlen/llama-cpp-python.git $tempDir
            Push-Location $tempDir
            
            & $pythonCmd -m pip install -e . --verbose
            $compileResult = $LASTEXITCODE
            
            Pop-Location
            Remove-Item -Recurse -Force $tempDir
            
            if ($compileResult -eq 0) {
                # Verify source compilation
                $testResult = & $pythonCmd -c "from llama_cpp import llama_cpp; print('CUDA support:', llama_cpp.llama_supports_gpu_offload())" 2>$null
                
                if ($testResult -match "True") {
                    Write-Host "[OK] CUDA support compiled successfully from source" -ForegroundColor Green
                } else {
                    Write-Host "[ERROR] Source compilation failed. Installing CPU version..." -ForegroundColor Red
                    & $pythonCmd -m pip uninstall -y llama-cpp-python
                    Remove-Item Env:CMAKE_ARGS -ErrorAction SilentlyContinue
                    & $pythonCmd -m pip install --upgrade llama-cpp-python
                }
            } else {
                Write-Host "[ERROR] Source compilation failed. Installing CPU version..." -ForegroundColor Red
                Remove-Item Env:CMAKE_ARGS -ErrorAction SilentlyContinue
                & $pythonCmd -m pip install --upgrade llama-cpp-python
            }
        } else {
            Write-Host "[ERROR] Git not available. Installing CPU version..." -ForegroundColor Red
            Remove-Item Env:CMAKE_ARGS -ErrorAction SilentlyContinue
            & $pythonCmd -m pip install --upgrade llama-cpp-python
        }
    }
} else {
    Write-Host "Installing llama-cpp-python (CPU version)..." -ForegroundColor Blue
    & $pythonCmd -m pip install --upgrade llama-cpp-python
}

# Install exo in development mode
Write-Host "Installing exo in development mode..." -ForegroundColor Blue
& $pythonCmd -m pip install -e . --use-pep517

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install exo" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[OK] Installation complete!" -ForegroundColor Green

# Test GPU support if available
if ($nvidiaGpuDetected -and $cudaAvailable) {
    Write-Host ""
    Write-Host "[TEST] Testing GPU support..." -ForegroundColor Blue
    
    $testScript = @"
try:
    from llama_cpp import llama_cpp
    gpu_support = llama_cpp.llama_supports_gpu_offload() if hasattr(llama_cpp, 'llama_supports_gpu_offload') else False
    print(f'GPU offload support: {gpu_support}')
    if gpu_support:
        print('[SUCCESS] CUDA support successfully enabled!')
    else:
        print('[ERROR] CUDA support not detected')
except Exception as e:
    print(f'Error testing GPU support: {e}')
"@
    
    & $pythonCmd -c $testScript
    
    if ($rtx5070TiDetected) {
        Write-Host ""
        Write-Host "[SUCCESS] RTX 5070 Ti GPU acceleration is enabled and optimized!" -ForegroundColor Green
        Write-Host "    - Blackwell architecture support" -ForegroundColor White
        Write-Host "    - 16GB GDDR7 VRAM optimization" -ForegroundColor White
        Write-Host "    - Aggressive hybrid GPU/RAM allocation" -ForegroundColor White
    } else {
        Write-Host "[SUCCESS] GPU acceleration is enabled and ready to use!" -ForegroundColor Green
    }
} elseif ($nvidiaGpuDetected) {
    Write-Host ""
    if ($rtx5070TiDetected) {
        Write-Host "[TIP] To enable RTX 5070 Ti GPU acceleration:" -ForegroundColor Yellow
        Write-Host "    1. Install CUDA toolkit 12.0+ from https://developer.nvidia.com/cuda-downloads" -ForegroundColor White
        Write-Host "    2. Run: .\fix_windows_rtx_5070_ti.ps1 (as Administrator)" -ForegroundColor White
        Write-Host "    3. Ensure driver version 566.03 or newer" -ForegroundColor White
    } else {
        Write-Host "[TIP] To enable GPU acceleration:" -ForegroundColor Yellow
        Write-Host "    1. Install CUDA toolkit from https://developer.nvidia.com/cuda-downloads" -ForegroundColor White
        Write-Host "    2. Run: .\fix_llamacpp_gpu.ps1" -ForegroundColor White
    }
}

Write-Host ""
Write-Host "[USAGE] To run exo:" -ForegroundColor Cyan
Write-Host "    1. Activate virtual environment: .\venv-windows\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "    2. Run: python -m exo" -ForegroundColor White
Write-Host ""
Write-Host "[TIP] Virtual environment is located at: .\venv-windows\" -ForegroundColor Yellow
Write-Host "      Always activate it before running exo commands" -ForegroundColor Yellow

if ($rtx5070TiDetected) {
    Write-Host ""
    Write-Host "[RTX 5070 Ti] Monitor VRAM usage during inference:" -ForegroundColor Magenta
    Write-Host "    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 1" -ForegroundColor White
    Write-Host ""
    Write-Host "Expected behavior:" -ForegroundColor White
    Write-Host "    - Small models (3-7GB): Full VRAM usage on RTX 5070 Ti" -ForegroundColor White
    Write-Host "    - Large models (13GB+): Hybrid GPU/RAM usage" -ForegroundColor White
    Write-Host "    - GPU utilization should be > 0% during inference" -ForegroundColor White
}

Write-Host ""
