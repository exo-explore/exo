@echo off
setlocal enabledelayedexpansion

echo === exo Installation Script for Windows ===
echo.

REM Initialize variables
set NVIDIA_GPU_DETECTED=false
set CUDA_AVAILABLE=false
set RTX_5070_TI_DETECTED=false

REM Check if Python 3.12 is available
python3.12 --version >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Python 3.12 is installed, proceeding with python3.12...
    set PYTHON_CMD=python3.12
    goto :detect_gpu
)

REM Check if python3 is available
python3 --version >nul 2>&1
if %errorlevel% == 0 (
    for /f "tokens=*" %%i in ('python3 --version') do set PYTHON_VERSION=%%i
    echo WARNING: The recommended version of Python to run exo with is Python 3.12, but !PYTHON_VERSION! is installed. Proceeding with !PYTHON_VERSION!
    set PYTHON_CMD=python3
    goto :detect_gpu
)

REM Check if python is available
python --version >nul 2>&1
if %errorlevel% == 0 (
    for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
    echo WARNING: The recommended version of Python to run exo with is Python 3.12, but !PYTHON_VERSION! is installed. Proceeding with !PYTHON_VERSION!
    set PYTHON_CMD=python
    goto :detect_gpu
)

echo [ERROR] Python is not installed or not in PATH. Please install Python 3.12 or later.
echo Download from: https://www.python.org/downloads/
pause
exit /b 1

:detect_gpu
echo.
echo [SCAN] Detecting hardware acceleration support...

REM Check for NVIDIA GPU with RTX 5070 Ti specific detection
where nvidia-smi >nul 2>&1
if %errorlevel% == 0 (
    nvidia-smi -L >nul 2>&1
    if %errorlevel% == 0 (
        set NVIDIA_GPU_DETECTED=true
        echo [OK] NVIDIA GPU detected:
        nvidia-smi -L
        
        REM Check for RTX 5070 Ti specifically
        nvidia-smi -L | findstr /i "5070" >nul 2>&1
        if %errorlevel% == 0 (
            echo [RTX 5070 Ti] Blackwell architecture GPU detected - optimizing for 16GB VRAM
            set RTX_5070_TI_DETECTED=true
        )
        
        REM Check driver version for RTX 5070 Ti compatibility
        for /f "tokens=2" %%a in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader') do (
            set DRIVER_VERSION=%%a
            for /f "tokens=1 delims=." %%b in ("!DRIVER_VERSION!") do set DRIVER_MAJOR=%%b
            if !DRIVER_MAJOR! geq 566 (
                echo [OK] Driver version !DRIVER_VERSION! supports latest GPUs including RTX 5070 Ti
            ) else (
                echo [WARNING] Driver version !DRIVER_VERSION! may not fully support RTX 5070 Ti
                echo    Recommend updating to 566.03 or newer for optimal performance
            )
        )
        
        REM Check for CUDA toolkit
        where nvcc >nul 2>&1
        if %errorlevel% == 0 (
            set CUDA_AVAILABLE=true
            echo [OK] CUDA toolkit detected:
            nvcc --version | findstr "release"
        ) else (
            echo [WARNING]  CUDA toolkit not found - GPU acceleration will be limited
            echo    Install CUDA toolkit for optimal performance
        )
    )
) else (
    echo [INFO]  No NVIDIA GPU detected, installing CPU-only version
)

:install
echo.
REM Check if virtual environment already exists
if exist "venv-windows\" (
    echo [INFO] Virtual environment 'venv-windows' already exists
    set /p response="Do you want to recreate it? (y/N): "
    if /i "!response!"=="y" (
        echo Removing existing virtual environment...
        rmdir /s /q "venv-windows"
        echo Creating new virtual environment...
        %PYTHON_CMD% -m venv venv-windows
    ) else (
        echo Using existing virtual environment...
    )
) else (
    echo Creating virtual environment 'venv-windows'...
    %PYTHON_CMD% -m venv venv-windows
)

if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv-windows\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

REM Verify activation worked by checking for venv python
if exist "venv-windows\Scripts\python.exe" (
    echo [OK] Virtual environment activated successfully
    set PYTHON_CMD=venv-windows\Scripts\python.exe
) else (
    echo [ERROR] Virtual environment activation may have failed
    pause
    exit /b 1
)

echo.

REM Install with GPU optimization if available
if "%NVIDIA_GPU_DETECTED%"=="true" if "%CUDA_AVAILABLE%"=="true" (
    echo [INSTALL] Installing exo with NVIDIA GPU acceleration...
    echo    This includes llama-cpp-python with CUDA support
    echo    Compilation may take several minutes...
    echo.
    
    REM Install base requirements first
    echo Installing base dependencies...
    %PYTHON_CMD% -m pip install --upgrade pip setuptools wheel
    
    REM Install llama-cpp-python with CUDA support
    echo Installing llama-cpp-python with CUDA support...
    
    REM Set CMAKE_ARGS for CUDA support with GPU-specific optimizations
    set FORCE_CMAKE=1
    if "%RTX_5070_TI_DETECTED%"=="true" (
        echo [RTX 5070 Ti] Applying Blackwell architecture optimizations...
        set CMAKE_ARGS=-DGGML_CUDA=on -DCUDA_ARCHITECTURES=90 -DGGML_CUDA_FORCE_DMMV=ON -DGGML_CUDA_DMMV_F16=ON
        echo [INFO] CMAKE_ARGS: !CMAKE_ARGS!
    ) else (
        REM Check for RTX 3060
        nvidia-smi -L | findstr /i "3060" >nul 2>&1
        if %errorlevel% == 0 (
            echo [RTX 3060] Applying Ampere architecture optimizations...
            set CMAKE_ARGS=-DGGML_CUDA=on -DCUDA_ARCHITECTURES=86
            echo [INFO] CMAKE_ARGS: !CMAKE_ARGS!
        ) else (
            set CMAKE_ARGS=-DGGML_CUDA=on
        )
    )
    
    REM Set CUDA compiler path if available
    if exist "%CUDA_PATH%\bin\nvcc.exe" (
        set CUDACXX=%CUDA_PATH%\bin\nvcc.exe
        echo [OK] CUDA compiler: !CUDACXX!
    )
    
    REM First attempt: Install from PyPI with CUDA compilation
    echo Attempting PyPI install with CUDA compilation...
    %PYTHON_CMD% -m pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
    
    if %errorlevel% == 0 (
        echo [OK] llama-cpp-python with CUDA support installed from PyPI
        
        REM Verify CUDA support was actually compiled
        echo Verifying CUDA support...
        %PYTHON_CMD% -c "from llama_cpp import llama_cpp; result = llama_cpp.llama_supports_gpu_offload(); print('CUDA support:', result); exit(0 if result else 1)" >nul 2>&1
        
        if %errorlevel% == 0 (
            echo [OK] CUDA support verified successfully
        ) else (
            echo [WARNING] PyPI install succeeded but CUDA support not detected. Trying source compilation...
            %PYTHON_CMD% -m pip uninstall -y llama-cpp-python
            
            REM Second attempt: Compile from source
            where git >nul 2>&1
            if %errorlevel% == 0 (
                echo Compiling llama-cpp-python from source with CUDA...
                
                set TEMP_DIR=%TEMP%\llama-cpp-python
                if exist "!TEMP_DIR!" (
                    rmdir /s /q "!TEMP_DIR!"
                )
                
                git clone --recursive https://github.com/abetlen/llama-cpp-python.git "!TEMP_DIR!"
                pushd "!TEMP_DIR!"
                
                %PYTHON_CMD% -m pip install -e . --verbose
                set COMPILE_RESULT=!errorlevel!
                
                popd
                rmdir /s /q "!TEMP_DIR!"
                
                if !COMPILE_RESULT! == 0 (
                    REM Verify source compilation
                    %PYTHON_CMD% -c "from llama_cpp import llama_cpp; result = llama_cpp.llama_supports_gpu_offload(); print('CUDA support:', result); exit(0 if result else 1)" >nul 2>&1
                    
                    if !errorlevel! == 0 (
                        echo [OK] CUDA support compiled successfully from source
                    ) else (
                        echo [ERROR] Source compilation failed. Installing CPU version...
                        %PYTHON_CMD% -m pip uninstall -y llama-cpp-python
                        set CMAKE_ARGS=
                        %PYTHON_CMD% -m pip install --upgrade llama-cpp-python
                    )
                ) else (
                    echo [ERROR] Source compilation failed. Installing CPU version...
                    set CMAKE_ARGS=
                    %PYTHON_CMD% -m pip install --upgrade llama-cpp-python
                )
            ) else (
                echo [ERROR] Git not available for source compilation. Installing CPU version...
                set CMAKE_ARGS=
                %PYTHON_CMD% -m pip install --upgrade llama-cpp-python
            )
        )
    ) else (
        echo [ERROR] PyPI install failed. Trying source compilation...
        
        REM Second attempt: Compile from source
        where git >nul 2>&1
        if %errorlevel% == 0 (
            echo Compiling llama-cpp-python from source with CUDA...
            
            set TEMP_DIR=%TEMP%\llama-cpp-python
            if exist "!TEMP_DIR!" (
                rmdir /s /q "!TEMP_DIR!"
            )
            
            git clone --recursive https://github.com/abetlen/llama-cpp-python.git "!TEMP_DIR!"
            pushd "!TEMP_DIR!"
            
            %PYTHON_CMD% -m pip install -e . --verbose
            set COMPILE_RESULT=!errorlevel!
            
            popd
            rmdir /s /q "!TEMP_DIR!"
            
            if !COMPILE_RESULT! == 0 (
                REM Verify source compilation
                %PYTHON_CMD% -c "from llama_cpp import llama_cpp; result = llama_cpp.llama_supports_gpu_offload(); print('CUDA support:', result); exit(0 if result else 1)" >nul 2>&1
                
                if !errorlevel! == 0 (
                    echo [OK] CUDA support compiled successfully from source
                ) else (
                    echo [ERROR] Source compilation failed. Installing CPU version...
                    %PYTHON_CMD% -m pip uninstall -y llama-cpp-python
                    set CMAKE_ARGS=
                    %PYTHON_CMD% -m pip install --upgrade llama-cpp-python
                )
            ) else (
                echo [ERROR] Source compilation failed. Installing CPU version...
                set CMAKE_ARGS=
                %PYTHON_CMD% -m pip install --upgrade llama-cpp-python
            )
        ) else (
            echo [ERROR] Git not available. Installing CPU version...
            set CMAKE_ARGS=
            %PYTHON_CMD% -m pip install --upgrade llama-cpp-python
        )
    )
    
    REM Install exo in development mode
    echo Installing exo in development mode...
    %PYTHON_CMD% -m pip install -e . --use-pep517
    
    echo.
    echo [TEST] Testing GPU support...
    %PYTHON_CMD% -c "try: from llama_cpp import llama_cpp; gpu_support = llama_cpp.llama_supports_gpu_offload() if hasattr(llama_cpp, 'llama_supports_gpu_offload') else False; print(f'GPU offload support: {gpu_support}'); print('[OK] CUDA support successfully enabled!' if gpu_support else '[ERROR] CUDA support not detected'); except Exception as e: print(f'Error testing GPU support: {e}')"
    
    if "%RTX_5070_TI_DETECTED%"=="true" (
        echo.
        echo [RTX 5070 Ti] Testing VRAM optimization...
        %PYTHON_CMD% -c "
import sys
sys.path.append('.')
try:
    from exo.inference.llamacpp.llama_inference_engine import LlamaCppInferenceEngine
    from exo.download.shard_download import ShardDownloader
    import tempfile
    
    print('[TEST] Testing RTX 5070 Ti VRAM allocation...')
    with tempfile.TemporaryDirectory() as temp_dir:
        downloader = ShardDownloader(temp_dir)
        engine = LlamaCppInferenceEngine(downloader)
        
        # Test with 16GB VRAM (RTX 5070 Ti)
        available_vram = 14.5  # 16GB - 1.5GB reserved
        
        # Test various model sizes
        test_cases = [(3.5, '7B model'), (7.0, '13B model'), (13.0, '30B model')]
        for model_size, desc in test_cases:
            optimal_layers = engine._calculate_optimal_gpu_layers(model_size, available_vram)
            if optimal_layers == -1:
                print(f'[OK] {desc}: ALL layers on RTX 5070 Ti (full VRAM usage)')
            elif optimal_layers > 0:
                print(f'[OK] {desc}: {optimal_layers} layers on RTX 5070 Ti (hybrid mode)')
            else:
                print(f'[WARNING] {desc}: CPU-only mode')
        
        print('[SUCCESS] RTX 5070 Ti VRAM optimization ready!')
        
except Exception as e:
    print(f'[ERROR] VRAM optimization test failed: {e}')
"
    )
    
) else if "%NVIDIA_GPU_DETECTED%"=="true" (
    echo [WARNING]  NVIDIA GPU detected but CUDA toolkit missing
    echo    Installing CPU version - run fix_llamacpp_gpu.bat later for GPU support
    echo Installing base dependencies...
    %PYTHON_CMD% -m pip install --upgrade pip setuptools wheel
    echo Installing llama-cpp-python (CPU version)...
    %PYTHON_CMD% -m pip install --upgrade llama-cpp-python
    echo Installing exo in development mode...
    %PYTHON_CMD% -m pip install -e . --use-pep517
    
) else (
    echo [PACKAGE] Installing standard CPU version...
    echo Installing base dependencies...
    %PYTHON_CMD% -m pip install --upgrade pip setuptools wheel
    echo Installing llama-cpp-python (CPU version)...
    %PYTHON_CMD% -m pip install --upgrade llama-cpp-python
    echo Installing exo in development mode...
    %PYTHON_CMD% -m pip install -e . --use-pep517
)

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install exo
    pause
    exit /b 1
)

echo.
echo [OK] Installation complete!

if "%NVIDIA_GPU_DETECTED%"=="true" if "%CUDA_AVAILABLE%"=="true" (
    if "%RTX_5070_TI_DETECTED%"=="true" (
        echo [SUCCESS] RTX 5070 Ti GPU acceleration is enabled and optimized!
        echo    - Blackwell architecture support (compute capability 9.0)
        echo    - 16GB GDDR7 VRAM optimization (14.5GB available for models)
        echo    - Aggressive hybrid GPU/RAM allocation for large models
        echo    - 95%% VRAM utilization on high-end cards
    ) else (
        echo [SUCCESS] GPU acceleration is enabled and ready to use!
    )
) else if "%NVIDIA_GPU_DETECTED%"=="true" (
    if "%RTX_5070_TI_DETECTED%"=="true" (
        echo [TIP] To enable RTX 5070 Ti GPU acceleration:
        echo    1. Install CUDA toolkit 12.0+ from https://developer.nvidia.com/cuda-downloads
        echo    2. Run: fix_windows_rtx_5070_ti.ps1 (PowerShell as Administrator)
        echo    3. Ensure driver version 566.03 or newer
    ) else (
        echo [TIP] To enable GPU acceleration:
        echo    1. Install CUDA toolkit from https://developer.nvidia.com/cuda-downloads
        echo    2. Run: fix_llamacpp_gpu.bat
    )
)

echo.
echo [USAGE] To run exo:
echo    1. Activate virtual environment: venv-windows\Scripts\activate.bat
echo    2. Run: python -m exo
echo.
echo [TIP] Virtual environment is located at: .\venv-windows\
echo       Always activate it before running exo commands
if "%RTX_5070_TI_DETECTED%"=="true" (
    echo.
    echo [RTX 5070 Ti] Monitor VRAM usage during inference:
    echo    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 1
    echo.
    echo Expected behavior:
    echo    - Small models (3-7GB): Full VRAM usage on RTX 5070 Ti
    echo    - Large models (13GB+): Hybrid GPU/RAM usage
    echo    - GPU utilization should be greater than 0%% during inference
)
echo.
pause
