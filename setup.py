import sys
import platform
import subprocess
import os
import atexit

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install

# Base requirements for all platforms
install_requires = [
  "aiohttp==3.10.11",
  "aiohttp_cors==0.7.0",
  "aiofiles==24.1.0",
  "grpcio==1.70.0",
  "grpcio-tools==1.70.0",
  "Jinja2==3.1.4",
  "llama-cpp-python>=0.2.90",
  "numpy==2.0.0",
  "nuitka==2.5.1",
  "nvidia-ml-py==12.560.30",
  "opencv-python==4.10.0.84",
  "pillow==10.4.0",
  "prometheus-client==0.20.0",
  "protobuf==5.28.1",
  "psutil==6.0.0",
  "pyamdgpuinfo==2.1.6;platform_system=='Linux'",
  "pydantic==2.9.2",
  "requests==2.32.3",
  "rich==13.7.1",
  "scapy==2.6.1",
  "torch>=2.0.0",  # PyTorch for GPU detection and optimization
  "tqdm==4.66.4",  "transformers==4.46.3",
  "uuid==1.30",
  "uvloop==0.21.0;sys_platform!='win32'",  # uvloop doesn't support Windows
  "tinygrad @ git+https://github.com/tinygrad/tinygrad.git@ec120ce6b9ce8e4ff4b5692566a683ef240e8bc8",
]

extras_require = {
  "formatting": ["yapf==0.40.2",],
  "apple_silicon": [
    "mlx==0.22.0",
    "mlx-lm==0.21.1",
  ],
  "windows": ["pywin32==308",],
  "nvidia-gpu": [
    "nvidia-ml-py==12.560.30",
  ],
  "rtx-50-series": [
    "torch>=2.6.0",
    "nvidia-ml-py==12.560.30",
    "llama-cpp-python>=0.2.90",
  ],
  "amd-gpu": ["pyrsmi==0.2.0"],
  "pytorch": ["torch>=2.0.0"],
  "pytorch-cuda": ["torch>=2.0.0"],
  "llamacpp": ["llama-cpp-python>=0.2.90"],
  "llamacpp-cuda": ["llama-cpp-python[cuda]>=0.2.90"],
  "llamacpp-metal": ["llama-cpp-python[metal]>=0.2.90"],
}

# Check if running on macOS with Apple Silicon
if sys.platform.startswith("darwin") and platform.machine() == "arm64":
  install_requires.extend(extras_require["apple_silicon"])

# Check if running Windows
if sys.platform.startswith("win32"):
  install_requires.extend(extras_require["windows"])


def _find_cuda_path():
  """Find CUDA installation path on both Windows and Linux"""
  cuda_paths = []
  
  if sys.platform.startswith("win"):
    # Windows CUDA paths
    cuda_paths = [
      r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
      r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA",
      os.environ.get('CUDA_PATH', ''),
    ]
  else:
    # Linux/macOS CUDA paths
    cuda_paths = [
      "/usr/local/cuda",
      "/opt/cuda",
      "/usr/cuda",
    ]
    # Add versioned CUDA paths
    import glob
    cuda_paths.extend(glob.glob("/usr/local/cuda-*"))
  
  for cuda_path in cuda_paths:
    if cuda_path and os.path.exists(cuda_path):
      if sys.platform.startswith("win"):
        nvcc_path = os.path.join(cuda_path, "bin", "nvcc.exe")
      else:
        nvcc_path = os.path.join(cuda_path, "bin", "nvcc")
      
      if os.path.exists(nvcc_path):
        return cuda_path
  
  return None

def _setup_cuda_environment():
  """Setup CUDA environment variables for compilation"""
  cuda_path = _find_cuda_path()
  
  if cuda_path:
    print(f"Found CUDA at: {cuda_path}")
    
    if sys.platform.startswith("win"):
      # Windows CUDA setup
      cuda_bin = os.path.join(cuda_path, "bin")
      cuda_lib = os.path.join(cuda_path, "lib", "x64")
      
      # Add to PATH
      current_path = os.environ.get('PATH', '')
      if cuda_bin not in current_path:
        os.environ['PATH'] = cuda_bin + os.pathsep + current_path
      
      # Set CUDA environment variables
      os.environ['CUDA_PATH'] = cuda_path
      os.environ['CUDA_HOME'] = cuda_path
      os.environ['CUDACXX'] = os.path.join(cuda_bin, "nvcc.exe")
      
    else:
      # Linux/macOS CUDA setup
      cuda_bin = os.path.join(cuda_path, "bin")
      cuda_lib = os.path.join(cuda_path, "lib64")
      
      # Add to PATH
      current_path = os.environ.get('PATH', '')
      if cuda_bin not in current_path:
        os.environ['PATH'] = cuda_bin + os.pathsep + current_path
      
      # Add to LD_LIBRARY_PATH
      current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
      if cuda_lib not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = cuda_lib + os.pathsep + current_ld_path
      
      # Set CUDA environment variables
      os.environ['CUDA_HOME'] = cuda_path
      os.environ['CUDACXX'] = os.path.join(cuda_bin, "nvcc")
    
    return True
  else:
    print("CUDA installation not found!")
    if sys.platform.startswith("win"):
      print("Please install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads")
    else:
      print("Please install CUDA Toolkit: sudo apt install nvidia-cuda-toolkit")
    return False

def _detect_rtx_50_series():
  """Detect RTX 50 series cards and check PyTorch compatibility"""
  try:
    # Check via nvidia-smi first
    out = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                        shell=True, text=True, capture_output=True, check=False)
    if out.returncode == 0:
      gpu_names = out.stdout.strip().split('\n')
      for gpu_name in gpu_names:
        if any(model in gpu_name.upper() for model in ['RTX 50', '5070', '5080', '5090']):
          print(f"RTX 50 series detected: {gpu_name}")
          
          # Check PyTorch version compatibility
          try:
            import torch
            version_parts = torch.__version__.split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            
            if major >= 2 and minor >= 6:
              print(f"PyTorch {torch.__version__} supports RTX 50 series")
            else:
              print(f"WARNING: PyTorch {torch.__version__} may not fully support RTX 50 series")
              print("   Recommend: pip install torch>=2.6 --index-url https://download.pytorch.org/whl/cu124")
          except ImportError:
            print("WARNING: PyTorch not installed - will install compatible version")
          
          return True
          
  except Exception as e:
    print(f"RTX 50 series detection failed: {e}")
  
  return False

def _add_gpu_requires():
  global install_requires
  
  # Detect RTX 50 series for special handling
  rtx_50_detected = _detect_rtx_50_series()
  
  # Add Nvidia-GPU with CUDA-enabled llama-cpp-python
  try:
    out = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], shell=True, text=True, capture_output=True, check=False)
    if out.returncode == 0:
      print("NVIDIA GPU detected - adding PyTorch and GPU support")
      install_requires.extend(extras_require["nvidia-gpu"])
      
      # For RTX 50 series, ensure PyTorch 2.6+ is installed
      if rtx_50_detected:
        print("Configuring RTX 50 series optimization...")
        # Add RTX 50 series specific requirements
        rtx_50_requirements = extras_require["rtx-50-series"]
        
        # Check if PyTorch 2.6+ is already installed AND has CUDA support
        torch_needs_upgrade = False
        pytorch_has_cuda = False
        
        try:
          import torch
          version_parts = torch.__version__.split('.')
          major, minor = int(version_parts[0]), int(version_parts[1])
          
          # Check version compatibility
          if not (major >= 2 and minor >= 6):
            torch_needs_upgrade = True
            print(f"PyTorch {torch.__version__} too old for RTX 50 series")
          
          # Check if it has CUDA support
          pytorch_has_cuda = torch.cuda.is_available()
          if not pytorch_has_cuda:
            torch_needs_upgrade = True
            print(f"PyTorch {torch.__version__} is CPU-only, need CUDA version")
          
        except ImportError:
          torch_needs_upgrade = True
          print("PyTorch not installed")
        
        if torch_needs_upgrade:
          print("Need to install PyTorch 2.6+ with CUDA for RTX 50 series")
          # Remove any existing torch requirements
          install_requires = [req for req in install_requires if not req.startswith("torch")]
          
          # For RTX 50 series, we need to force CUDA installation
          # We'll handle this in a custom installation step
          print("RTX 50 series detected - will install PyTorch with CUDA during setup")
        else:
          print("PyTorch 2.6+ with CUDA already installed")
      
      # Check if CMAKE_ARGS is set for CUDA (indicating install script is being used)
      cmake_args = os.environ.get('CMAKE_ARGS', '')
      
      if 'DGGML_CUDA=on' in cmake_args:
        # Install script is handling CUDA compilation - remove llama-cpp-python from here
        install_requires = [req for req in install_requires if not req.startswith("llama-cpp-python")]
        print("NVIDIA GPU detected! CUDA compilation enabled via CMAKE_ARGS")
        if rtx_50_detected:
          print("RTX 50 series CUDA optimization enabled")
      else:
        # Auto-enable CUDA compilation for pip install -e .
        print("NVIDIA GPU detected! Automatically enabling CUDA support...")
        if rtx_50_detected:
          print("RTX 50 series detected - enabling optimized CUDA compilation")
        
        # Setup CUDA environment
        if _setup_cuda_environment():
          # Enhanced CMAKE_ARGS for RTX 50 series
          if rtx_50_detected:
            # RTX 5070 Ti specific settings (Blackwell architecture) - same as PowerShell script
            cuda_args = '-DGGML_CUDA=on -DCUDA_ARCHITECTURES=90 -DGGML_CUDA_FORCE_DMMV=ON -DGGML_CUDA_DMMV_F16=ON'
            print("RTX 5070 Ti: Enabling Blackwell architecture optimizations")
            print("CMAKE_ARGS: RTX 5070 Ti specific compilation flags")
            
            # Windows-specific CUDA setup for RTX 5070 Ti
            if sys.platform.startswith("win"):
              cuda_path = _find_cuda_path()
              if cuda_path:
                os.environ['CUDACXX'] = os.path.join(cuda_path, "bin", "nvcc.exe")
                print(f"CUDACXX set to: {os.environ['CUDACXX']}")
          else:
            # Standard CUDA args for other GPUs
            cuda_args = '-DGGML_CUDA=on -DGGML_CUDA_BLACKWELL=on -DCUDA_ARCHITECTURES="90;89;86;80;75"'
          
          os.environ['CMAKE_ARGS'] = cuda_args
          os.environ['FORCE_CMAKE'] = '1'
          
          # Remove standard llama-cpp-python and let CMAKE_ARGS handle the compilation
          install_requires = [req for req in install_requires if not req.startswith("llama-cpp-python")]
          
          # Add llama-cpp-python without version constraint to force recompilation
          install_requires.append("llama-cpp-python")
          
          print(f"CMAKE_ARGS set to: {cuda_args}")
          if rtx_50_detected:
            print("This will compile llama-cpp-python with RTX 5070 Ti optimizations.")
            print("Compilation may take 10-15 minutes...")
          else:
            print("This will compile llama-cpp-python with optimized CUDA support.")
        else:
          print("WARNING: CUDA not properly configured. Installing CPU-only version.")
          print("For GPU acceleration, please install CUDA Toolkit and restart.")
        
  except subprocess.CalledProcessError:
    # If nvidia-smi fails, check if user wants PyTorch anyway
    if os.environ.get('INSTALL_PYTORCH', '').lower() in ['1', 'true', 'yes']:
      print("INSTALL_PYTORCH=1 detected - adding PyTorch support")
      install_requires.extend(extras_require["pytorch"])
  # Add AMD-GPU
  # This will mostly work only on Linux, amd/rocm-smi is not yet supported on Windows
  try:
    out = subprocess.run(['amd-smi', 'list', '--csv'], shell=True, text=True, capture_output=True, check=False)
    if out.returncode == 0:
      install_requires.extend(extras_require["amd-gpu"])
  except:
    try:
      out = subprocess.run(['rocm-smi', 'list', '--csv'], shell=True, text=True, capture_output=True, check=False)
      if out.returncode == 0:
        install_requires.extend(extras_require["amd-gpu"])
    except:
      pass


def _run_windows_gpu_diagnostic():
  """Run Windows-specific GPU diagnostics with RTX 5070 Ti support"""
  if not sys.platform.startswith("win"):
    return True
  
  try:
    print("Running Windows GPU diagnostics...")
    
    # Check NVIDIA driver and CUDA via nvidia-smi
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,compute_cap', '--format=csv,noheader'], 
                          shell=True, text=True, capture_output=True, check=False)
    if result.returncode == 0:
      gpu_info = result.stdout.strip().split('\n')[0]  # First GPU
      print(f"GPU detected: {gpu_info}")
      
      # Check for RTX 5070 Ti specifically
      if "5070" in gpu_info:
        print("RTX 5070 Ti detected!")
        
        # Check compute capability (should be 9.0 for Blackwell)
        if "9.0" in gpu_info:
          print("Compute capability 9.0 (Blackwell architecture) confirmed")
        else:
          print("WARNING: Compute capability may not be 9.0")
      
      # Check driver requirements for latest GPUs
      try:
        driver_version = gpu_info.split(', ')[2]  # Driver version is 3rd field
        major_version = int(driver_version.split('.')[0])
        if major_version >= 566:
          print("Driver supports RTX 5070 Ti (Blackwell architecture)")
        elif major_version >= 560:
          print("Driver may support RTX 5070 Ti but newer version recommended")
        else:
          print("WARNING: Driver may be too old for RTX 5070 Ti (need 566.03+)")
      except:
        print("Could not parse driver version for compatibility check")
        
      return True
    else:
      print("WARNING: nvidia-smi failed or no GPU detected")
      return False
      
  except Exception as e:
    print(f"Windows GPU diagnostic failed: {e}")
    return False

def run_post_install():
  """Run post-install RTX validation if RTX 50 series detected"""
  try:
    # Quick check for RTX 50 series
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                          shell=True, text=True, capture_output=True, check=False)
    if result.returncode == 0:
      gpu_names = result.stdout.strip().split('\n')
      for gpu_name in gpu_names:
        if any(model in gpu_name.upper() for model in ['RTX 50', '5070', '5080', '5090']):
          print(f"\nRTX 50 series detected: {gpu_name}")
          print("Running post-install validation...")
          
          # Run Windows-specific diagnostics
          if sys.platform.startswith("win"):
            _run_windows_gpu_diagnostic()
          
          # Test GPU support
          gpu_test_passed = _test_gpu_support()
          
          # Run post-install validation script if available
          try:
            post_install_script = os.path.join(os.path.dirname(__file__), "post_install_rtx_test.py")
            if os.path.exists(post_install_script):
              subprocess.run([sys.executable, post_install_script], check=False)
            else:
              print("Post-install validation script not found (optional)")
          except Exception as e:
            print(f"WARNING: Post-install validation failed: {e}")
          
          # Final status
          if gpu_test_passed:
            print("\n🎉 SUCCESS: RTX 5070 Ti GPU acceleration setup complete!")
            print("Your models should now load into VRAM instead of system RAM.")
            print("\nTo verify VRAM usage during inference:")
            print("  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 1")
          else:
            print("\n❌ WARNING: GPU acceleration may not be working properly")
            print("Try running the manual fix script: fix_windows_rtx_5070_ti.ps1")
          
          break
  except Exception:
    pass

def _test_gpu_support():
  """Test if GPU support is properly working"""
  try:
    print("Testing GPU support...")
    
    # Test llama-cpp-python GPU support
    from llama_cpp import llama_cpp
    if hasattr(llama_cpp, 'llama_supports_gpu_offload'):
      gpu_support = llama_cpp.llama_supports_gpu_offload()
      print(f"GPU offload support: {gpu_support}")
      
      if gpu_support:
        print("SUCCESS: GPU offload is available!")
        
        # Test device count if available
        if hasattr(llama_cpp, 'llama_get_device_count'):
          try:
            device_count = llama_cpp.llama_get_device_count()
            print(f"CUDA device count: {device_count}")
            if device_count > 0:
              print("SUCCESS: CUDA devices detected!")
              return True
          except Exception as e:
            print(f"Device count error: {e}")
        
        return gpu_support
      else:
        print("ERROR: GPU offload NOT available")
        return False
    else:
      print("ERROR: GPU offload function not available")
      return False
      
  except ImportError as e:
    print(f"ERROR: Cannot import llama_cpp: {e}")
    return False
  except Exception as e:
    print(f"ERROR: GPU test failed: {e}")
    return False

def install_pytorch_cuda():
  """Install PyTorch with CUDA support for RTX 50 series"""
  print("Installing PyTorch 2.6+ with CUDA support...")
  
  # Uninstall CPU version if present
  try:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"], 
                  check=False, capture_output=True)
  except:
    pass
  
  # Install CUDA version
  cmd = [
    sys.executable, "-m", "pip", "install", 
    "torch>=2.6", "torchvision", "torchaudio",
    "--index-url", "https://download.pytorch.org/whl/cu124"
  ]
  
  try:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print("SUCCESS: PyTorch with CUDA installed")
    
    # Test PyTorch CUDA support
    try:
      import torch
      if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        print(f"PyTorch CUDA devices: {device_count}")
        print(f"Primary GPU: {gpu_name}")
        return True
      else:
        print("WARNING: PyTorch installed but CUDA not available")
        return False
    except Exception as e:
      print(f"WARNING: PyTorch CUDA test failed: {e}")
      return False
    
  except subprocess.CalledProcessError as e:
    print(f"ERROR: PyTorch CUDA installation failed: {e}")
    return False

def install_llamacpp_cuda():
  """Install llama-cpp-python with CUDA support and RTX 5070 Ti optimizations"""
  print("Installing llama-cpp-python with CUDA support...")
  print("This may take 10-15 minutes to compile...")
  
  # Set environment variables for CUDA compilation
  env = os.environ.copy()
  
  # RTX 5070 Ti specific settings (Blackwell architecture)
  rtx_50_detected = _detect_rtx_50_series()
  if rtx_50_detected:
    print("RTX 5070 Ti detected - applying Blackwell architecture optimizations")
    env['CMAKE_ARGS'] = '-DGGML_CUDA=on -DCUDA_ARCHITECTURES=90 -DGGML_CUDA_FORCE_DMMV=ON -DGGML_CUDA_DMMV_F16=ON'
  else:
    env['CMAKE_ARGS'] = '-DGGML_CUDA=on -DGGML_CUDA_BLACKWELL=on -DCUDA_ARCHITECTURES="90;89;86;80;75"'
  
  env['FORCE_CMAKE'] = '1'
  
  # Find CUDA path and set it up properly for Windows
  cuda_path = _find_cuda_path()
  if cuda_path:
    env['CUDA_HOME'] = cuda_path
    env['CUDA_PATH'] = cuda_path
    
    if sys.platform.startswith("win"):
      # Windows-specific CUDA setup for RTX 5070 Ti
      env['CUDACXX'] = os.path.join(cuda_path, "bin", "nvcc.exe")
      
      # Add CUDA bin to PATH for Windows
      cuda_bin = os.path.join(cuda_path, "bin")
      current_path = env.get('PATH', '')
      if cuda_bin not in current_path:
        env['PATH'] = cuda_bin + os.pathsep + current_path
  else:
    print("WARNING: CUDA not found - GPU acceleration may not work")
  
  # Uninstall existing version
  try:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "llama-cpp-python", "-y"], 
                  check=False, capture_output=True)
  except:
    pass
  
  # Install with CUDA
  cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python", "--force-reinstall", "--no-cache-dir", "--verbose"]
  
  if rtx_50_detected:
    print(f"RTX 5070 Ti CMAKE_ARGS: {env['CMAKE_ARGS']}")
    print(f"RTX 5070 Ti CUDACXX: {env.get('CUDACXX', 'not set')}")
  
  try:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env, timeout=1800)  # 30 min timeout
    print("SUCCESS: llama-cpp-python with CUDA installed")
    
    # Test GPU support immediately after installation
    test_success = _test_gpu_support()
    if test_success:
      print("SUCCESS: GPU support verified!")
    else:
      print("WARNING: GPU support test failed")
    
    return True
  except subprocess.CalledProcessError as e:
    print(f"ERROR: llama-cpp-python CUDA installation failed: {e}")
    print("Trying pre-built CUDA version as fallback...")
    
    # Try pre-built version as fallback
    try:
      fallback_cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python[cuda]", "--force-reinstall", "--no-cache-dir"]
      subprocess.run(fallback_cmd, check=True, capture_output=True, text=True, timeout=600)
      print("SUCCESS: Pre-built CUDA version installed")
      return True
    except:
      print("ERROR: Both compilation and pre-built installation failed")
      return False
  except subprocess.TimeoutExpired:
    print("ERROR: llama-cpp-python installation timed out")
    return False

class PostDevelopCommand(develop):
  """Post-installation for development mode"""
  def run(self):
    develop.run(self)
    
    # Check if RTX 50 series and install CUDA versions
    if _detect_rtx_50_series():
      print("RTX 50 series detected - installing CUDA packages...")
      
      # Install PyTorch with CUDA
      pytorch_success = install_pytorch_cuda()
      
      # Install llama-cpp-python with CUDA
      llamacpp_success = install_llamacpp_cuda()
      
      if pytorch_success and llamacpp_success:
        print("SUCCESS: All CUDA packages installed successfully")
      else:
        print("WARNING: Some CUDA packages failed to install")
    
    run_post_install()

class PostInstallCommand(install):
  """Post-installation for installation mode"""
  def run(self):
    install.run(self)
    
    # Check if RTX 50 series and install CUDA versions
    if _detect_rtx_50_series():
      print("RTX 50 series detected - installing CUDA packages...")
      
      # Install PyTorch with CUDA
      pytorch_success = install_pytorch_cuda()
      
      # Install llama-cpp-python with CUDA
      llamacpp_success = install_llamacpp_cuda()
      
      if pytorch_success and llamacpp_success:
        print("SUCCESS: All CUDA packages installed successfully")
      else:
        print("WARNING: Some CUDA packages failed to install")
    
    run_post_install()

_add_gpu_requires()

setup(
  name="exo",
  version="0.0.1",
  packages=find_packages(),
  install_requires=install_requires,
  extras_require=extras_require,
  package_data={
    "exo": [
      "tinychat/**/*",
      "inference/llamacpp/**/*",
      "*.py",  # Include post-install scripts
    ]
  },
  entry_points={"console_scripts": ["exo = exo.main:run"]},
  cmdclass={
    'develop': PostDevelopCommand,
    'install': PostInstallCommand,
  },
)
