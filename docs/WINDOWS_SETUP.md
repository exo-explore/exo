# Exo Windows Setup Guide

This guide covers installing and running exo on Windows systems. **Windows support is now fully functional!** ✅

## Prerequisites

### Required Software
1. **Python 3.12+**: Download from [python.org](https://www.python.org/downloads/)
   - ✅ Make sure to check "Add Python to PATH" during installation
   - ✅ Verify installation: `python --version`

2. **Git**: Download from [git-scm.com](https://git-scm.com/download/win)
   - ✅ Required for cloning the repository

### Optional Software (for GPU acceleration)
3. **NVIDIA CUDA Toolkit** (for NVIDIA GPUs):
   - Download from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
   - ✅ Verify installation: `nvidia-smi`

4. **Visual Studio Build Tools** (for some packages):
   - Download from [Microsoft](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - ✅ Select "C++ build tools" workload

## ✅ Supported Features on Windows

- **All Inference Engines**: MLX (via compatibility layer), TinyGrad, LlamaCpp, Dummy
- **Networking**: UDP discovery, manual discovery, GRPC peer networking
- **Device Detection**: Automatic GPU detection and capability assessment
- **Model Download**: Hugging Face integration with caching
- **Web Interface**: TinyChat web UI and ChatGPT-compatible API
- **Cross-platform Clustering**: Windows nodes can join Linux/macOS clusters seamlessly

## Installation

### Method 1: PowerShell (Recommended)
```powershell
# Clone the repository
git clone https://github.com/exo-explore/exo.git
cd exo

# Run the PowerShell installation script
.\install.ps1
```

### Method 2: Command Prompt
```cmd
# Clone the repository
git clone https://github.com/exo-explore/exo.git
cd exo

# Run the batch installation script
install.bat
```

### Method 3: Manual Installation
```powershell
# Clone the repository
git clone https://github.com/exo-explore/exo.git
cd exo

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install exo
pip install -e .
```

## Configuration

### Firewall Configuration
Windows Firewall may block network discovery. To allow exo:

1. **Automatic** (when first running exo):
   - Click "Allow access" when Windows prompts

2. **Manual**:
   - Go to Windows Defender Firewall settings
   - Click "Allow an app or feature through Windows Defender Firewall"
   - Click "Change Settings" → "Allow another app"
   - Browse to your Python executable in `.venv\Scripts\python.exe`
   - Allow for both Private and Public networks

### GPU Detection
Exo automatically detects available GPUs:

- **NVIDIA GPUs**: Requires CUDA toolkit installation
- **AMD GPUs**: Limited Windows support (CPU fallback recommended)
- **Intel GPUs**: Currently uses CPU inference

## Running Exo

### Starting the Service
```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Start exo
exo
```

### Example API Usage
```powershell
# Test the API with PowerShell
Invoke-RestMethod -Uri "http://localhost:52415/v1/chat/completions" `
  -Method POST `
  -ContentType "application/json" `
  -Body (@{
    model = "llama-3.1-8b"
    messages = @(@{ role = "user"; content = "Hello, world!" })
    temperature = 0.7
  } | ConvertTo-Json -Depth 10)
```

## Network Discovery

### UDP Discovery (Default)
- Works on most Windows systems
- May require firewall configuration
- Automatically discovers other exo nodes on the local network

### Manual Discovery (Recommended for Windows)
Create a config file for manual peer discovery:

```json
{
  "peers": {
    "node1": {
      "address": "192.168.1.100",
      "port": 50051,
      "device_capabilities": {
        "model": "Windows PC",
        "chip": "NVIDIA RTX 4080",
        "memory": 16384,
        "flops": {"fp32": 83.0, "fp16": 166.0, "int8": 332.0}
      }
    }
  }
}
```

Use manual discovery:
```powershell
exo --discovery-module manual --discovery-config-path peers.json
```

### Tailscale Discovery
For cross-platform networking:

1. Install [Tailscale](https://tailscale.com/download/windows)
2. Get API key from Tailscale admin console
3. Run exo with Tailscale discovery:
```powershell
exo --discovery-module tailscale --tailscale-api-key YOUR_API_KEY
```

## Troubleshooting

### Common Issues

**"Python not found"**
- Ensure Python is installed and in PATH
- Restart PowerShell/Command Prompt after Python installation

**"ModuleNotFoundError: No module named 'uvloop'"**
- This is normal on Windows - uvloop is Unix-only
- Exo automatically falls back to standard asyncio

**"Permission denied" during installation**
- Run PowerShell as Administrator
- Or use: `pip install -e . --user`

**GPU not detected**
- For NVIDIA: Ensure CUDA toolkit is installed
- For AMD: Consider using CPU inference on Windows
- Check GPU with: `nvidia-smi` (NVIDIA) or Device Manager

**Network discovery not working**
- Check Windows Firewall settings
- Try manual discovery instead
- Ensure all nodes are on the same network

**Performance issues**
- Close unnecessary applications
- Use GPU inference when available
- Consider upgrading RAM for larger models

### Debug Mode
Enable debug output for troubleshooting:
```powershell
$env:DEBUG = "1"
exo
```

### Log Files
Check the console output for error messages. For persistent logging:
```powershell
exo 2>&1 | Tee-Object -FilePath "exo.log"
```

## ✅ Testing Your Installation

### Quick Test
```powershell
# Activate virtual environment
.\venv-windows\Scripts\Activate.ps1

# Test basic functionality
python -m exo.main --help

# Test specific inference engine
python -m exo.main --inference-engine llamacpp --disable-tui
```

### Full Functionality Test
```powershell
# Start exo with web interface
python -m exo.main --inference-engine llamacpp

# In another terminal, test the API
curl http://localhost:52415/v1/chat/completions -H "Content-Type: application/json" -d '{\"model\": \"llama-3.2-1b\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'
```

### Verify Components
- **Import Test**: `python -c "from exo.main import main; print('✅ Exo imports successfully')"`
- **LlamaCpp Test**: `python -c "from llama_cpp import Llama; print('✅ LlamaCpp available')"`
- **Device Detection**: Check that your GPU is detected in the web interface

## Platform-Specific Notes

### Windows vs. Unix Differences
- **Event Loop**: Windows uses standard asyncio (uvloop unavailable)
- **Signals**: Different signal handling implementation
- **File Paths**: Uses Windows path separators
- **Terminal**: PowerShell/Command Prompt instead of bash
- **Virtual Environment**: Uses `venv-windows` instead of `.venv`

### Known Windows-Specific Behaviors
- Some network interface warnings (normal, can be ignored)
- Hugging Face symlink warnings (performance impact only)
- Process cleanup may require manual termination

### Recommended Windows Setup
- Use manual or Tailscale discovery for reliability
- Prefer local inference for single-machine setups
- Use Windows Terminal for better console experience
- Consider WSL2 for Unix-like environment if needed

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/exo-explore/exo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/exo-explore/exo/discussions)
- **Discord**: Check README for community links

## Performance Tips

1. **Memory**: Ensure sufficient RAM for your models
2. **Storage**: Use SSD for faster model loading
3. **Network**: Use wired connections for better peer communication
4. **GPU**: Install latest drivers for optimal performance
5. **Antivirus**: Add exo directory to exclusions for better performance
