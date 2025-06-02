import pytest
from unittest.mock import patch
from exo.topology.device_capabilities import mac_device_capabilities, DeviceCapabilities, DeviceFlops, TFLOPS, device_capabilities


@pytest.mark.asyncio
@patch("subprocess.check_output")
async def test_mac_device_capabilities_pro(mock_check_output):
    # Mock the subprocess output
    mock_check_output.return_value = b"""
Hardware:

Hardware Overview:

Model Name: MacBook Pro
Model Identifier: Mac15,9
Model Number: Z1CM000EFB/A
Chip: Apple M3 Max
Total Number of Cores: 16 (12 performance and 4 efficiency)
Memory: 128 GB
System Firmware Version: 10000.000.0
OS Loader Version: 10000.000.0
Serial Number (system): XXXXXXXXXX
Hardware UUID: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
Provisioning UDID: XXXXXXXX-XXXXXXXXXXXXXXXX
Activation Lock Status: Enabled
"""

    # Call the function
    result = await mac_device_capabilities()

    # Check the results
    assert isinstance(result, DeviceCapabilities)
    assert result.model == "MacBook Pro"
    assert result.chip == "Apple M3 Max"
    assert result.memory == 131072  # 128 GB in MB
    assert str(result) == "Model: MacBook Pro. Chip: Apple M3 Max. Memory: 131072MB. Flops: 14.20 TFLOPS, fp16: 28.40 TFLOPS, int8: 56.80 TFLOPS"


@pytest.mark.asyncio
@patch("subprocess.check_output")
async def test_mac_device_capabilities_air(mock_check_output):
    # Mock the subprocess output
    mock_check_output.return_value = b"""
Hardware:

Hardware Overview:

Model Name: MacBook Air
Model Identifier: Mac14,2
Model Number: MLY33B/A
Chip: Apple M2
Total Number of Cores: 8 (4 performance and 4 efficiency)
Memory: 8 GB
System Firmware Version: 10000.00.0
OS Loader Version: 10000.00.0
Serial Number (system): XXXXXXXXXX
Hardware UUID: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
Provisioning UDID: XXXXXXXX-XXXXXXXXXXXXXXXX
Activation Lock Status: Disabled
"""

    # Call the function
    result = await mac_device_capabilities()

    # Check the results
    assert isinstance(result, DeviceCapabilities)
    assert result.model == "MacBook Air"
    assert result.chip == "Apple M2"
    assert result.memory == 8192  # 8 GB in MB


@pytest.mark.skip(reason="Unskip this test when running on a MacBook Pro, Apple M3 Max, 128GB")
@pytest.mark.asyncio
async def test_mac_device_capabilities_real():
    # Call the function without mocking
    result = await mac_device_capabilities()

    # Check the results
    assert isinstance(result, DeviceCapabilities)
    assert result.model == "MacBook Pro"
    assert result.chip == "Apple M3 Max"
    assert result.memory == 131072  # 128 GB in MB
    assert result.flops == DeviceFlops(fp32=14.20*TFLOPS, fp16=28.40*TFLOPS, int8=56.80*TFLOPS)
    assert str(result) == "Model: MacBook Pro. Chip: Apple M3 Max. Memory: 131072MB. Flops: 14.20 TFLOPS, fp16: 28.40 TFLOPS, int8: 56.80 TFLOPS"


@pytest.mark.asyncio
async def test_device_capabilities():
    caps = await device_capabilities()
    assert caps.model != ""
    assert caps.chip != ""
    assert caps.memory > 0
    assert caps.flops is not None



import unittest # Add unittest for IsolatedAsyncioTestCase if not fully pytest style
from unittest.mock import patch, MagicMock
import platform # For mocking platform calls
import sys # Added sys for sys.modules manipulation in mocks

# Imports already in device_capabilities.py that we might need to mock or use:
# import psutil
# import pynvml # (mocked)
# import pyamdgpuinfo # (mocked)
# from pyrsmi import rocml # (mocked)

from exo.topology.device_capabilities import (
    get_cpu_arch_for_chip_info,
    DeviceCapabilities,
    DeviceFlops,
    TFLOPS,
    mac_device_capabilities,
    linux_device_capabilities,
    windows_device_capabilities,
    device_capabilities as generic_device_capabilities # Alias to avoid conflict
)

# Need to handle imports for psutil, pynvml etc. for mocking.
# These will be mocked at the point of use.

class TestGetCpuArch(unittest.TestCase):
    @patch('platform.processor')
    @patch('platform.machine')
    def test_get_cpu_arch_intel(self, mock_machine, mock_processor):
        mock_processor.return_value = "Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz"
        mock_machine.return_value = "x86_64"
        self.assertEqual(get_cpu_arch_for_chip_info(), "Intel CPU")

    @patch('platform.processor')
    @patch('platform.machine')
    def test_get_cpu_arch_amd(self, mock_machine, mock_processor):
        mock_processor.return_value = "AMD Ryzen 9 5950X 16-Core Processor"
        mock_machine.return_value = "x86_64"
        self.assertEqual(get_cpu_arch_for_chip_info(), "AMD CPU")

    @patch('platform.processor')
    @patch('platform.machine')
    def test_get_cpu_arch_arm(self, mock_machine, mock_processor):
        mock_processor.return_value = "" # Processor string might be empty for some ARM
        mock_machine.return_value = "aarch64"
        self.assertEqual(get_cpu_arch_for_chip_info(), "ARM CPU")

    @patch('platform.processor')
    @patch('platform.machine')
    def test_get_cpu_arch_x86_generic(self, mock_machine, mock_processor):
        mock_processor.return_value = "Genuine Processzor x86" # Not Intel or AMD
        mock_machine.return_value = "x86_64"
        self.assertEqual(get_cpu_arch_for_chip_info(), "x86_64 CPU")

    @patch('platform.processor', side_effect=Exception("Test Exception"))
    @patch('platform.machine', side_effect=Exception("Test Exception"))
    def test_get_cpu_arch_exception_fallback(self, mock_machine, mock_processor):
        self.assertEqual(get_cpu_arch_for_chip_info(), "Unknown CPU")


class TestAvailableMemoryMac(unittest.IsolatedAsyncioTestCase):
    @patch('exo.topology.device_capabilities.get_mac_system_info')
    @patch('psutil.virtual_memory')
    async def test_mac_available_memory(self, mock_psutil_vm, mock_get_mac_info):
        mock_get_mac_info.return_value = ("MacBook Pro", "Apple M1", 16 * 1024) # name, chip, total_mem_MB

        # Mock psutil.virtual_memory to return an object with 'available' and 'total'
        mock_vm_stats = MagicMock()
        mock_vm_stats.available = 8 * 1024 * 1024 * 1024  # 8 GB available in bytes
        mock_vm_stats.total = 16 * 1024 * 1024 * 1024 # 16 GB total in bytes
        mock_psutil_vm.return_value = mock_vm_stats

        # Patch subprocess for get_mac_system_info if it's still used there, though we mocked get_mac_system_info itself
        # Also ensure psutil is "imported" for the context of the device_capabilities module
        with patch.dict(sys.modules, {'psutil': sys.modules['psutil'] if 'psutil' in sys.modules else MagicMock(virtual_memory=mock_psutil_vm)}):
             with patch('subprocess.check_output'): # In case get_mac_system_info is not fully mocked for some reason
                caps = await mac_device_capabilities()

        self.assertEqual(caps.memory, 16 * 1024)
        self.assertEqual(caps.available_memory, 8 * 1024) # Should be in MB

class TestAvailableMemoryLinux(unittest.IsolatedAsyncioTestCase):
    @patch('exo.topology.device_capabilities.Device') # from tinygrad
    @patch('exo.topology.device_capabilities.pynvml', create=True) # Mock the pynvml module itself in target namespace
    async def test_linux_nvidia_available_memory(self, mock_pynvml, mock_tinygrad_device):
        mock_tinygrad_device.DEFAULT = "CUDA" # Simulate NVIDIA detection by Tinygrad

        mock_pynvml.nvmlInit.return_value = None
        mock_handle_instance = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle_instance
        mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA GeForce RTX 3090"

        mock_mem_stats = MagicMock()
        mock_mem_stats.total = 24 * 1024 * 1024 * 1024 # 24GB
        mock_mem_stats.free = 10 * 1024 * 1024 * 1024  # 10GB
        mock_mem_stats.used = 14 * 1024 * 1024 * 1024 # 14GB
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem_stats
        mock_pynvml.nvmlShutdown.return_value = None

        caps = await linux_device_capabilities()

        self.assertEqual(caps.chip, "NVIDIA GEFORCE RTX 3090")
        self.assertEqual(caps.memory, 24 * 1024) # MB
        self.assertEqual(caps.available_memory, 10 * 1024) # MB
        mock_pynvml.nvmlInit.assert_called_once()
        mock_pynvml.nvmlShutdown.assert_called_once()

    @patch('exo.topology.device_capabilities.Device') # from tinygrad
    @patch('exo.topology.device_capabilities.pyamdgpuinfo', create=True) # Mock module in target namespace
    async def test_linux_amd_available_memory(self, mock_pyamdgpuinfo, mock_tinygrad_device):
        mock_tinygrad_device.DEFAULT = "AMD"

        mock_gpu_info = MagicMock()
        mock_gpu_info.name = "AMD Radeon RX 6800 XT"
        mock_gpu_info.memory_info = {
            "vram_size": 16 * 1024 * 1024 * 1024, # 16 GB
            "vram_used": 6 * 1024 * 1024 * 1024   # 6 GB
        }
        mock_pyamdgpuinfo.get_gpu.return_value = mock_gpu_info

        caps = await linux_device_capabilities()
        self.assertEqual(caps.chip, "AMD Radeon RX 6800 XT")
        self.assertEqual(caps.memory, 16 * 1024) # MB
        self.assertEqual(caps.available_memory, (16-6) * 1024) # MB

    @patch('exo.topology.device_capabilities.Device') # from tinygrad
    @patch('psutil.virtual_memory') # Mock psutil directly in the target module's scope
    @patch('exo.topology.device_capabilities.get_cpu_arch_for_chip_info') # Mock our helper
    async def test_linux_cpu_fallback_available_memory(self, mock_get_cpu_info, mock_psutil_vm, mock_tinygrad_device):
        mock_tinygrad_device.DEFAULT = "CPU" # Tinygrad on CPU
        mock_get_cpu_info.return_value = "Intel CPU"

        mock_vm_stats = MagicMock()
        mock_vm_stats.total = 32 * 1024 * 1024 * 1024 # 32 GB
        mock_vm_stats.available = 15 * 1024 * 1024 * 1024 # 15 GB
        mock_psutil_vm.return_value = mock_vm_stats

        # Ensure that device_capabilities.psutil refers to our mock
        with patch.dict(sys.modules, {'psutil': MagicMock(virtual_memory=mock_psutil_vm)}):
            caps = await linux_device_capabilities()

        self.assertTrue("Intel CPU" in caps.chip)
        self.assertTrue("(Tinygrad Device: CPU)" in caps.chip)
        self.assertEqual(caps.memory, 32 * 1024)
        self.assertEqual(caps.available_memory, 15 * 1024)

class TestAvailableMemoryGeneric(unittest.IsolatedAsyncioTestCase):
    @patch('psutil.MACOS', False)
    @patch('psutil.LINUX', False)
    @patch('psutil.WINDOWS', False) # Simulate unknown OS
    @patch('psutil.virtual_memory')
    @patch('exo.topology.device_capabilities.get_cpu_arch_for_chip_info')
    async def test_generic_fallback_available_memory(self, mock_get_cpu_info, mock_psutil_vm, mock_win, mock_lin, mock_mac):
        mock_get_cpu_info.return_value = "Some Weird CPU"
        mock_vm_stats = MagicMock()
        mock_vm_stats.total = 8 * 1024 * 1024 * 1024 # 8 GB
        mock_vm_stats.available = 3 * 1024 * 1024 * 1024 # 3 GB
        mock_psutil_vm.return_value = mock_vm_stats

        # Ensure that device_capabilities.psutil refers to our mock
        with patch.dict(sys.modules, {'psutil': MagicMock(virtual_memory=mock_psutil_vm, MACOS=False, LINUX=False, WINDOWS=False)}):
            caps = await generic_device_capabilities() # Test the main fallback in device_capabilities()

        self.assertEqual(caps.chip, "Some Weird CPU")
        self.assertEqual(caps.memory, 8 * 1024)
        self.assertEqual(caps.available_memory, 3 * 1024)

# Note: Tests for Windows capabilities would follow a similar pattern,
# mocking win32com, pynvml (for Windows NVIDIA), pyrsmi.rocml (for Windows AMD).
# Due to complexity of mocking COM objects (win32com) and for brevity,
# those are omitted here but would be needed for full coverage.
