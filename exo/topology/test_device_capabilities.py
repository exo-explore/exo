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
