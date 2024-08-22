import unittest
from unittest.mock import patch
from exo.topology.device_capabilities import mac_device_capabilities, DeviceCapabilities, DeviceFlops, TFLOPS


class TestMacDeviceCapabilities(unittest.TestCase):
  @patch("subprocess.check_output")
  def test_mac_device_capabilities_pro(self, mock_check_output):
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
    result = mac_device_capabilities()

    # Check the results
    self.assertIsInstance(result, DeviceCapabilities)
    self.assertEqual(result.model, "MacBook Pro")
    self.assertEqual(result.chip, "Apple M3 Max")
    self.assertEqual(result.memory, 131072)  # 16 GB in MB
    self.assertEqual(
      str(result),
      "Model: MacBook Pro. Chip: Apple M3 Max. Memory: 131072MB. Flops: 14.20 TFLOPS, fp16: 28.40 TFLOPS, int8: 56.80 TFLOPS",
    )

  @patch("subprocess.check_output")
  def test_mac_device_capabilities_air(self, mock_check_output):
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
    result = mac_device_capabilities()

    # Check the results
    self.assertIsInstance(result, DeviceCapabilities)
    self.assertEqual(result.model, "MacBook Air")
    self.assertEqual(result.chip, "Apple M2")
    self.assertEqual(result.memory, 8192)  # 8 GB in MB

  @unittest.skip("Unskip this test when running on a MacBook Pro, Apple M3 Max, 128GB")
  def test_mac_device_capabilities_real(self):
    # Call the function without mocking
    result = mac_device_capabilities()

    # Check the results
    self.assertIsInstance(result, DeviceCapabilities)
    self.assertEqual(result.model, "MacBook Pro")
    self.assertEqual(result.chip, "Apple M3 Max")
    self.assertEqual(result.memory, 131072)  # 128 GB in MB
    self.assertEqual(result.flops, DeviceFlops(fp32=14.20*TFLOPS, fp16=28.40*TFLOPS, int8=56.80*TFLOPS))
    self.assertEqual(
      str(result),
      "Model: MacBook Pro. Chip: Apple M3 Max. Memory: 131072MB. Flops: 14.20 TFLOPS, fp16: 28.40 TFLOPS, int8: 56.80 TFLOPS",
    )


if __name__ == "__main__":
  unittest.main()
