import unittest
from unittest.mock import patch
from exo.topology.device_capabilities import mac_device_capabilities, DeviceCapabilities, DeviceFlops, TFLOPS
import json

class TestMacDeviceCapabilities(unittest.TestCase):

    @patch('subprocess.check_output')
    def test_mac_device_capabilities_m3_max(self, mock_check_output):
        # Mock the subprocess output with JSON data
        mock_json_output = {
            "SPHardwareDataType": [
                {
                    "machine_model": "Mac15,9",
                    "machine_name": "MacBook Pro",
                    "cpu_type": "Apple M3 Max",
                    "number_processors": 16,
                    "physical_memory": "128 GB"
                }
            ]
        }
        mock_check_output.return_value = json.dumps(mock_json_output).encode('utf-8')

        # Call the function
        result = mac_device_capabilities()

        # Check the results
        self.assertIsInstance(result, DeviceCapabilities)
        self.assertEqual(result.model, "Mac15,9")
        self.assertEqual(result.chip, "Apple M3 Max")
        self.assertEqual(result.memory, 131072)  # 128 GB in MB
        self.assertEqual(result.flops, DeviceFlops(fp32=14.20*TFLOPS, fp16=28.40*TFLOPS, int8=56.80*TFLOPS))
        self.assertEqual(str(result), "Model: Mac15,9. Chip: Apple M3 Max. Memory: 131072MB. Flops: fp32: 14.20 TFLOPS, fp16: 28.40 TFLOPS, int8: 56.80 TFLOPS")


    @patch('subprocess.check_output')
    def test_mac_device_capabilities_m2(self, mock_check_output):
        # Mock the subprocess output with JSON data
        mock_json_output = {
            "SPHardwareDataType": [
                {
                    "machine_model": "Mac14,2",
                    "machine_name": "MacBook Air",
                    "chip_type": "Apple M2",  # Changed from cpu_type to chip_type
                    "number_processors": 8,
                    "physical_memory": "8 GB"
                }
            ]
        }
        mock_check_output.return_value = json.dumps(mock_json_output).encode('utf-8')

        # Call the function
        result = mac_device_capabilities()

        # Check the results
        self.assertIsInstance(result, DeviceCapabilities)
        self.assertEqual(result.model, "Mac14,2")
        self.assertEqual(result.chip, "Apple M2")
        self.assertEqual(result.memory, 8192)  # 8 GB in MB
        self.assertEqual(result.flops, DeviceFlops(fp32=3.55 * TFLOPS, fp16=7.10 * TFLOPS, int8=14.20 * TFLOPS))
        self.assertEqual(str(result),
                         "Model: Mac14,2. Chip: Apple M2. Memory: 8192MB. Flops: fp32: 3.55 TFLOPS, fp16: 7.10 TFLOPS, int8: 14.20 TFLOPS")


    @patch('subprocess.check_output')
    @patch('psutil.virtual_memory')
    def test_mac_device_capabilities_mocked_m3_max(self, mock_virtual_memory, mock_check_output):
        # Mock the system_profiler JSON output
        mock_system_profiler_output = {
            "SPHardwareDataType": [
                {
                    "machine_model": "Mac15,9",
                    "machine_name": "MacBook Pro",
                    "chip_type": "Apple M3 Max",
                    "number_processors": 16,
                    "physical_memory": "128 GB"
                }
            ]
        }
        mock_check_output.side_effect = [
            json.dumps(mock_system_profiler_output).encode('utf-8'),  # For system_profiler
            b"Apple M3 Max"  # For sysctl
        ]

        # Mock the virtual memory
        mock_virtual_memory.return_value.total = 128 * 1024 * 1024 * 1024  # 128 GB in bytes

        # Call the function
        result = mac_device_capabilities()

        # Check the results
        self.assertIsInstance(result, DeviceCapabilities)
        self.assertEqual(result.model, "Mac15,9")
        self.assertEqual(result.chip, "Apple M3 Max")
        self.assertEqual(result.memory, 131072)  # 128 GB in MB
        self.assertEqual(result.flops, DeviceFlops(fp32=14.20 * TFLOPS, fp16=28.40 * TFLOPS, int8=56.80 * TFLOPS))

        # Check the string representation
        result_str = str(result)
        self.assertIn("Model: Mac15,9", result_str)
        self.assertIn("Chip: Apple M3 Max", result_str)
        self.assertIn("Memory: 131072MB", result_str)
        self.assertIn("Flops: fp32: 14.20 TFLOPS, fp16: 28.40 TFLOPS, int8: 56.80 TFLOPS", result_str)


if __name__ == '__main__':
    unittest.main()
