"""
Rockchip NPU detection for RKLLM inference engine.

Detects RK3588/RK3576 SoCs via device-tree or RKLLM library presence.
"""

import os

# Known RKLLM library paths
RKLLM_LIB_PATHS = [
  os.path.expanduser('~/RKLLAMA/lib/librkllmrt.so'),
  '/usr/lib/librkllmrt.so',
  '/usr/local/lib/librkllmrt.so',
]

# Rockchip SoC identifiers with NPU support
ROCKCHIP_NPU_SOCS = ['rk3588', 'rk3576']


def detect_rockchip_npu() -> bool:
  """
  Check if running on a Rockchip RK3588/RK3576 with NPU support.

  Detection methods (in order):
  1. Check /proc/device-tree/compatible for Rockchip SoC identifiers
  2. Check for RKLLM runtime library at known paths

  Returns:
    True if Rockchip NPU is detected, False otherwise.
  """
  # Method 1: Check device-tree for Rockchip SoC
  compatible_path = '/proc/device-tree/compatible'
  if os.path.exists(compatible_path):
    try:
      with open(compatible_path, 'rb') as f:
        compatible = f.read().decode('utf-8', errors='ignore').lower()
        for soc in ROCKCHIP_NPU_SOCS:
          if soc in compatible:
            return True
    except Exception:
      pass

  # Method 2: Check for RKLLM library as fallback indicator
  for path in RKLLM_LIB_PATHS:
    if os.path.exists(path):
      return True

  return False


def get_rockchip_soc_name() -> str:
  """
  Get the specific Rockchip SoC name if detected.

  Returns:
    SoC name (e.g., 'RK3588') or 'Unknown' if not detected.
  """
  compatible_path = '/proc/device-tree/compatible'
  if os.path.exists(compatible_path):
    try:
      with open(compatible_path, 'rb') as f:
        compatible = f.read().decode('utf-8', errors='ignore').lower()
        for soc in ROCKCHIP_NPU_SOCS:
          if soc in compatible:
            return soc.upper()
    except Exception:
      pass

  # Check if library exists but can't identify SoC
  for path in RKLLM_LIB_PATHS:
    if os.path.exists(path):
      return "Rockchip (Unknown SoC)"

  return "Unknown"


def get_rkllm_library_path() -> str:
  """
  Find the RKLLM runtime library path.

  Returns:
    Path to librkllmrt.so or None if not found.
  """
  for path in RKLLM_LIB_PATHS:
    if os.path.exists(path):
      return path
  return None
