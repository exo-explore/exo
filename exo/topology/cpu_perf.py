import os
from typing import Any

from pydantic import BaseModel
TFLOPS_EXT = 1e12  # 1 TeraFLOP = 10^12 FLOPs
TFLOPS = 1.00


class DeviceFlops(BaseModel):
  # units of TFLOPS
  fp32: float
  fp16: float
  int8: float

  def __str__(self):
    return f"fp32: {self.fp32 / TFLOPS:.2f} TFLOPS, fp16: {self.fp16 / TFLOPS:.2f} TFLOPS, int8: {self.int8 / TFLOPS:.2f} TFLOPS"

  def to_dict(self):
    return self.model_dump()


class DeviceCapabilities(BaseModel):
  model: str
  chip: str
  memory: int
  flops: DeviceFlops

  def __str__(self):
    return f"Model: {self.model}. Chip: {self.chip}. Memory: {self.memory}MB. Flops: {self.flops}"

  def model_post_init(self, __context: Any) -> None:
    if isinstance(self.flops, dict):
      self.flops = DeviceFlops(**self.flops)

  def to_dict(self):
    return {"model": self.model, "chip": self.chip, "memory": self.memory, "flops": self.flops.to_dict()}


def get_cpu_info():
    """Retrieve CPU frequency and core count"""
    with open("/proc/cpuinfo", "r") as f:
        cpuinfo = f.read()
        
    freq_line = next(line for line in cpuinfo.split('\n') if 'cpu MHz' in line)
    freq_ghz = float(freq_line.split(':')[1].strip()) / 1000
    
    return freq_ghz, os.cpu_count()

def calculate_precision_flops(freq_ghz, cores, ops_per_cycle):
    """
    Calculate theoretical FLOPS for a specific precision:
    - freq_ghz: CPU base frequency in GHz
    - cores: Number of physical cores
    - ops_per_cycle: Operations per clock cycle for the precision
    """
    return round(((freq_ghz * 1e9) * cores * ops_per_cycle / TFLOPS_EXT), 2 )# Convert to TFLOPS

def get_device_flops(
    fp32_ops=32,  # Typical for AVX2/FMA (8 elements * 2 ops * 2 FMA units)
    fp16_ops=64,  # Typical for AVX-512 FP16
    int8_ops=128  # Typical for AVX-512 VNNI
):
    """
    Calculate theoretical peak FLOPS for different precisions.
    Default values assume modern x86 CPU with vector extensions.
    """
    try:
        freq, cores = get_cpu_info()
        
        return DeviceFlops(
            fp32=calculate_precision_flops(freq, cores, fp32_ops),
            fp16=calculate_precision_flops(freq, cores, fp16_ops),
            int8=calculate_precision_flops(freq, cores, int8_ops)
        )
    except Exception as e:
        print(f"Error: {e}")
        return DeviceFlops(fp32=0, fp16=0, int8=0)

# Usage example
flops = get_device_flops()
print(flops)
print(flops.to_dict())
