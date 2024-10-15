from exo.inference.inference_engine import InferenceEngine
from exo import DEBUG
from dataclasses import dataclass, asdict
import subprocess
import psutil

TFLOPS = 1.00


@dataclass
class DeviceFlops:
  # units of TFLOPS
  fp32: float
  fp16: float
  int8: float

  def __str__(self):
    return f"fp32: {self.fp32 / TFLOPS:.2f} TFLOPS, fp16: {self.fp16 / TFLOPS:.2f} TFLOPS, int8: {self.int8 / TFLOPS:.2f} TFLOPS"

  def to_dict(self):
    return asdict(self)


@dataclass
class DeviceCapabilities:
  model: str
  chip: str
  memory: int
  flops: DeviceFlops

  def __str__(self):
    return f"Model: {self.model}. Chip: {self.chip}. Memory: {self.memory}MB. Flops: {self.flops}"

  def __post_init__(self):
    if isinstance(self.flops, dict):
      self.flops = DeviceFlops(**self.flops)

  def to_dict(self):
    return {"model": self.model, "chip": self.chip, "memory": self.memory, "flops": self.flops.to_dict()}


UNKNOWN_DEVICE_CAPABILITIES = DeviceCapabilities(model="Unknown Model", chip="Unknown Chip", memory=0, flops=DeviceFlops(fp32=0, fp16=0, int8=0))


async def device_capabilities(inference_engine: InferenceEngine) -> DeviceCapabilities:
  if psutil.MACOS:
    return await mac_device_capabilities(inference_engine)
  elif psutil.LINUX:
    return await linux_device_capabilities(inference_engine)
  else:
    return DeviceCapabilities(
      model="Unknown Device",
      chip="Unknown Chip",
      memory=psutil.virtual_memory().total // 2**20,
      flops=DeviceFlops(fp32=0, fp16=0, int8=0),
    )


async def mac_device_capabilities(inference_engine: InferenceEngine) -> DeviceCapabilities:
  # Fetch the model of the Mac using system_profiler
  model = subprocess.check_output(["system_profiler", "SPHardwareDataType"]).decode("utf-8")
  model_line = next((line for line in model.split("\n") if "Model Name" in line), None)
  model_id = model_line.split(": ")[1] if model_line else "Unknown Model"
  chip_line = next((line for line in model.split("\n") if "Chip" in line), None)
  chip_id = chip_line.split(": ")[1] if chip_line else "Unknown Chip"
  memory_line = next((line for line in model.split("\n") if "Memory" in line), None)
  memory_str = memory_line.split(": ")[1] if memory_line else "Unknown Memory"
  memory_units = memory_str.split()
  memory_value = int(memory_units[0])
  if memory_units[1] == "GB":
    memory = memory_value*1024
  else:
    memory = memory_value

  return DeviceCapabilities(model=model_id, chip=chip_id, memory=memory, flops=DeviceFlops(*await inference_engine.benchmark_tflops()))


async def linux_device_capabilities(inference_engine: InferenceEngine) -> DeviceCapabilities:
  import psutil
  from tinygrad import Device

  if DEBUG >= 2: print(f"tinygrad {Device.DEFAULT=}")
  if Device.DEFAULT == "CUDA" or Device.DEFAULT == "NV" or Device.DEFAULT == "GPU":
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_raw_name = pynvml.nvmlDeviceGetName(handle).upper()
    gpu_name = gpu_raw_name.rsplit(" ", 1)[0] if gpu_raw_name.endswith("GB") else gpu_raw_name
    gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    flops = DeviceFlops(*await inference_engine.benchmark_tflops())

    if DEBUG >= 2: print(f"NVIDIA device {gpu_name=} {gpu_memory_info=}")

    return DeviceCapabilities(
      model=f"Linux Box ({gpu_name})",
      chip=gpu_name,
      memory=gpu_memory_info.total // 2**20,
      flops=flops,
    )
  elif Device.DEFAULT == "AMD":
    # TODO AMD support
    return DeviceCapabilities(
      model="Linux Box (AMD)",
      chip="Unknown AMD",
      memory=psutil.virtual_memory().total // 2**20,
      flops=flops,
    )
  else:
    return DeviceCapabilities(
      model=f"Linux Box (Device: {Device.DEFAULT})",
      chip=f"Unknown Chip (Device: {Device.DEFAULT})",
      memory=psutil.virtual_memory().total // 2**20,
      flops=flops,
    )
