from exo.inference.inference_engine import InferenceEngine
from typing import Any
from pydantic import BaseModel
from exo import DEBUG
import subprocess
import psutil
from exo.topology.device_flops import DeviceFlops


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


class LazyDeviceCapabilities:
  def __init__(self, inference_engine: InferenceEngine):
    self.inference_engine = inference_engine
    self._caps: DeviceCapabilities | None = None

  async def __getattr__(self, name):
    if name == 'caps':
      if self._caps is None:
        try:
          self._caps = await device_capabilities(self.inference_engine)
        except Exception as e:
          self._caps = UNKNOWN_DEVICE_CAPABILITIES
      return self._caps
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


UNKNOWN_DEVICE_CAPABILITIES = DeviceCapabilities(
  model="Unknown Device",
  chip="Unknown Chip",
  memory=psutil.virtual_memory().total // 2**20,
  flops=DeviceFlops(fp32=0, fp16=0, int8=0),
)


async def device_capabilities(inference_engine: InferenceEngine) -> DeviceCapabilities:
  if psutil.MACOS:
    return await mac_device_capabilities(inference_engine)
  elif psutil.LINUX:
    return await linux_device_capabilities(inference_engine)
  else:
    return UNKNOWN_DEVICE_CAPABILITIES


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

  return DeviceCapabilities(model=model_id, chip=chip_id, memory=memory, flops=await inference_engine.benchmark_tflops())


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
    flops = await inference_engine.benchmark_tflops()

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
