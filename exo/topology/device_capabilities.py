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

CHIP_FLOPS = {
  # Source: https://www.cpu-monkey.com
  # Note: currently no distinction between variants of M3 Max and M3 Pro, we pick the lower one to be conservative
  ### M chips
  "Apple M1": DeviceFlops(fp32=2.29*TFLOPS, fp16=4.58*TFLOPS, int8=9.16*TFLOPS),
  "Apple M1 Pro": DeviceFlops(fp32=5.30*TFLOPS, fp16=10.60*TFLOPS, int8=21.20*TFLOPS),
  "Apple M1 Max": DeviceFlops(fp32=10.60*TFLOPS, fp16=21.20*TFLOPS, int8=42.40*TFLOPS),
  "Apple M1 Ultra": DeviceFlops(fp32=21.20*TFLOPS, fp16=42.40*TFLOPS, int8=84.80*TFLOPS),
  "Apple M2": DeviceFlops(fp32=3.55*TFLOPS, fp16=7.10*TFLOPS, int8=14.20*TFLOPS),
  "Apple M2 Pro": DeviceFlops(fp32=5.68*TFLOPS, fp16=11.36*TFLOPS, int8=22.72*TFLOPS),
  "Apple M2 Max": DeviceFlops(fp32=13.49*TFLOPS, fp16=26.98*TFLOPS, int8=53.96*TFLOPS),
  "Apple M2 Ultra": DeviceFlops(fp32=26.98*TFLOPS, fp16=53.96*TFLOPS, int8=107.92*TFLOPS),
  "Apple M3": DeviceFlops(fp32=3.55*TFLOPS, fp16=7.10*TFLOPS, int8=14.20*TFLOPS),
  "Apple M3 Max": DeviceFlops(fp32=14.20*TFLOPS, fp16=28.40*TFLOPS, int8=56.80*TFLOPS),
  "Apple M3 Pro": DeviceFlops(fp32=4.97*TFLOPS, fp16=9.94*TFLOPS, int8=19.88*TFLOPS),
  "Apple M4": DeviceFlops(fp32=3.55*TFLOPS, fp16=7.10*TFLOPS, int8=14.20*TFLOPS),
  ### A chips
  "Apple A13 Bionic": DeviceFlops(fp32=0.69*TFLOPS, fp16=1.38*TFLOPS, int8=2.76*TFLOPS),
  "Apple A14 Bionic": DeviceFlops(fp32=0.75*TFLOPS, fp16=1.50*TFLOPS, int8=3.00*TFLOPS),
  "Apple A15 Bionic": DeviceFlops(fp32=1.37*TFLOPS, fp16=2.74*TFLOPS, int8=5.48*TFLOPS),
  "Apple A16 Bionic": DeviceFlops(fp32=1.79*TFLOPS, fp16=3.58*TFLOPS, int8=7.16*TFLOPS),
  "Apple A17 Pro": DeviceFlops(fp32=2.15*TFLOPS, fp16=4.30*TFLOPS, int8=8.60*TFLOPS),
  ### NVIDIA GPUs
  # RTX 40 series
  "NVIDIA GEFORCE RTX 4090": DeviceFlops(fp32=82.58*TFLOPS, fp16=165.16*TFLOPS, int8=330.32*TFLOPS),
  "NVIDIA GEFORCE RTX 4080": DeviceFlops(fp32=48.74*TFLOPS, fp16=97.48*TFLOPS, int8=194.96*TFLOPS),
  "NVIDIA GEFORCE RTX 4080 SUPER": DeviceFlops(fp32=52.0*TFLOPS, fp16=104.0*TFLOPS, int8=208.0*TFLOPS),
  "NVIDIA GEFORCE RTX 4070 TI SUPER": DeviceFlops(fp32=40.0*TFLOPS, fp16=80.0*TFLOPS, int8=160.0*TFLOPS),
  "NVIDIA GEFORCE RTX 4070 TI": DeviceFlops(fp32=39.43*TFLOPS, fp16=78.86*TFLOPS, int8=157.72*TFLOPS),
  "NVIDIA GEFORCE RTX 4070 SUPER": DeviceFlops(fp32=30.0*TFLOPS, fp16=60.0*TFLOPS, int8=120.0*TFLOPS),
  "NVIDIA GEFORCE RTX 4070": DeviceFlops(fp32=29.0*TFLOPS, fp16=58.0*TFLOPS, int8=116.0*TFLOPS),
  "NVIDIA GEFORCE RTX 4060 TI 16GB": DeviceFlops(fp32=22.0*TFLOPS, fp16=44.0*TFLOPS, int8=88.0*TFLOPS),
  "NVIDIA GEFORCE RTX 4060 TI": DeviceFlops(fp32=22.0*TFLOPS, fp16=44.0*TFLOPS, int8=88.0*TFLOPS),
  # RTX 30 series
  "NVIDIA GEFORCE RTX 3050": DeviceFlops(fp32=9.11*TFLOPS, fp16=18.22*TFLOPS, int8=36.44*TFLOPS),
  "NVIDIA GEFORCE RTX 3060": DeviceFlops(fp32=13.0*TFLOPS, fp16=26.0*TFLOPS, int8=52.0*TFLOPS),
  "NVIDIA GEFORCE RTX 3060 TI": DeviceFlops(fp32=16.2*TFLOPS, fp16=32.4*TFLOPS, int8=64.8*TFLOPS),
  "NVIDIA GEFORCE RTX 3070": DeviceFlops(fp32=20.3*TFLOPS, fp16=40.6*TFLOPS, int8=81.2*TFLOPS),
  "NVIDIA GEFORCE RTX 3070 TI": DeviceFlops(fp32=21.8*TFLOPS, fp16=43.6*TFLOPS, int8=87.2*TFLOPS),
  "NVIDIA GEFORCE RTX 3080 (10 GB)": DeviceFlops(fp32=29.8*TFLOPS, fp16=59.6*TFLOPS, int8=119.2*TFLOPS),
  "NVIDIA GEFORCE RTX 3080 (12 GB)": DeviceFlops(fp32=30.6*TFLOPS, fp16=61.2*TFLOPS, int8=122.4*TFLOPS),
  "NVIDIA GEFORCE RTX 3080 TI": DeviceFlops(fp32=34.1*TFLOPS, fp16=68.2*TFLOPS, int8=136.4*TFLOPS),
  "NVIDIA GEFORCE RTX 3090": DeviceFlops(fp32=35.6*TFLOPS, fp16=71.2*TFLOPS, int8=142.4*TFLOPS),
  "NVIDIA GEFORCE RTX 3090 TI": DeviceFlops(fp32=40.0*TFLOPS, fp16=80.0*TFLOPS, int8=160.0*TFLOPS),
  # RTX 20 series
  "NVIDIA GEFORCE RTX 2060": DeviceFlops(fp32=6.45*TFLOPS, fp16=12.9*TFLOPS, int8=25.8*TFLOPS),
  "NVIDIA GEFORCE RTX 2060 SUPER": DeviceFlops(fp32=7.2*TFLOPS, fp16=14.4*TFLOPS, int8=28.8*TFLOPS),
  "NVIDIA GEFORCE RTX 2070": DeviceFlops(fp32=7.46*TFLOPS, fp16=14.93*TFLOPS, int8=29.86*TFLOPS),
  "NVIDIA GEFORCE RTX 2070 SUPER": DeviceFlops(fp32=9.06*TFLOPS, fp16=18.12*TFLOPS, int8=36.24*TFLOPS),
  "NVIDIA GEFORCE RTX 2080": DeviceFlops(fp32=10.07*TFLOPS, fp16=20.14*TFLOPS, int8=40.28*TFLOPS),
  "NVIDIA GEFORCE RTX 2080 TI": DeviceFlops(fp32=13.45*TFLOPS, fp16=26.9*TFLOPS, int8=40.28*TFLOPS),
  "NVIDIA GEFORCE RTX 2080 SUPER": DeviceFlops(fp32=11.15*TFLOPS, fp16=22.30*TFLOPS, int8=44.60*TFLOPS),
  "NVIDIA TITAN RTX": DeviceFlops(fp32=16.31*TFLOPS, fp16=32.62*TFLOPS, int8=65.24*TFLOPS),
  # GTX 10 series
  "NVIDIA GEFORCE GTX 1050 TI": DeviceFlops(fp32=2.0*TFLOPS, fp16=4.0*TFLOPS, int8=8.0*TFLOPS),
  # QUADRO RTX Ampere series
  "NVIDIA RTX A2000": DeviceFlops(fp32=7.99*TFLOPS, fp16=7.99*TFLOPS, int8=31.91*TFLOPS),
  "NVIDIA RTX A4000": DeviceFlops(fp32=19.17*TFLOPS, fp16=19.17*TFLOPS, int8=76.68*TFLOPS),
  "NVIDIA RTX A4500": DeviceFlops(fp32=23.65*TFLOPS, fp16=23.65*TFLOPS, int8=94.6*TFLOPS),
  "NVIDIA RTX A5000": DeviceFlops(fp32=27.8*TFLOPS, fp16=27.8*TFLOPS, int8=111.2*TFLOPS),
  "NVIDIA RTX A6000": DeviceFlops(fp32=38.71*TFLOPS, fp16=38.71*TFLOPS, int8=154.84*TFLOPS),
  # NVIDIA Ada Lovelace Architecture-Based
  "NVIDIA RTX 4000 ADA GENERATION": DeviceFlops(fp32=26.7*TFLOPS, fp16=26.7*TFLOPS, int8=258.0*TFLOPS),
  # Common Server GPUs
  "NVIDIA A40 48GB PCIE": DeviceFlops(fp32=37.4*TFLOPS, fp16=149.7*TFLOPS, int8=299.3*TFLOPS),
  "NVIDIA A100 40GB PCIE": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
  "NVIDIA A800 40GB PCIE": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
  "NVIDIA A100 80GB PCIE": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
  "NVIDIA A800 80GB PCIE": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
  "NVIDIA A100 80GB SXM": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
  "NVIDIA A800 80GB SXM": DeviceFlops(fp32=19.5*TFLOPS, fp16=312.0*TFLOPS, int8=624.0*TFLOPS),
  # ... add more devices if needed ...
  ### AMD GPUs
  # RX 6000 series
  "AMD Radeon RX 6900 XT": DeviceFlops(fp32=23.04*TFLOPS, fp16=46.08*TFLOPS, int8=92.16*TFLOPS),
  "AMD Radeon RX 6800 XT": DeviceFlops(fp32=20.74*TFLOPS, fp16=41.48*TFLOPS, int8=82.96*TFLOPS),
  "AMD Radeon RX 6800": DeviceFlops(fp32=16.17*TFLOPS, fp16=32.34*TFLOPS, int8=64.68*TFLOPS),
  "AMD Radeon RX 6700 XT": DeviceFlops(fp32=13.21*TFLOPS, fp16=26.42*TFLOPS, int8=52.84*TFLOPS),
  "AMD Radeon RX 6700": DeviceFlops(fp32=11.4*TFLOPS, fp16=22.8*TFLOPS, int8=45.6*TFLOPS),
  "AMD Radeon RX 6600 XT": DeviceFlops(fp32=10.6*TFLOPS, fp16=21.2*TFLOPS, int8=42.4*TFLOPS),
  "AMD Radeon RX 6600": DeviceFlops(fp32=8.93*TFLOPS, fp16=17.86*TFLOPS, int8=35.72*TFLOPS),
  "AMD Radeon RX 6500 XT": DeviceFlops(fp32=5.77*TFLOPS, fp16=11.54*TFLOPS, int8=23.08*TFLOPS),
  "AMD Radeon RX 6400": DeviceFlops(fp32=3.57*TFLOPS, fp16=7.14*TFLOPS, int8=14.28*TFLOPS),
  # RX 7000 series
  "AMD Radeon RX 7900 XTX": DeviceFlops(fp32=61.4*TFLOPS, fp16=122.8*TFLOPS, int8=245.6*TFLOPS),
  "AMD Radeon RX 7900 XT": DeviceFlops(fp32=53.4*TFLOPS, fp16=106.8*TFLOPS, int8=213.6*TFLOPS),
  "AMD Radeon RX 7800 XT": DeviceFlops(fp32=42.6*TFLOPS, fp16=85.2*TFLOPS, int8=170.4*TFLOPS),
  "AMD Radeon RX 7700 XT": DeviceFlops(fp32=34.2*TFLOPS, fp16=68.4*TFLOPS, int8=136.8*TFLOPS),
  "AMD Radeon RX 7600": DeviceFlops(fp32=21.5*TFLOPS, fp16=43.0*TFLOPS, int8=86.0*TFLOPS),
  "AMD Radeon RX 7500": DeviceFlops(fp32=16.2*TFLOPS, fp16=32.4*TFLOPS, int8=64.8*TFLOPS),
  ### Qualcomm embedded chips: TODO
}
CHIP_FLOPS.update({f"LAPTOP GPU {key}": value for key, value in CHIP_FLOPS.items()})
CHIP_FLOPS.update({f"Laptop GPU {key}": value for key, value in CHIP_FLOPS.items()})
CHIP_FLOPS.update({f"{key} LAPTOP GPU": value for key, value in CHIP_FLOPS.items()})
CHIP_FLOPS.update({f"{key} Laptop GPU": value for key, value in CHIP_FLOPS.items()})


def device_capabilities() -> DeviceCapabilities:
  if psutil.MACOS:
    return mac_device_capabilities()
  elif psutil.LINUX:
    return linux_device_capabilities()
  else:
    return DeviceCapabilities(
      model="Unknown Device",
      chip="Unknown Chip",
      memory=psutil.virtual_memory().total // 2**20,
      flops=DeviceFlops(fp32=0, fp16=0, int8=0),
    )


def mac_device_capabilities() -> DeviceCapabilities:
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

  # Assuming static values for other attributes for demonstration
  return DeviceCapabilities(model=model_id, chip=chip_id, memory=memory, flops=CHIP_FLOPS.get(chip_id, DeviceFlops(fp32=0, fp16=0, int8=0)))


def linux_device_capabilities() -> DeviceCapabilities:
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

    if DEBUG >= 2: print(f"NVIDIA device {gpu_name=} {gpu_memory_info=}")

    return DeviceCapabilities(
      model=f"Linux Box ({gpu_name})",
      chip=gpu_name,
      memory=gpu_memory_info.total // 2**20,
      flops=CHIP_FLOPS.get(gpu_name, DeviceFlops(fp32=0, fp16=0, int8=0)),
    )
  elif Device.DEFAULT == "AMD":
    # TODO AMD support
    return DeviceCapabilities(
      model="Linux Box (AMD)",
      chip="Unknown AMD",
      memory=psutil.virtual_memory().total // 2**20,
      flops=DeviceFlops(fp32=0, fp16=0, int8=0),
    )
  else:
    return DeviceCapabilities(
      model=f"Linux Box (Device: {Device.DEFAULT})",
      chip=f"Unknown Chip (Device: {Device.DEFAULT})",
      memory=psutil.virtual_memory().total // 2**20,
      flops=DeviceFlops(fp32=0, fp16=0, int8=0),
    )
