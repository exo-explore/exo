from exo import DEBUG
from dataclasses import dataclass, asdict
import psutil
import subprocess
import json

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
        return {
            'model': self.model,
            'chip': self.chip,
            'memory': self.memory,
            'flops': self.flops.to_dict()
        }

UNKNOWN_DEVICE_CAPABILITIES = DeviceCapabilities(model="Unknown Model", chip="Unknown Chip", memory=0, flops=DeviceFlops(fp32=0, fp16=0, int8=0))

CHIP_FLOPS = {
    # Source: https://www.cpu-monkey.com
    # Note: currently no distinction between variants of M3 Max and M3 Pro, we pick the lower one to be conservative
    ### M chips
    "Apple M1": DeviceFlops(fp32=2.29*TFLOPS, fp16=4.58*TFLOPS, int8=9.16*TFLOPS),
    "Apple M1 Pro": DeviceFlops(fp32=4.5 * TFLOPS, fp16=9.0 * TFLOPS, int8=18.0 * TFLOPS),
    "Apple M1 Max": DeviceFlops(fp32=10.4 * TFLOPS, fp16=20.8 * TFLOPS, int8=41.6 * TFLOPS),
    "Apple M1 Ultra": DeviceFlops(fp32=21.20*TFLOPS, fp16=42.40*TFLOPS, int8=84.80*TFLOPS),
    "Apple M2": DeviceFlops(fp32=3.55*TFLOPS, fp16=7.10*TFLOPS, int8=14.20*TFLOPS),
    "Apple M2 Pro": DeviceFlops(fp32=5.68*TFLOPS, fp16=11.36*TFLOPS, int8=22.72*TFLOPS),
    "Apple M2 Max": DeviceFlops(fp32=13.49*TFLOPS, fp16=26.98*TFLOPS, int8=53.96*TFLOPS),
    "Apple M2 Ultra": DeviceFlops(fp32=26.98*TFLOPS, fp16=53.96*TFLOPS, int8=107.92*TFLOPS),
    "Apple M3": DeviceFlops(fp32=3.55*TFLOPS, fp16=7.10*TFLOPS, int8=14.20*TFLOPS),
    "Apple M3 Pro": DeviceFlops(fp32=4.97*TFLOPS, fp16=9.94*TFLOPS, int8=19.88*TFLOPS),
    "Apple M3 Max": DeviceFlops(fp32=14.20*TFLOPS, fp16=28.40*TFLOPS, int8=56.80*TFLOPS),

    ### A chips
    "Apple A13 Bionic": DeviceFlops(fp32=0.69*TFLOPS, fp16=1.38*TFLOPS, int8=2.76*TFLOPS),
    "Apple A14 Bionic": DeviceFlops(fp32=0.75*TFLOPS, fp16=1.50*TFLOPS, int8=3.00*TFLOPS),
    "Apple A15 Bionic": DeviceFlops(fp32=1.37*TFLOPS, fp16=2.74*TFLOPS, int8=5.48*TFLOPS),
    "Apple A16 Bionic": DeviceFlops(fp32=1.79*TFLOPS, fp16=3.58*TFLOPS, int8=7.16*TFLOPS),
    "Apple A17 Pro": DeviceFlops(fp32=2.15*TFLOPS, fp16=4.30*TFLOPS, int8=8.60*TFLOPS),
    ### NVIDIA GPUs: TODO
    ### AMD GPUs: TODO
    ### Qualcomm embedded chips: TODO
}


def device_capabilities() -> DeviceCapabilities:
    if psutil.MACOS:
        return mac_device_capabilities()
    elif psutil.LINUX:
        return linux_device_capabilities()
    else:
        return DeviceCapabilities(model=f"Unknown Device", chip=f"Unknown Chip", memory=psutil.virtual_memory().total // 2**20, flops=DeviceFlops(fp32=0, fp16=0, int8=0))


def mac_device_capabilities() -> DeviceCapabilities:
    hw_info_json = subprocess.check_output(['system_profiler', 'SPHardwareDataType', '-json']).decode('utf-8')
    hw_info = json.loads(hw_info_json)

    # Extract relevant information from the JSON
    hardware_data = hw_info['SPHardwareDataType'][0]
    model_id = hardware_data.get('machine_model', 'Unknown Model')
    chip_id = "Unknown Chip"
    cores = hardware_data.get('number_processors', 0)
    memory = hardware_data.get('physical_memory', '0 GB').split()[0]
    memory = int(memory) * 1024  # Convert GB to MB

    # Try to identify the chip using sysctl
    try:
        sysctl_output = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode('utf-8').strip()
        if "Apple M1" in sysctl_output:
            if "Max" in sysctl_output:
                chip_id = "Apple M1 Max"
            elif "Pro" in sysctl_output:
                chip_id = "Apple M1 Pro"
            else:
                chip_id = "Apple M1"
    except subprocess.CalledProcessError:
        pass

    # If sysctl didn't work, infer based on cores and memory
    if chip_id == "Unknown Chip":
        if "MacBook Pro" in model_id:
            if cores == 10 and memory >= 64 * 1024:
                chip_id = "Apple M1 Max"
            elif cores == 10:
                chip_id = "Apple M1 Pro"
            elif cores > 10:
                chip_id = "Apple M1 Max"

    flops = CHIP_FLOPS.get(chip_id, DeviceFlops(fp32=10.4 * TFLOPS, fp16=20.8 * TFLOPS, int8=41.6 * TFLOPS))

    if DEBUG >= 1:
        print(f"\nDetailed Mac Device Capabilities:")
        print(f"Model: {model_id}")
        print(f"Chip: {chip_id}")
        print(f"Total Cores: {cores}")
        print(f"Memory: {memory} MB")
        print(f"TFLOPS Calculations:")
        print(f"  FP32: {flops.fp32 / TFLOPS:.2f} TFLOPS")
        print(f"  FP16: {flops.fp16 / TFLOPS:.2f} TFLOPS")
        print(f"  INT8: {flops.int8 / TFLOPS:.2f} TFLOPS")
        if chip_id == "Unknown Chip":
            print(f"Note: Chip was not directly identified. TFLOPS values are estimates.")

    return DeviceCapabilities(model=model_id, chip=chip_id, memory=memory, flops=flops)


def linux_device_capabilities() -> DeviceCapabilities:
    import psutil
    from tinygrad import Device

    if DEBUG >= 2: print(f"tinygrad {Device.DEFAULT=}")
    if Device.DEFAULT == "CUDA" or Device.DEFAULT == "NV" or Device.DEFAULT=="GPU":
        import pynvml, pynvml_utils
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        if DEBUG >= 2: print(f"NVIDIA device {gpu_name=} {gpu_memory_info=}")

        return DeviceCapabilities(model=f"Linux Box ({gpu_name})", chip=gpu_name, memory=gpu_memory_info.total // 2**20, flops=CHIP_FLOPS.get(gpu_name, DeviceFlops(fp32=0, fp16=0, int8=0)))
    elif Device.DEFAULT == "AMD":
        # TODO AMD support
        return DeviceCapabilities(model="Linux Box (AMD)", chip="Unknown AMD", memory=psutil.virtual_memory().total // 2**20, flops=DeviceFlops(fp32=0, fp16=0, int8=0))
    else:
        return DeviceCapabilities(model=f"Linux Box (Device: {Device.DEFAULT})", chip=f"Unknown Chip (Device: {Device.DEFAULT})", memory=psutil.virtual_memory().total // 2**20, flops=DeviceFlops(fp32=0, fp16=0, int8=0))
