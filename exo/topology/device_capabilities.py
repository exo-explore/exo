from dataclasses import dataclass
import subprocess
import platform

@dataclass
class DeviceCapabilities:
    model: str
    chip: str
    memory: int

def device_capabilities() -> DeviceCapabilities:
    system = platform.system()
    if system == 'Darwin':
        return mac_device_capabilities()
    elif system == 'Linux':
        return linux_device_capabilities()
    # elif system == 'Windows':
    #     return windows_device_capabilities()
    else:
        return DeviceCapabilities(model="Unknown Model", chip="Unknown Chip", memory=0)

def mac_device_capabilities() -> DeviceCapabilities:
    # Fetch the model of the Mac using system_profiler
    model = subprocess.check_output(['system_profiler', 'SPHardwareDataType']).decode('utf-8')
    model_line = next((line for line in model.split('\n') if "Model Name" in line), None)
    model_id = model_line.split(': ')[1] if model_line else "Unknown Model"
    chip_line = next((line for line in model.split('\n') if "Chip" in line), None)
    chip_id = chip_line.split(': ')[1] if chip_line else "Unknown Chip"
    memory_line = next((line for line in model.split('\n') if "Memory" in line), None)
    memory_str = memory_line.split(': ')[1] if memory_line else "Unknown Memory"
    memory_units = memory_str.split()
    memory_value = int(memory_units[0])
    if memory_units[1] == "GB":
        memory = memory_value * 1024
    else:
        memory = memory_value

    # Assuming static values for other attributes for demonstration
    return DeviceCapabilities(model=model_id, chip=chip_id, memory=memory)

def linux_device_capabilities() -> DeviceCapabilities:
    import psutil
    from tinygrad import Device
    
    print(f"tinygrad {Device.DEFAULT=}")
    if Device.DEFAULT == "CUDA" or Device.DEFAULT == "NV" or Device.DEFAULT=="GPU":
        import pynvml, pynvml_utils
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        print(f"NVIDIA device {gpu_name=} {gpu_memory_info=}")

        return DeviceCapabilities(model=f"Linux Box ({gpu_name})", chip=gpu_name, memory=gpu_memory_info.total)
    elif Device.DEFAULT == "AMD":
        # TODO AMD support
        return DeviceCapabilities(model="Linux Box (AMD)", chip="Unknown AMD", memory=psutil.virtual_memory().total)
    else:
        return DeviceCapabilities(model=f"Linux Box (Device: {Device.DEFAULT})", chip=f"Unknown Chip (Device: {Device.DEFAULT})", memory=psutil.virtual_memory().total // 2**20)
