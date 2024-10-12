from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
import socket
import yaml

class ReadManualConfig():
  def __init__(
    self,
    discovery_config: str = "topology.yml",
  ):
    self.discovery_config = discovery_config
    self.whoami = f"{socket.gethostname()}"
    self.node_id = "NONE"
    self.node_address = "NONE"
    self.node_port = "NONE"
    self.model = "NONE"
    self.chip = "NONE"
    self.memory = "NONE"
    self.fp32 = "NONE"
    self.fp16 = "NONE"
    self.int8 = "NONE"

  def device_capabilities(self, gethostname) -> DeviceCapabilities:
    with open(self.discovery_config, 'r') as f:
      config_devices = yaml.safe_load(f)
      f.close()

    for device in config_devices:
      print(f"{device['server']} {device['id']}:")
      print(f"Search for {gethostname} !")
      if f"{gethostname}" == f"{device['server']}":
        print(f"Adresse: {device['address']} {device['port']}")
        self.node_id = f"{device['id']}"
        self.node_address = f"{device['address']}"
        self.node_port = f"{device['port']}"
        print(f"Capabilities:")
        for capability, value in device['device_capabilities'].items():
          print(f"{capability}: {value}")
          if f"{capability}" == "model":
            self.model = f"{value}"
          if f"{capability}" == "chip":
            self.chip = f"{value}"
          if f"{capability}" == "memory":
            self.memory = (int(f"{value}"))
          if f"{capability}" == "flops":
            for flopstr, flopvalue in device['device_capabilities']['flops'].items():
                if f"{flopstr}" == "fp32":
                  self.fp32 = (int(f"{flopvalue}"))
                if f"{flopstr}" == "fp16":
                  self.fp16 = (int(f"{flopvalue}"))
                if f"{flopstr}" == "int8":
                  self.int8 = (int(f"{flopvalue}"))

        return DeviceCapabilities(
          model=self.model,
          chip=self.chip,
          memory=self.memory // 2**20,
          flops=DeviceFlops(fp32=self.fp32, fp16=self.fp16, int8=self.int8),
        )
      
    return DeviceCapabilities(
      model="Unknown Device",
      chip="Unknown Chip",
      memory=0 // 2**20,
      flops=DeviceFlops(fp32=0, fp16=0, int8=0),
    )
  
  