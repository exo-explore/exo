from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.helpers import DEBUG
import socket
import yaml

class ReadManualConfig():
  def __init__(
    self,
    discovery_config: str = "topology.yml",
  ):
    self._discovery_config = discovery_config
    self._config_devices = None
    self._whoami = f"{socket.gethostname()}"
    self._node_id = "NONE"
    self._node_host = "NONE"
    self._node_port = "NONE"
    self._model = "NONE"
    self._chip = "NONE"
    self._memory = "NONE"
    self._fp32 = "NONE"
    self._fp16 = "NONE"
    self._int8 = "NONE"

  def device_capabilities(self, gethostname) -> DeviceCapabilities:
    with open(self._discovery_config, 'r') as f:
      self._config_devices = yaml.safe_load(f)
      f.close()

    for device in self._config_devices:
      if (str(f"{gethostname}")) == (str(f"{device['server']}")):
        if DEBUG >= 2: print(f"Read Id {device['id']} == Adresse: {device['address']} {device['port']}")
        self._node_id = (str(f"{device['id']}"))
        self._node_host = (str(f"{device['address']}"))
        self._node_port = (int(f"{device['port']}"))
        if DEBUG >= 2: print(f"Capabilities:")
        for capability, value in device['device_capabilities'].items():
          if DEBUG >= 2: print(f"{capability}: {value}")
          if f"{capability}" == "model":
            self._model = (str(f"{value}"))
          if f"{capability}" == "chip":
            self._chip = (str(f"{value}"))
          if f"{capability}" == "memory":
            self._memory = (float(f"{value}"))
          if f"{capability}" == "flops":
            for flopstr, flopvalue in device['device_capabilities']['flops'].items():
                if f"{flopstr}" == "fp32":
                  self._fp32 = (float(f"{flopvalue}"))
                if f"{flopstr}" == "fp16":
                  self._fp16 = (float(f"{flopvalue}"))
                if f"{flopstr}" == "int8":
                  self._int8 = (float(f"{flopvalue}"))

        return DeviceCapabilities(
          model=self._model,
          chip=self._chip,
          memory=self._memory // 2**20,
          flops=DeviceFlops(fp32=self._fp32, fp16=self._fp16, int8=self._int8),
        )
      
    return DeviceCapabilities(
      model="Unknown Device",
      chip="Unknown Chip",
      memory=0 // 2**20,
      flops=DeviceFlops(fp32=0, fp16=0, int8=0),
    )
  
  @property
  def discovery_config(self):
    return self._discovery_config
  
  @property
  def config_devices(self):
    return self._config_devices

  @property
  def whoami(self):
    return self._whoami

  @property
  def node_id(self):
    return self._node_id

  @property
  def node_host(self):
    return self._node_host

  @property
  def node_port(self):
    return self._node_port

  @property
  def model(self):
    return self._model

  @property
  def chip(self):
    return self._chip

  @property
  def memory(self):
    return self._memory

  @property
  def fp32(self):
    return self._fp32

  @property
  def fp16(self):
    return self._fp16

  @property
  def int8(self):
    return self._int8
  
