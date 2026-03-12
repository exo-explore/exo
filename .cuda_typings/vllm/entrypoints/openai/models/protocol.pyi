from dataclasses import dataclass

@dataclass
class BaseModelPath:
    name: str
    model_path: str

@dataclass
class LoRAModulePath:
    name: str
    path: str
    base_model_name: str | None = ...
