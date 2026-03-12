from pydantic import BaseModel

class LoadLoRAAdapterRequest(BaseModel):
    lora_name: str
    lora_path: str
    load_inplace: bool

class UnloadLoRAAdapterRequest(BaseModel):
    lora_name: str
    lora_int_id: int | None
