from dataclasses import dataclass

@dataclass
class Shard:
    model_id: str
    n_layers: int
    start_layer: int
    end_layer: int
