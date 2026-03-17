from dataclasses import dataclass

@dataclass
class KVCacheSpec:
    block_size: int
    num_kv_heads: int
    head_size: int

@dataclass
class KVCacheGroupSpec:
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec

@dataclass
class KVCacheTensorSpec:
    shared_by: list[str]
    size: int

@dataclass
class KVCacheConfig:
    num_blocks: int
    kv_cache_groups: list[KVCacheGroupSpec]
    kv_cache_tensors: list[KVCacheTensorSpec]
