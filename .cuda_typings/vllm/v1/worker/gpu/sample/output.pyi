import torch
from dataclasses import dataclass
from vllm.v1.outputs import LogprobsTensors as LogprobsTensors

@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None
    num_nans: torch.Tensor | None
