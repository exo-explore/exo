from vllm.config import ModelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike

class LLMEngine:
    tokenizer: TokenizerLike | None
    model_config: ModelConfig

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> LLMEngine: ...
    def add_request(
        self,
        request_id: str,
        prompt: str,
        params: SamplingParams,
        arrival_time: float | None = ...,
    ) -> None: ...
    def step(self) -> list[RequestOutput]: ...
    def has_unfinished_requests(self) -> bool: ...
    def get_tokenizer(self) -> TokenizerLike: ...
