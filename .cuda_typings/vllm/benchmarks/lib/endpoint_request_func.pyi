import aiohttp
from _typeshed import Incomplete
from collections.abc import Awaitable
from dataclasses import dataclass, field
from tqdm.asyncio import tqdm as tqdm
from typing import Literal, Protocol

AIOHTTP_TIMEOUT: Incomplete

class StreamedResponseHandler:
    buffer: str
    def __init__(self) -> None: ...
    def add_chunk(self, chunk_bytes: bytes) -> list[str]: ...

@dataclass
class RequestFuncInput:
    prompt: str | list[str]
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: str | None = ...
    logprobs: int | None = ...
    extra_headers: dict | None = ...
    extra_body: dict | None = ...
    multi_modal_content: dict | list[dict] | None = ...
    ignore_eos: bool = ...
    language: str | None = ...
    request_id: str | None = ...

@dataclass
class RequestFuncOutput:
    generated_text: str = ...
    success: bool = ...
    latency: float = ...
    output_tokens: int = ...
    ttft: float = ...
    itl: list[float] = field(default_factory=list)
    tpot: float = ...
    prompt_len: int = ...
    error: str = ...
    start_time: float = ...
    input_audio_duration: float = ...

class RequestFunc(Protocol):
    def __call__(
        self,
        request_func_input: RequestFuncInput,
        session: aiohttp.ClientSession,
        pbar: tqdm | None = None,
    ) -> Awaitable[RequestFuncOutput]: ...

async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput: ...
async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
    mm_position: Literal["first", "last"] = "last",
) -> RequestFuncOutput: ...
async def async_request_openai_audio(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput: ...
async def async_request_openai_embeddings(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput: ...
async def async_request_vllm_rerank(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput: ...
async def async_request_openai_embeddings_chat(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
    mm_position: Literal["first", "last"] = "last",
) -> RequestFuncOutput: ...
async def async_request_openai_embeddings_clip(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput: ...
async def async_request_openai_embeddings_vlm2vec(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput: ...
async def async_request_infinity_embeddings(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput: ...
async def async_request_infinity_embeddings_clip(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput: ...
async def async_request_vllm_pooling(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput: ...

ASYNC_REQUEST_FUNCS: dict[str, RequestFunc]
POOLING_BACKENDS: Incomplete
OPENAI_COMPATIBLE_BACKENDS: Incomplete
