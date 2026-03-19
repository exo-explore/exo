class SamplingParams:
    n: int
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    seed: int | None
    stop: str | list[str] | None
    max_tokens: int | None
    logprobs: int | None
    repetition_penalty: float
