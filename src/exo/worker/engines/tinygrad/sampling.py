from typing import NamedTuple

from tinygrad.tensor import Tensor


class SampleResult(NamedTuple):
    token_id: int
    logprob: float
    top_logprobs: list[tuple[int, float]]   # (token_id, logprob)

def sample_token(
    logits: Tensor,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_logprobs_count: int = 0,
    request_logprobs: bool = False,
) -> SampleResult:
    last_logits = logits[0, -1, :]

    if temperature == 0:
        token_id = int(last_logits.argmax().item())  # pyright: ignore[reportUnknownMemberType]
    else:
        # Gumble-max trick for reducing GPU -> CPU sync.

        scaled = last_logits / temperature
        gumble_noise = -(-Tensor.rand(scaled.shape).log()).log()  # pyright: ignore[reportUnknownMemberType]
        token_id = int((scaled + gumble_noise).argmax().item())  # pyright: ignore[reportUnknownMemberType]

    selected_logprob: float = 0.0
    top_logprobs: list[tuple[int, float]] = []

    if request_logprobs:
        log_probs = last_logits.log_softmax(axis = -1)
        selected_logprob = float(log_probs[token_id].item())

        if top_logprobs_count > 0:
            values, indices = log_probs.topk(top_logprobs_count)
            top_logprobs = [(int(idx), float(val)) for val, idx in zip(values.tolist(), indices.tolist(), strict=True)]  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType, reportArgumentType]

    return SampleResult(
        token_id=token_id,
        logprob=selected_logprob,
        top_logprobs=top_logprobs,
    )
