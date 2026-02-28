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
        probs = (last_logits / temperature).softmax(axis = -1)

        r = Tensor.rand(1)  # pyright: ignore[reportUnknownMemberType]
        cumsum = probs.cumsum(axis = 0)
        mask = (cumsum > r).float()
        token_id = int(mask.argmax().item())  # pyright: ignore[reportUnknownMemberType]

    selected_logprob: float = 0.0
    top_logprobs: list[tuple[int, float]] = []

    if request_logprobs:
        log_probs = last_logits.log_softmax(axis = -1)
        log_probs_list: list[float] = log_probs.tolist()  # pyright: ignore[reportAssignmentType]
        selected_logprob = log_probs_list[token_id]

        if top_logprobs_count > 0:
            values, indices = log_probs.topk(top_logprobs_count)
            top_logprobs = [(int(idx), float(val)) for val, idx in zip(values.tolist(), indices.tolist())]

    return SampleResult(
        token_id=token_id,
        logprob=selected_logprob,
        top_logprobs=top_logprobs,
    )
