from pathlib import Path

from tinygrad.tensor import Tensor

def safe_load(fn: Tensor | str | Path) -> dict[str, Tensor]: ...
def safe_load_metadata(
    fn: Tensor | str | Path,
) -> tuple[Tensor, int, dict[str, object]]: ...
def safe_save(
    tensors: dict[str, Tensor],
    fn: str,
    metadata: dict[str, object] | None = None,
) -> None: ...
def load_state_dict(
    model: object,
    state_dict: dict[str, Tensor],
    strict: bool = True,
    verbose: bool = True,
    consume: bool = False,
    realize: bool = True,
) -> list[Tensor]: ...
