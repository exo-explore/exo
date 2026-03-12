from collections.abc import Callable as Callable, Iterable, Sequence
from tqdm.auto import tqdm
from typing import Any, overload

@overload
def maybe_tqdm(
    it: Sequence[_T], *, use_tqdm: bool | Callable[..., tqdm], **tqdm_kwargs: Any
) -> Sequence[_T]: ...
@overload
def maybe_tqdm(
    it: Iterable[_T], *, use_tqdm: bool | Callable[..., tqdm], **tqdm_kwargs: Any
) -> Iterable[_T]: ...
