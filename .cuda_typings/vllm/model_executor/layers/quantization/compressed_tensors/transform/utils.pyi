from compressed_tensors.transform import TransformArgs, TransformScheme
from typing import NamedTuple

__all__ = ["TransformTuple"]

class TransformTuple(NamedTuple):
    scheme_name: str
    scheme: TransformScheme
    args: TransformArgs
