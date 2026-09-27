class DType:
    name: str
    itemsize: int


class _DTypes:
    float16: DType
    bfloat16: DType
    float32: DType
    int32: DType
    uint8: DType
    bool: DType


dtypes: _DTypes
