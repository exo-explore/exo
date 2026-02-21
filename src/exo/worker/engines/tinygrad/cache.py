from tinygrad.tensor import Tensor


class KVCache:
    def __init__(self, num_layers: int) -> None:
        self._keys: list[Tensor | None] = [None] * num_layers
        self._values: list[Tensor | None] = [None] * num_layers

    def update(
        self,
        layers_idx: int,
        key: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self._keys[layers_idx] is None:
            self._keys[layers_idx] = key
            self._values[layers_idx] = value
        else:
            existing_k = self._keys[layers_idx]
            existing_v = self._values[layers_idx]
            assert existing_k is not None and existing_v is not None
            self._keys[layers_idx] = existing_k.cat(key, dim=2)
            self._values[layers_idx] = existing_v.cat(value, dim=2)

        result_k = self._keys[layers_idx]
        result_v = self._values[layers_idx]
        assert result_k is not None and result_v is not None
        return result_k, result_v

    @property
    def seq_len(self) -> int:
        return 0 if self._keys[0] is None else int(self._keys[0].shape[2])
