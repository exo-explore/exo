from tinygrad import Tensor

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
            self._keys[layers_idx] = self._keys[layers_idx].cat(key, dim = 2)
            self._values[layers_idx] = self._values[layers_idx].cat(value, dim = 2)

        return self._keys[layers_idx], self._values[layers_idx]

    @property
    def seq_len(self) -> int:
        return 0 if self._keys[0] is None else self._keys[0].shape[2]
