from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor


class KVCache:
    def __init__(self, 
                 num_layers: int, 
                 num_kv_heads: int, 
                 head_dim: int, 
                 max_seq_len: int,
                 ) -> None:
        self.keys: list[Tensor] = [Tensor.zeros(1, num_kv_heads, max_seq_len, head_dim, dtype=dtypes.float16) for _ in range(num_layers)]  # pyright: ignore[reportUnknownMemberType]
        self.values: list[Tensor] = [Tensor.zeros(1, num_kv_heads, max_seq_len, head_dim, dtype=dtypes.float16) for _ in range(num_layers)]  # pyright: ignore[reportUnknownMemberType]

        self.max_seq_len = max_seq_len
        self._positions: Tensor = Tensor.arange(max_seq_len).reshape(1, 1, max_seq_len, 1)  # pyright: ignore[reportUnknownMemberType]
        self.col_indices: Tensor = Tensor.arange(max_seq_len).reshape(1, 1, 1, max_seq_len)  # pyright: ignore[reportUnknownMemberType]

    def update(
        self,
        layers_idx: int,
        key: Tensor,
        value: Tensor,
        position: int | Tensor = 0,
    ) -> tuple[Tensor, Tensor]:
        seq_len = key.shape[2]

        """
            Mask: For positions:
            [self.position, self.position + seq_len]

            We are using mask + pad since tinygrad tensor
            cannot handle slice assignment. 
        """

        positions = self._positions

        if isinstance(position, Tensor):
            mask = positions == position
            self.keys[layers_idx] = Tensor.where(
                mask, key.half(), self.keys[layers_idx]
            )
            self.values[layers_idx] = Tensor.where(
                mask, value.half(), self.values[layers_idx]
            )
        else:
            mask = (positions >= position) & (positions < position + seq_len)
            pad_prev = position
            pad_next = self.max_seq_len - position - seq_len
            new_k = key.pad(
                ((0, 0), (0, 0), (pad_prev, pad_next), (0, 0))
            ).half()
            new_v = value.pad(
                ((0, 0), (0, 0), (pad_prev, pad_next), (0, 0))
            ).half()

            self.keys[layers_idx] = new_k
            self.values[layers_idx] = new_v

            if position > 0:
                self.keys[layers_idx] = Tensor.where(
                    mask, new_k, self.keys[layers_idx]
                )

                self.values[layers_idx] = Tensor.where(
                    mask, new_v, self.values[layers_idx]
                )

        return self.keys[layers_idx], self.values[layers_idx]

    @property
    def seq_len(self) -> int:
        return int(self.keys[0].shape[2])
