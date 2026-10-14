import pathlib
import tempfile

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

cache = ArraysCache(2, left_padding=[2])
cache.prepare(lengths=[3])
cache[0] = mx.array([0])
cache[1] = mx.array([0])
for _step in range(256):
    cache[0] = cache[0] + 1
    cache.advance(1)
    mx.eval(cache[0])

with tempfile.TemporaryDirectory() as directory:
    for name, metadata in (
        ("lengths", cache.lengths),
        ("left-padding", cache.left_padding),
    ):
        path = pathlib.Path(directory, f"arrays-cache-{name}.dot")
        mx.export_to_dot(str(path), metadata)
        edges = path.read_text(encoding="utf-8").count("->")
        if edges > 8:
            raise AssertionError(f"{name} graph has {edges} edges, expected <= 8")

assert cache[0].item() == 256
assert cache.lengths.item() == 3 - 256
assert cache.left_padding.item() == 2 - 256
print("arrays-cache metadata regression: PASS")
