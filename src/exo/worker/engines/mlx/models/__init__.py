"""One folder per model family.

Each family folder holds everything exo needs for that family beyond what mlx-lm
provides: tensor-parallel sharding, pipeline fixups, prompt encoders, output parsers,
tokenizer quirks and cache hooks. Shared mechanisms live in
``exo.worker.engines.mlx.sharding`` and must not name a model.
"""
