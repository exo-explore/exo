from typing import Any, Generator

_patch_applied = False


def patch_mlx_lm_for_distributed() -> None:
    """
    Patches mlx_lm's generate_step to work with distributed inference.

    mlx_lm's prefill loop only evaluates cache state, not logits.
    With distributed inference, model() triggers mx.distributed.all_gather() which must be
    evaluated for all devices to synchronize. When prompt > prefill_step_size, the
    all_gather is never evaluated, causing GPU timeout.

    This patch uses mx.depends to make cache state depend on logits, ensuring all_gather is
    evaluated when cache is eval'd.
    """
    global _patch_applied
    if _patch_applied:
        return
    _patch_applied = True

    import importlib

    import mlx.core as mx

    gen_module = importlib.import_module("mlx_lm.generate")
    original_generate_step = gen_module.generate_step  # pyright: ignore[reportAny]

    def patched_generate_step(
        prompt: mx.array,
        model: Any,  # pyright: ignore[reportAny]
        **kwargs: Any,  # pyright: ignore[reportAny]
    ) -> Generator[Any, None, None]:
        """Patched generate_step that works with distributed inference."""
        prompt_cache = kwargs.get("prompt_cache")

        class DistributedModelWrapper:
            """Wrapper that adds mx.depends between logits and cache state."""

            def __init__(
                self,
                inner_model: Any,  # pyright: ignore[reportAny]
                cache: Any,  # pyright: ignore[reportAny]
            ) -> None:
                self._inner: Any = inner_model
                self._cache: Any = cache

            def __call__(
                self,
                *args: Any,  # pyright: ignore[reportAny]
                **kw: Any,  # pyright: ignore[reportAny]
            ) -> mx.array:
                logits: mx.array = self._inner(*args, **kw)  # pyright: ignore[reportAny]
                cache: Any = kw.get("cache") or self._cache  # pyright: ignore[reportAny]
                if cache is not None:
                    for c in cache:  # pyright: ignore[reportAny]
                        if hasattr(c, "state") and c.state is not None:  # pyright: ignore[reportAny]
                            c.state = mx.depends(c.state, logits)  # pyright: ignore[reportAny, reportUnknownMemberType]
                return logits

            def __getattr__(self, name: str) -> Any:  # pyright: ignore[reportAny]
                return getattr(self._inner, name)  # pyright: ignore[reportAny]

        if prompt_cache is None:
            prompt_cache = model.make_cache()  # pyright: ignore[reportAny]
            kwargs["prompt_cache"] = prompt_cache

        wrapped_model = DistributedModelWrapper(model, prompt_cache)
        yield from original_generate_step(prompt, wrapped_model, **kwargs)

    gen_module.generate_step = patched_generate_step  # pyright: ignore[reportAttributeAccessIssue]
