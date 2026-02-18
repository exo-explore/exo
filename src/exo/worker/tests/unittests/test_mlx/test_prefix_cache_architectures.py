import copy
import gc
import importlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.generator.generate import mlx_generate
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    load_tokenizer_for_model_id,
)

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# ── Config reduction ──────────────────────────────────────────────────────── #

_REDUCE = {
    "num_hidden_layers": 4,
    "hidden_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 512,
    "moe_intermediate_size": 128,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "n_routed_experts": 4,
    "num_local_experts": 4,
    "num_nextn_predict_layers": 0,
    "first_k_dense_replace": 0,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 2,
    "num_attention_groups": 4,
}


def _reduce_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    result = dict(cfg)
    for key, val in _REDUCE.items():
        if key in result:
            result[key] = val
    return result


def _reduce_config(cfg: dict[str, Any]) -> dict[str, Any]:
    result = _reduce_dict(cfg)
    n_layers = cast(int, result.get("num_hidden_layers", 4))

    if "text_config" in result and isinstance(result["text_config"], dict):
        result["text_config"] = _reduce_dict(
            cast(dict[str, Any], result["text_config"])
        )
        tc: dict[str, Any] = result["text_config"]
        if "num_nextn_predict_layers" in tc:
            tc["num_nextn_predict_layers"] = 0

    if "layer_types" in result and isinstance(result["layer_types"], list):
        result["layer_types"] = result["layer_types"][:n_layers]

    if "attention_other_setting" in result and isinstance(
        result["attention_other_setting"], dict
    ):
        aos: dict[str, Any] = dict(
            cast(dict[str, Any], result["attention_other_setting"])
        )
        if "num_attention_heads" in aos:
            aos["num_attention_heads"] = result.get("num_attention_heads", 4)
        if "num_attention_groups" in aos:
            aos["num_attention_groups"] = result.get(
                "num_attention_groups", cast(int, aos["num_attention_groups"])
            )
        result["attention_other_setting"] = aos

    if "moe_layers_enum" in result and isinstance(result["moe_layers_enum"], str):
        indices = [int(x) for x in result["moe_layers_enum"].split(",") if x.strip()]
        valid = [i for i in indices if i < n_layers]
        result["moe_layers_enum"] = ",".join(str(i) for i in valid) if valid else ""

    return result


# ── Helpers ───────────────────────────────────────────────────────────────── #


def _find_snapshot(hub_name: str) -> Path | None:
    model_dir = HF_CACHE / f"models--mlx-community--{hub_name}"
    snaps = model_dir / "snapshots"
    if not snaps.exists():
        return None
    children = sorted(snaps.iterdir())
    return children[0] if children else None


def _copy_tokenizer(src: Path, dst: Path) -> None:
    for f in src.iterdir():
        name = f.name
        if (
            "tokeniz" in name.lower()
            or "tiktoken" in name.lower()
            or name.startswith("vocab")
            or name.endswith(".jinja")
            or "tool_declaration" in name
        ) and f.is_file():
            shutil.copy2(f, dst / name)


def _build_model(module_name: str, cfg: dict[str, Any]) -> Model:
    mod = importlib.import_module(f"mlx_lm.models.{module_name}")
    args = mod.ModelArgs.from_dict(cfg)  # pyright: ignore[reportAny]
    model: nn.Module = mod.Model(args)  # pyright: ignore[reportAny]
    flat = cast(list[tuple[str, mx.array]], tree_flatten(model.parameters()))
    random_weights = [
        (k, mx.random.normal(shape=v.shape, dtype=mx.float16)) for k, v in flat
    ]
    model.update(cast(dict[str, Any], tree_unflatten(random_weights)))
    mx.eval(model.parameters())
    return cast(Model, model)


def _collect_tokens(
    model: Model,
    tokenizer: TokenizerWrapper,
    task: TextGenerationTaskParams,
    prompt: str,
    kv_prefix_cache: KVPrefixCache | None,
) -> list[int]:
    tokens: list[int] = []
    for resp in mlx_generate(
        model=model,
        tokenizer=tokenizer,
        task=task,
        prompt=prompt,
        kv_prefix_cache=kv_prefix_cache,
        group=None,
    ):
        tokens.append(resp.token)
        if resp.finish_reason is not None:
            break
    return tokens


# ── Architecture definitions ──────────────────────────────────────────────── #


@dataclass(frozen=True)
class ArchSpec:
    name: str
    hub_name: str
    module: str
    tokenizer_hub: str | None = None  # fallback for models without bundled tokenizer


ARCHITECTURES: list[ArchSpec] = [
    ArchSpec("llama", "Llama-3.2-1B-Instruct-4bit", "llama"),
    ArchSpec("glm_moe_dsa", "GLM-5-MXFP4-Q8", "glm_moe_dsa"),
    ArchSpec(
        "glm4_moe", "GLM-4.5-Air-8bit", "glm4_moe", tokenizer_hub="GLM-4.7-8bit-gs32"
    ),
    ArchSpec(
        "glm4_moe_lite",
        "GLM-4.7-Flash-8bit",
        "glm4_moe_lite",
        tokenizer_hub="GLM-4.7-8bit-gs32",
    ),
    ArchSpec("glm4_moe_47", "GLM-4.7-8bit-gs32", "glm4_moe"),
    ArchSpec("qwen3", "Qwen3-4B-Instruct-2507-4bit", "qwen3"),
    ArchSpec("qwen3_moe", "Qwen3-30B-A3B-4bit", "qwen3_moe"),
    ArchSpec("qwen3_next", "Qwen3-Next-80B-A3B-Thinking-4bit", "qwen3_next"),
    ArchSpec("minimax", "MiniMax-M2.1-3bit", "minimax"),
    ArchSpec("gpt_oss", "gpt-oss-20b-MXFP4-Q8", "gpt_oss"),
    ArchSpec("step3p5", "Step-3.5-Flash-4bit", "step3p5"),
    ArchSpec("kimi_k25", "Kimi-K2.5", "kimi_k25"),
]


def _arch_available(spec: ArchSpec) -> bool:
    snap = _find_snapshot(spec.hub_name)
    if snap is None:
        return False
    if spec.tokenizer_hub is not None:
        return _find_snapshot(spec.tokenizer_hub) is not None
    return True


def _make_task() -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("test"),
        input=[
            InputMessage(
                role="user",
                content="Use the calculator to compute 1847 * 263 + 5921",
            )
        ],
        max_output_tokens=20,
        temperature=0.0,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate a mathematical expression",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }
        ],
    )


# ── Test class ────────────────────────────────────────────────────────────── #


@pytest.mark.slow
class TestPrefixCacheArchitectures:
    """Verify prefix cache produces identical output to fresh generation for every architecture."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        mx.clear_cache()
        gc.collect()

    @pytest.mark.parametrize(
        "spec",
        ARCHITECTURES,
        ids=[a.name for a in ARCHITECTURES],
    )
    def test_prefix_cache_exact_hit(self, spec: ArchSpec) -> None:
        if not _arch_available(spec):
            pytest.skip(f"Model {spec.hub_name} not cached locally")

        snapshot = _find_snapshot(spec.hub_name)
        assert snapshot is not None

        tmpdir = Path(tempfile.mkdtemp(prefix=f"exo_test_{spec.name}_"))
        try:
            # Build reduced config
            with open(snapshot / "config.json") as f:
                cfg = cast(dict[str, Any], json.load(f))
            reduced = _reduce_config(copy.deepcopy(cfg))
            (tmpdir / "config.json").write_text(json.dumps(reduced))

            # Copy tokenizer
            tok_src = snapshot
            if spec.tokenizer_hub is not None:
                alt = _find_snapshot(spec.tokenizer_hub)
                if alt is not None:
                    tok_src = alt
            _copy_tokenizer(tok_src, tmpdir)

            # Load tokenizer and model
            model_id = ModelId(f"mlx-community/{spec.hub_name}")
            tokenizer = load_tokenizer_for_model_id(model_id, tmpdir)
            mx.random.seed(0)
            model = _build_model(spec.module, reduced)

            task = _make_task()
            prompt = apply_chat_template(tokenizer=tokenizer, task_params=task)

            # Run 1: fresh
            mx.random.seed(42)
            fresh = _collect_tokens(model, tokenizer, task, prompt, None)
            assert len(fresh) > 0, "Fresh generation produced no tokens"

            # Run 2: populate cache
            kv = KVPrefixCache(None)
            mx.random.seed(42)
            populate = _collect_tokens(model, tokenizer, task, prompt, kv)

            # Run 3: exact cache hit
            mx.random.seed(42)
            cached = _collect_tokens(model, tokenizer, task, prompt, kv)

            assert fresh == populate, (
                f"Fresh vs populate mismatch: {fresh[:5]} vs {populate[:5]}"
            )
            assert fresh == cached, (
                f"Fresh vs cached mismatch: {fresh[:5]} vs {cached[:5]}"
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
