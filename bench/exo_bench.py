# type: ignore
#!/usr/bin/env python3
"""Tool-calling eval for exo's OpenAI-compatible API.

Tests whether models correctly:
- Trigger tool calls when appropriate
- Return valid JSON arguments matching function schemas
- Handle multi-turn tool use (call -> result -> final answer)
- Avoid calling tools when unnecessary

Start exo with a model first, then run:
    uv run python tool_call_eval.py --model <model-id>
    uv run python tool_call_eval.py --model <model-id> --host 10.0.0.5 --port 52415
    uv run python tool_call_eval.py --model <model-id> --repeat 3
    uv run python tool_call_eval.py --model <model-id> --scenarios weather_simple calculator_multi_turn
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any

from harness import (
    ExoClient,
    ExoHttpError,
    add_common_instance_args,
    instance_id_from_instance,
    nodes_used_in_instance,
    resolve_model_short_id,
    run_planning_phase,
    settle_and_fetch_placements,
    wait_for_instance_gone,
    wait_for_instance_ready,
)
from loguru import logger
from transformers import AutoTokenizer

# Monkey-patch for transformers 5.x compatibility
# Kimi's tokenization_kimi.py imports bytes_to_unicode from the old location
# which was moved in transformers 5.0.0rc2
try:
    import transformers.models.gpt2.tokenization_gpt2 as gpt2_tokenization
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    if not hasattr(gpt2_tokenization, "bytes_to_unicode"):
        gpt2_tokenization.bytes_to_unicode = bytes_to_unicode  # type: ignore[attr-defined]
except ImportError:
    pass  # transformers < 5.0 or bytes_to_unicode not available


def load_tokenizer_for_bench(model_id: str) -> Any:
    """
    Load tokenizer for benchmarking, with special handling for Kimi models.

    Kimi uses a custom TikTokenTokenizer that transformers 5.x can't load via AutoTokenizer.
    This function replicates the logic from utils_mlx.py for bench compatibility.
    """
    model_id_lower = model_id.lower()

    if "kimi-k2" in model_id_lower:
        import importlib.util
        import types

        from huggingface_hub import snapshot_download

        # Download/get the model path
        model_path = Path(
            snapshot_download(
                model_id,
                allow_patterns=["*.json", "*.py", "*.tiktoken", "*.model"],
            )
        )

        sys.path.insert(0, str(model_path))

        # Load tool_declaration_ts first (tokenization_kimi imports it with relative import)
        tool_decl_path = model_path / "tool_declaration_ts.py"
        if tool_decl_path.exists():
            spec = importlib.util.spec_from_file_location(
                "tool_declaration_ts", tool_decl_path
            )
            if spec and spec.loader:
                tool_decl_module = importlib.util.module_from_spec(spec)
                sys.modules["tool_declaration_ts"] = tool_decl_module
                spec.loader.exec_module(tool_decl_module)

        # Load tokenization_kimi with patched source (convert relative to absolute import)
        tok_path = model_path / "tokenization_kimi.py"
        source = tok_path.read_text()
        source = source.replace("from .tool_declaration_ts", "from tool_declaration_ts")
        spec = importlib.util.spec_from_file_location("tokenization_kimi", tok_path)
        if spec:
            tok_module = types.ModuleType("tokenization_kimi")
            tok_module.__file__ = str(tok_path)
            sys.modules["tokenization_kimi"] = tok_module
            exec(compile(source, tok_path, "exec"), tok_module.__dict__)  # noqa: S102
            TikTokenTokenizer = tok_module.TikTokenTokenizer  # noqa: N806
        else:
            from tokenization_kimi import TikTokenTokenizer  # type: ignore[import-not-found]  # noqa: I001

        hf_tokenizer: Any = TikTokenTokenizer.from_pretrained(model_path)

        # Patch encode to use internal tiktoken model directly
        # transformers 5.x has a bug in the encode->pad path for slow tokenizers
        def _patched_encode(text: str, **kwargs: object) -> list[int]:
            # Pass allowed_special="all" to handle special tokens like <|im_user|>
            return list(hf_tokenizer.model.encode(text, allowed_special="all"))

        hf_tokenizer.encode = _patched_encode

        return hf_tokenizer

    # Default: use AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def format_peak_memory(b: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024.0:
            return f"{b:.2f}{unit}"
        b /= 1024.0
    raise ValueError("You're using petabytes of memory. Something went wrong...")


def parse_int_list(values: list[str]) -> list[int]:
    items: list[int] = []
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if part:
                items.append(int(part))
    return items


def run_one_completion(
    client: ExoClient, model_id: str, pp_hint: int, tg: int, prompt_sizer: PromptSizer
) -> tuple[dict[str, Any], int]:
    content, pp_tokens = prompt_sizer.build(pp_hint)
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "max_tokens": tg,
    }

    t0 = time.perf_counter()
    out = client.post_bench_chat_completions(payload)
    elapsed = time.perf_counter() - t0

    stats = out.get("generation_stats")

    # Extract preview, handling None content (common for thinking models)
    choices = out.get("choices") or [{}]
    message = choices[0].get("message", {}) if choices else {}
    content = message.get("content") or ""
    preview = content[:200] if content else ""

    return {
        "elapsed_s": elapsed,
        "output_text_preview": preview,
        "stats": stats,
    }, pp_tokens


class PromptSizer:
    def __init__(self, tokenizer: Any, atom: str = "a "):
        self.tokenizer = tokenizer
        self.atom = atom
        self.count_fn = PromptSizer._make_counter(tokenizer)
        self.base_tokens = self.count_fn("")

    @staticmethod
    def _make_counter(tokenizer: Any) -> Callable[[str], int]:
        def count_fn(user_content: str) -> int:
            messages = [{"role": "user", "content": user_content}]
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            # Fix for transformers 5.x
            if hasattr(ids, "input_ids"):
                ids = ids.input_ids
            return int(len(ids))

        return count_fn

    def build(self, target_prompt_tokens: int) -> tuple[str, int]:
        target = int(target_prompt_tokens)
        if target < self.base_tokens:
            raise RuntimeError(
                f"Target ({target}) is smaller than template overhead ({self.base_tokens})."
            )

        # Estimate tokens per atom using a sample
        sample_count = 100
        sample_content = self.atom * sample_count
        sample_tokens = self.count_fn(sample_content) - self.base_tokens
        tokens_per_atom = sample_tokens / sample_count

        # Estimate starting point
        needed_tokens = target - self.base_tokens
        estimated_atoms = int(needed_tokens / tokens_per_atom)

        # Binary search to find exact atom count
        low, high = 0, estimated_atoms * 2 + 100
        while low < high:
            mid = (low + high) // 2
            tok = self.count_fn(self.atom * mid)
            if tok < target:
                low = mid + 1
            else:
                high = mid

        content = self.atom * low
        tok = self.count_fn(content)
        logger.info(f"{tok=}")

        if tok != target:
            raise RuntimeError(
                f"Overshot: got {tok} tokens (target {target}). "
                f"Pick a different atom (try ' a' or '\\n' or '0 ')."
            )

        return content, tok


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="exo-bench",
        description="Benchmark exo model throughput across placement previews.",
    )
    add_common_instance_args(ap)
    ap.add_argument(
        "--pp",
        nargs="+",
        required=True,
        help="Prompt-size hints (ints). Accepts commas.",
    )
    ap.add_argument(
        "--tg",
        nargs="+",
        required=True,
        help="Generation lengths (ints). Accepts commas.",
    )
    ap.add_argument(
        "--repeat", type=int, default=1, help="Repetitions per (pp,tg) pair."
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warmup runs per placement (uses first pp/tg).",
    )
    ap.add_argument(
        "--json-out",
        default="bench/results.json",
        help="Write raw per-run results JSON to this path.",
    )
    ap.add_argument("--stdout", action="store_true", help="Write results to stdout")
    ap.add_argument(
        "--dry-run", action="store_true", help="List selected placements and exit."
    )
    ap.add_argument(
        "--all-combinations",
        action="store_true",
        help="Force all pp×tg combinations (cartesian product) even when lists have equal length.",
    )
    args = ap.parse_args()

    pp_list = parse_int_list(args.pp)
    tg_list = parse_int_list(args.tg)
    if not pp_list or not tg_list:
        logger.error("pp and tg lists must be non-empty")
        return 2
    if args.repeat <= 0:
        logger.error("--repeat must be >= 1")
        return 2

    # Log pairing mode
    use_combinations = args.all_combinations or len(pp_list) != len(tg_list)
    if use_combinations:
        logger.info(
            f"pp/tg mode: combinations (product) - {len(pp_list) * len(tg_list)} pairs"
        )
    else:
        logger.info(f"pp/tg mode: tandem (zip) - {len(pp_list)} pairs")

    client = ExoClient(args.host, args.port, timeout_s=args.timeout)
    short_id, full_model_id = resolve_model_short_id(client, args.model)

    tokenizer = load_tokenizer_for_bench(full_model_id)
    if tokenizer is None:
        raise RuntimeError("[exo-bench] tokenizer load failed")

    try:
        prompt_sizer = PromptSizer(tokenizer)
        logger.debug(f"[exo-bench] loaded tokenizer: {full_model_id} for prompt sizer")
    except Exception:
        logger.error("[exo-bench] tokenizer usable but prompt sizing failed")
        raise

    selected = settle_and_fetch_placements(
        client, full_model_id, args, settle_timeout=args.settle_timeout
    )

    if not selected:
        logger.error("No valid placements matched your filters.")
        return 1

    selected.sort(
        key=lambda p: (
            str(p.get("instance_meta", "")),
            str(p.get("sharding", "")),
            -nodes_used_in_instance(p["instance"]),
        ),
        reverse=True,
    )

    logger.debug(f"exo-bench model: short_id={short_id} full_id={full_model_id}")
    logger.info(f"placements: {len(selected)}")
    for p in selected:
        logger.info(
            f"  - {p['sharding']} / {p['instance_meta']} / nodes={nodes_used_in_instance(p['instance'])}"
        )

    if args.dry_run:
        return 0

    settle_deadline = (
        time.monotonic() + args.settle_timeout if args.settle_timeout > 0 else None
    )

    logger.info("Planning phase: checking downloads...")
    download_duration_s = run_planning_phase(
        client,
        full_model_id,
        selected[0],
        args.danger_delete_downloads,
        args.timeout,
        settle_deadline,
    )
    if download_duration_s is not None:
        logger.info(f"Download: {download_duration_s:.1f}s (freshly downloaded)")
    else:
        logger.info("Download: model already cached")

    all_rows: list[dict[str, Any]] = []

    for preview in selected:
        instance = preview["instance"]
        instance_id = instance_id_from_instance(instance)

        sharding = str(preview["sharding"])
        instance_meta = str(preview["instance_meta"])
        n_nodes = nodes_used_in_instance(instance)

        logger.info("=" * 80)
        logger.info(
            f"PLACEMENT: {sharding} / {instance_meta} / nodes={n_nodes} / instance_id={instance_id}"
        )

        client.request_json("POST", "/instance", body={"instance": instance})
        try:
            wait_for_instance_ready(client, instance_id)
        except (RuntimeError, TimeoutError) as e:
            logger.error(f"Failed to initialize placement: {e}")
            with contextlib.suppress(ExoHttpError):
                client.request_json("DELETE", f"/instance/{instance_id}")
            continue

        time.sleep(1)

        try:
            for i in range(args.warmup):
                run_one_completion(
                    client, full_model_id, pp_list[0], tg_list[0], prompt_sizer
                )
                logger.debug(f"  warmup {i + 1}/{args.warmup} done")

            # If pp and tg lists have same length, run in tandem (zip)
            # Otherwise (or if --all-combinations), run all combinations (cartesian product)
            if use_combinations:
                pp_tg_pairs = list(itertools.product(pp_list, tg_list))
            else:
                pp_tg_pairs = list(zip(pp_list, tg_list, strict=True))

            for pp, tg in pp_tg_pairs:
                runs: list[dict[str, Any]] = []
                for r in range(args.repeat):
                    time.sleep(3)
                    try:
                        row, actual_pp_tokens = run_one_completion(
                            client, full_model_id, pp, tg, prompt_sizer
                        )
                    except Exception as e:
                        logger.error(e)
                        continue
                    row.update(
                        {
                            "model_short_id": short_id,
                            "model_id": full_model_id,
                            "placement_sharding": sharding,
                            "placement_instance_meta": instance_meta,
                            "placement_nodes": n_nodes,
                            "instance_id": instance_id,
                            "pp_tokens": actual_pp_tokens,
                            "tg": tg,
                            "repeat_index": r,
                            **(
                                {"download_duration_s": download_duration_s}
                                if download_duration_s is not None
                                else {}
                            ),
                        }
                    )
                    runs.append(row)
                    all_rows.append(row)

                if runs:
                    prompt_tps = mean(x["stats"]["prompt_tps"] for x in runs)
                    gen_tps = mean(x["stats"]["generation_tps"] for x in runs)
                    ptok = mean(x["stats"]["prompt_tokens"] for x in runs)
                    gtok = mean(x["stats"]["generation_tokens"] for x in runs)
                    peak = mean(
                        x["stats"]["peak_memory_usage"]["inBytes"] for x in runs
                    )

                    logger.info(
                        f"prompt_tps={prompt_tps:.2f} gen_tps={gen_tps:.2f}    "
                        f"prompt_tokens={ptok} gen_tokens={gtok}    "
                        f"peak_memory={format_peak_memory(peak)}\n"
                    )
                time.sleep(2)
        finally:
            try:
                client.request_json("DELETE", f"/instance/{instance_id}")
            except ExoHttpError as e:
                if e.status != 404:
                    raise
            wait_for_instance_gone(client, instance_id)
            logger.debug(f"Deleted instance {instance_id}")

            time.sleep(5)

    if args.stdout:
        json.dump(all_rows, sys.stdout, indent=2, ensure_ascii=False)
    elif args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2, ensure_ascii=False)
        logger.debug(f"\nWrote results JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
