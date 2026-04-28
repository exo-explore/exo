# type: ignore
#!/usr/bin/env python3
"""Disaggregated prefill-decode benchmark for exo (MLX → MLX).

Spins up two MLX instances on the cluster, marks one as Prefill source and
the other as Decode target via /v1/instance-links, then sends chat
completions to the API. The master routes the request to the decode
instance and stamps `prefill_endpoint` pointing at the prefill instance —
the worker decides per-request whether to ship prefill remotely
(uncached_count > REMOTE_PREFILL_MIN_TOKENS).

Usage:
    uv run python bench/prefill_decode_bench.py --model <id> --pp 2048,8192 --tg 128
    uv run python bench/prefill_decode_bench.py --model <id> --pp 4096 --tg 128 --repeat 3
    uv run python bench/prefill_decode_bench.py --model <id> --pp 2048 --tg 128 --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import itertools
import json
import sys
import time
import tomllib
from pathlib import Path
from statistics import mean
from typing import Any

from exo_bench import (
    PromptSizer,
    SystemMetricsSampler,
    format_peak_memory,
    load_tokenizer_for_bench,
    parse_int_list,
)
from harness import (
    ExoClient,
    ExoHttpError,
    add_common_instance_args,
    instance_id_from_instance,
    node_ids_from_instance,
    nodes_used_in_instance,
    resolve_model_short_id,
    run_planning_phase,
    settle_and_fetch_placements,
    unwrap_instance,
    wait_for_instance_gone,
    wait_for_instance_ready,
)
from loguru import logger


def _node_id_to_friendly(client: ExoClient) -> dict[str, str]:
    identities = client.get_node_identities() or {}
    out: dict[str, str] = {}
    for node_id, identity in identities.items():
        if isinstance(identity, dict):
            name = identity.get("friendlyName") or identity.get("friendly_name")
            if isinstance(name, str):
                out[str(node_id)] = name
    return out


def _placement_node_friendly_names(
    placement: dict[str, Any], id_to_friendly: dict[str, str]
) -> list[str]:
    instance = placement["instance"]
    return [id_to_friendly.get(nid, nid) for nid in node_ids_from_instance(instance)]


def _filter_by_node(
    placements: list[dict[str, Any]],
    friendly_name: str,
    id_to_friendly: dict[str, str],
) -> list[dict[str, Any]]:
    target = friendly_name.lower()
    matched: list[dict[str, Any]] = []
    for p in placements:
        names = [n.lower() for n in _placement_node_friendly_names(p, id_to_friendly)]
        if any(target == n or target in n for n in names):
            matched.append(p)
    return matched


def _node_id_by_friendly(id_to_friendly: dict[str, str], target: str) -> str | None:
    target_lc = target.lower()
    for nid, name in id_to_friendly.items():
        if target_lc == name.lower() or target_lc in name.lower():
            return nid
    return None


def _load_toml(path: str) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        return tomllib.load(f)


_TOP_LEVEL_TOML_KEYS = {
    "host",
    "port",
    "timeout",
    "settle_timeout",
    "model",
    "pp",
    "tg",
    "repeat",
    "warmup",
    "json_out",
    "instance_meta",
    "sharding",
    "min_nodes",
    "max_nodes",
    "force_download",
    "danger_delete_downloads",
    "all_combinations",
}


def _inject_toml_into_argv() -> None:
    """If --config X is in sys.argv, pre-load it and inject required CLI args
    (--model, --pp, --tg) so argparse's required=True checks pass."""
    argv = sys.argv
    if "--config" not in argv:
        return
    idx = argv.index("--config")
    if idx + 1 >= len(argv):
        return
    cfg_path = argv[idx + 1]
    cfg = _load_toml(cfg_path)
    decode = cfg.get("decode", {})

    def _has(flag: str) -> bool:
        return any(a == flag or a.startswith(flag + "=") for a in argv)

    # --model: prefer top-level, then [decode].model
    if not _has("--model"):
        model = cfg.get("model") or decode.get("model")
        if model:
            argv += ["--model", str(model)]
    if not _has("--pp"):
        pp = cfg.get("pp")
        if pp:
            argv += (
                ["--pp", *(str(x) for x in pp)]
                if isinstance(pp, list)
                else [
                    "--pp",
                    str(pp),
                ]
            )
    if not _has("--tg"):
        tg = cfg.get("tg")
        if tg:
            argv += (
                ["--tg", *(str(x) for x in tg)]
                if isinstance(tg, list)
                else [
                    "--tg",
                    str(tg),
                ]
            )


def _merge_toml_into_args(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    """Apply top-level toml keys onto args namespace where args has a default."""
    for key, value in cfg.items():
        if key in {"prefill", "decode"}:
            continue
        if key not in _TOP_LEVEL_TOML_KEYS:
            continue
        attr = key
        current = getattr(args, attr, None)
        if current in (None, [], False):
            setattr(args, attr, value)


def _side_args(
    base: argparse.Namespace, overrides: dict[str, Any]
) -> argparse.Namespace:
    out = copy.copy(base)
    for k in (
        "instance_meta",
        "sharding",
        "min_nodes",
        "max_nodes",
        "skip_pipeline_jaccl",
        "skip_tensor_ring",
    ):
        if k in overrides:
            setattr(out, k, overrides[k])
    return out


def _pick_two_distinct_placements(
    placements: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if len(placements) < 2:
        return None
    seen_nodes: set[tuple[str, ...]] = set()
    chosen: list[dict[str, Any]] = []
    for p in placements:
        nodes = tuple(sorted(str(n) for n in p.get("nodes", [])))
        if nodes in seen_nodes:
            continue
        seen_nodes.add(nodes)
        chosen.append(p)
        if len(chosen) == 2:
            return chosen[0], chosen[1]
    return None


def _create_instance_link(
    client: ExoClient,
    prefill_instance_id: str,
    decode_instance_id: str,
) -> str:
    out = client.request_json(
        "POST",
        "/v1/instance-links",
        body={
            "prefill_instances": [prefill_instance_id],
            "decode_instances": [decode_instance_id],
        },
    )
    return str(out.get("commandId", ""))


def _list_instance_links(client: ExoClient) -> list[dict[str, Any]]:
    out = client.request_json("GET", "/v1/instance-links")
    return out if isinstance(out, list) else []


def _delete_instance_link(client: ExoClient, link_id: str) -> None:
    client.request_json("DELETE", f"/v1/instance-links/{link_id}")


def run_one(
    client: ExoClient,
    model_id: str,
    pp_hint: int,
    tg: int,
    prompt_sizer: PromptSizer,
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
    choices = out.get("choices") or [{}]
    message = choices[0].get("message", {}) if choices else {}
    text = message.get("content") or ""
    preview = text[:200] if text else ""

    return {
        "elapsed_s": elapsed,
        "output_text_preview": preview,
        "stats": stats,
    }, pp_tokens


def _run_phase(
    *,
    client: ExoClient,
    label: str,
    pp_tg_pairs: list[tuple[int, int]],
    model_id: str,
    prompt_sizer: PromptSizer,
    warmup: int,
    repeat: int,
    common_meta: dict[str, Any],
    sampler: SystemMetricsSampler | None = None,
) -> list[dict[str, Any]]:
    logger.info(f"=== phase: {label} (model={model_id}) ===")
    rows: list[dict[str, Any]] = []
    for i in range(warmup):
        run_one(client, model_id, pp_tg_pairs[0][0], pp_tg_pairs[0][1], prompt_sizer)
        logger.debug(f"  warmup {i + 1}/{warmup} done")

    for pp, tg in pp_tg_pairs:
        logger.info(f"--- {label}: pp={pp} tg={tg} ---")
        runs: list[dict[str, Any]] = []
        inference_windows: list[tuple[float, float]] = []
        for r in range(repeat):
            time.sleep(2)
            try:
                inf_t0 = time.monotonic()
                row, actual_pp_tokens = run_one(client, model_id, pp, tg, prompt_sizer)
                inference_windows.append((inf_t0, time.monotonic()))
            except Exception as e:
                logger.error(e)
                continue
            row.update(common_meta)
            row.update(
                {
                    "phase": label,
                    "phase_model_id": model_id,
                    "pp_tokens": actual_pp_tokens,
                    "tg": tg,
                    "repeat_index": r,
                }
            )
            runs.append(row)
            rows.append(row)

        if runs:
            prompt_tps = mean(x["stats"]["prompt_tps"] for x in runs)
            gen_tps = mean(x["stats"]["generation_tps"] for x in runs)
            ptok = mean(x["stats"]["prompt_tokens"] for x in runs)
            gtok = mean(x["stats"]["generation_tokens"] for x in runs)
            peak = mean(x["stats"]["peak_memory_usage"]["inBytes"] for x in runs)
            avg_elapsed = mean(x["elapsed_s"] for x in runs)
            energy_str = ""
            if sampler is not None and inference_windows:
                joules = sum(
                    sampler.energy_between(t0, t1) for t0, t1 in inference_windows
                )
                inf_seconds = sum(t1 - t0 for t0, t1 in inference_windows)
                avg_watts = joules / inf_seconds if inf_seconds > 0 else 0.0
                energy_per_run = joules / len(runs) if runs else 0.0
                energy_str = (
                    f"    energy={joules:.1f}J ({avg_watts:.1f}W avg over "
                    f"{inf_seconds:.1f}s inference, {energy_per_run:.1f}J/run)"
                )
                for run_row, (t0, t1) in zip(runs, inference_windows, strict=False):
                    run_row["energy_joules"] = sampler.energy_between(t0, t1)
                    run_row["inference_window_s"] = t1 - t0
            logger.info(
                f"[{label}] prompt_tps={prompt_tps:.2f} gen_tps={gen_tps:.2f}    "
                f"prompt_tokens={ptok} gen_tokens={gtok}    "
                f"peak_memory={format_peak_memory(peak)}    "
                f"avg_elapsed={avg_elapsed:.2f}s{energy_str}"
            )
        time.sleep(2)
    return rows


def _summarise(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, float]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for r in rows:
        key = (int(r["pp_tokens"]), int(r["tg"]))
        grouped.setdefault(key, []).append(r)
    out: dict[tuple[int, int], dict[str, float]] = {}
    for key, runs in grouped.items():
        energy_runs = [x.get("energy_joules") for x in runs if "energy_joules" in x]
        window_runs = [
            x.get("inference_window_s") for x in runs if "inference_window_s" in x
        ]
        out[key] = {
            "prompt_tps": mean(x["stats"]["prompt_tps"] for x in runs),
            "gen_tps": mean(x["stats"]["generation_tps"] for x in runs),
            "elapsed_s": mean(x["elapsed_s"] for x in runs),
            "prompt_tokens": mean(x["stats"]["prompt_tokens"] for x in runs),
            "gen_tokens": mean(x["stats"]["generation_tokens"] for x in runs),
            "energy_j": mean(energy_runs) if energy_runs else 0.0,
            "inference_window_s": mean(window_runs) if window_runs else 0.0,
        }
    return out


def _normalised_seconds(summary: dict[str, float], pp: int, tg: int) -> float | None:
    """Wall-clock time implied by reported tps for the *configured* pp/tg.

    elapsed_s is not comparable across phases when models EOS at different
    lengths. This formula reconstructs "what would this phase take to do
    pp prompt tokens + tg generation tokens" using its own reported rates.
    """
    p_tps = summary.get("prompt_tps", 0.0)
    g_tps = summary.get("gen_tps", 0.0)
    if p_tps <= 0 or g_tps <= 0:
        return None
    return pp / p_tps + tg / g_tps


def _print_diff(
    disagg_rows: list[dict[str, Any]],
    decode_alone_rows: list[dict[str, Any]],
    prefill_alone_rows: list[dict[str, Any]],
) -> None:
    disagg = _summarise(disagg_rows)
    decode_alone = _summarise(decode_alone_rows)
    prefill_alone = _summarise(prefill_alone_rows)
    keys = set(disagg.keys()) | set(decode_alone.keys()) | set(prefill_alone.keys())

    width = 110
    for key in sorted(keys):
        pp, tg = key
        logger.info("─" * width)
        logger.info(f"  pp={pp}  tg={tg}")
        logger.info("─" * width)
        logger.info(
            f"  {'phase':<16} {'elapsed':>9}  {'norm':>9}  "
            f"{'prompt_tps':>11}  {'gen_tps':>8}  "
            f"{'p_tok':>6}  {'g_tok':>6}  "
            f"{'energy':>9}  {'avg_W':>7}"
        )
        for label, summary in (
            ("disaggregated", disagg.get(key)),
            ("decode_alone", decode_alone.get(key)),
            ("prefill_alone", prefill_alone.get(key)),
        ):
            if summary is None:
                logger.info(
                    f"  {label:<16} {'—':>9}  {'—':>9}  "
                    f"{'—':>11}  {'—':>8}  {'—':>6}  {'—':>6}  "
                    f"{'—':>9}  {'—':>7}"
                )
                continue
            norm = _normalised_seconds(summary, pp, tg)
            norm_str = f"{norm:>8.2f}s" if norm is not None else f"{'—':>9}"
            energy = summary.get("energy_j", 0.0)
            window = summary.get("inference_window_s", 0.0)
            energy_str = f"{energy:>8.1f}J" if energy > 0 else f"{'—':>9}"
            avg_w = energy / window if window > 0 else 0.0
            avg_w_str = f"{avg_w:>6.1f}W" if avg_w > 0 else f"{'—':>7}"
            logger.info(
                f"  {label:<16} "
                f"{summary['elapsed_s']:>8.2f}s  "
                f"{norm_str}  "
                f"{summary['prompt_tps']:>11.1f}  "
                f"{summary['gen_tps']:>8.2f}  "
                f"{summary['prompt_tokens']:>6.0f}  "
                f"{summary['gen_tokens']:>6.0f}  "
                f"{energy_str}  "
                f"{avg_w_str}"
            )

        d = disagg.get(key)
        da = decode_alone.get(key)
        pa = prefill_alone.get(key)
        d_norm = _normalised_seconds(d, pp, tg) if d else None
        if d_norm and da:
            da_norm = _normalised_seconds(da, pp, tg)
            if da_norm:
                logger.info(
                    f"  norm speedup vs decode_alone:  {da_norm / d_norm:.2f}x  "
                    f"(prefill {d['prompt_tps'] / da['prompt_tps']:.2f}x, "
                    f"decode {d['gen_tps'] / da['gen_tps']:.2f}x)"
                )
        if d_norm and pa:
            pa_norm = _normalised_seconds(pa, pp, tg)
            if pa_norm:
                logger.info(
                    f"  norm speedup vs prefill_alone: {pa_norm / d_norm:.2f}x  "
                    f"(prefill {d['prompt_tps'] / pa['prompt_tps']:.2f}x, "
                    f"decode {d['gen_tps'] / pa['gen_tps']:.2f}x)"
                )
    logger.info("─" * width)


def main() -> int:
    _inject_toml_into_argv()
    ap = argparse.ArgumentParser(
        prog="prefill-decode-bench",
        description="Benchmark MLX-MLX disaggregated prefill/decode via instance links.",
    )
    add_common_instance_args(ap)
    ap.add_argument(
        "--pp",
        nargs="+",
        required=True,
        help="Prompt-size hints (ints, must be >1000). Accepts commas.",
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
        help="Warmup runs (uses first pp/tg).",
    )
    ap.add_argument(
        "--json-out",
        default="bench/prefill_decode_results.json",
        help="Write raw per-run results JSON to this path.",
    )
    ap.add_argument("--stdout", action="store_true", help="Write results to stdout")
    ap.add_argument(
        "--dry-run", action="store_true", help="List selected placements and exit."
    )
    ap.add_argument(
        "--all-combinations",
        action="store_true",
        help="Force all pp×tg combinations even when lists have equal length.",
    )
    ap.add_argument(
        "--prefill-model",
        default=None,
        help="Model id for the prefill instance. Defaults to --model.",
    )
    ap.add_argument(
        "--prefill-node",
        default=None,
        help="friendly_name of the node hosting the prefill instance.",
    )
    ap.add_argument(
        "--decode-node",
        default=None,
        help="friendly_name of the node hosting the decode instance.",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="TOML config file. CLI flags override toml values.",
    )
    ap.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run each (pp,tg) pair without the prefill/decode link "
        "(decode instance does its own prefill) and report the diff.",
    )
    args = ap.parse_args()
    cfg = _load_toml(args.config) if args.config else {}
    _merge_toml_into_args(args, cfg)
    prefill_overrides = cfg.get("prefill", {}) if cfg else {}
    decode_overrides = cfg.get("decode", {}) if cfg else {}
    if args.prefill_model is None and "model" in prefill_overrides:
        args.prefill_model = prefill_overrides["model"]
    if args.prefill_node is None and "node" in prefill_overrides:
        args.prefill_node = prefill_overrides["node"]
    if args.decode_node is None and "node" in decode_overrides:
        args.decode_node = decode_overrides["node"]
    if "model" in decode_overrides and not args.model:
        args.model = decode_overrides["model"]

    pp_list = parse_int_list(args.pp)
    tg_list = parse_int_list(args.tg)
    if not pp_list or not tg_list:
        logger.error("pp and tg lists must be non-empty")
        return 2
    for pp in pp_list:
        if pp <= 1000:
            logger.error(
                f"pp={pp} must be >1000 (remote prefill triggers when uncached >1000)"
            )
            return 2
    if args.repeat <= 0:
        logger.error("--repeat must be >= 1")
        return 2

    use_combinations = args.all_combinations or len(pp_list) != len(tg_list)
    if use_combinations:
        logger.info(
            f"pp/tg mode: combinations (product) — {len(pp_list) * len(tg_list)} pairs"
        )
    else:
        logger.info(f"pp/tg mode: tandem (zip) — {len(pp_list)} pairs")

    client = ExoClient(args.host, args.port, timeout_s=args.timeout)

    decode_short_id, decode_full_id = resolve_model_short_id(
        client, args.model, force_download=args.force_download
    )
    if args.prefill_model:
        prefill_short_id, prefill_full_id = resolve_model_short_id(
            client, args.prefill_model, force_download=args.force_download
        )
    else:
        prefill_short_id, prefill_full_id = decode_short_id, decode_full_id

    tokenizer = load_tokenizer_for_bench(decode_full_id)
    if tokenizer is None:
        raise RuntimeError("[prefill-decode-bench] decode tokenizer load failed")
    try:
        decode_prompt_sizer = PromptSizer(tokenizer)
    except Exception:
        logger.error("[prefill-decode-bench] decode prompt sizing failed")
        raise

    if prefill_full_id == decode_full_id:
        prefill_prompt_sizer = decode_prompt_sizer
    else:
        prefill_tokenizer = load_tokenizer_for_bench(prefill_full_id)
        if prefill_tokenizer is None:
            raise RuntimeError("[prefill-decode-bench] prefill tokenizer load failed")
        prefill_prompt_sizer = PromptSizer(prefill_tokenizer)

    id_to_friendly = _node_id_to_friendly(client)

    prefill_args = _side_args(args, prefill_overrides)
    decode_args = _side_args(args, decode_overrides)

    if prefill_full_id == decode_full_id and prefill_overrides == decode_overrides:
        placements = settle_and_fetch_placements(
            client, decode_full_id, args, settle_timeout=args.settle_timeout
        )
        prefill_candidates = (
            _filter_by_node(placements, args.prefill_node, id_to_friendly)
            if args.prefill_node
            else placements
        )
        decode_candidates = (
            _filter_by_node(placements, args.decode_node, id_to_friendly)
            if args.decode_node
            else placements
        )
        if args.prefill_node and not prefill_candidates:
            logger.error(f"No placement on prefill node {args.prefill_node!r}.")
            return 1
        if args.decode_node and not decode_candidates:
            logger.error(f"No placement on decode node {args.decode_node!r}.")
            return 1
        if args.prefill_node and args.decode_node:
            prefill_p = prefill_candidates[0]
            decode_p = decode_candidates[0]
        else:
            pair = _pick_two_distinct_placements(placements)
            if pair is None:
                logger.error(
                    "Need at least two distinct-node MLX placements for the same model."
                )
                return 1
            prefill_p, decode_p = pair
            if args.prefill_node:
                prefill_p = prefill_candidates[0]
            if args.decode_node:
                decode_p = decode_candidates[0]
    else:
        prefill_node_id = (
            _node_id_by_friendly(id_to_friendly, args.prefill_node)
            if args.prefill_node
            else None
        )
        decode_node_id = (
            _node_id_by_friendly(id_to_friendly, args.decode_node)
            if args.decode_node
            else None
        )
        if args.prefill_node and prefill_node_id is None:
            logger.error(f"Unknown node {args.prefill_node!r}.")
            return 1
        if args.decode_node and decode_node_id is None:
            logger.error(f"Unknown node {args.decode_node!r}.")
            return 1
        prefill_placements = settle_and_fetch_placements(
            client,
            prefill_full_id,
            prefill_args,
            settle_timeout=args.settle_timeout,
            node_id=prefill_node_id,
        )
        decode_placements = settle_and_fetch_placements(
            client,
            decode_full_id,
            decode_args,
            settle_timeout=args.settle_timeout,
            node_id=decode_node_id,
        )
        if not prefill_placements:
            logger.error(
                f"No placement found for prefill model {prefill_full_id}"
                f"{f' on node {args.prefill_node!r}' if args.prefill_node else ''}."
            )
            return 1
        if not decode_placements:
            logger.error(
                f"No placement found for decode model {decode_full_id}"
                f"{f' on node {args.decode_node!r}' if args.decode_node else ''}."
            )
            return 1
        prefill_p = prefill_placements[0]
        decode_p = decode_placements[0]

    prefill_node_names = _placement_node_friendly_names(prefill_p, id_to_friendly)
    decode_node_names = _placement_node_friendly_names(decode_p, id_to_friendly)
    _ = unwrap_instance

    prefill_instance = prefill_p["instance"]
    decode_instance = decode_p["instance"]
    prefill_id = instance_id_from_instance(prefill_instance)
    decode_id = instance_id_from_instance(decode_instance)
    prefill_meta = str(prefill_p.get("instance_meta", ""))
    decode_meta = str(decode_p.get("instance_meta", ""))
    prefill_nodes = nodes_used_in_instance(prefill_instance)
    decode_nodes = nodes_used_in_instance(decode_instance)

    logger.info("=" * 80)
    logger.info(
        f"PREFILL: {prefill_meta} / nodes={prefill_nodes} ({','.join(prefill_node_names)}) "
        f"/ {prefill_short_id} ({prefill_full_id}) / instance_id={prefill_id}"
    )
    logger.info(
        f"DECODE:  {decode_meta} / nodes={decode_nodes} ({','.join(decode_node_names)}) "
        f"/ {decode_short_id} ({decode_full_id}) / instance_id={decode_id}"
    )

    if args.dry_run:
        return 0

    settle_deadline = (
        time.monotonic() + args.settle_timeout if args.settle_timeout > 0 else None
    )

    logger.info("Planning phase: prefill...")
    run_planning_phase(
        client,
        prefill_full_id,
        prefill_p,
        args.danger_delete_downloads,
        args.timeout,
        settle_deadline,
    )
    logger.info("Planning phase: decode...")
    run_planning_phase(
        client,
        decode_full_id,
        decode_p,
        args.danger_delete_downloads,
        args.timeout,
        settle_deadline,
    )

    if use_combinations:
        pp_tg_pairs = list(itertools.product(pp_list, tg_list))
    else:
        pp_tg_pairs = list(zip(pp_list, tg_list, strict=True))

    common_meta = {
        "decode_model_short_id": decode_short_id,
        "decode_model_id": decode_full_id,
        "prefill_model_short_id": prefill_short_id,
        "prefill_model_id": prefill_full_id,
        "prefill_instance_id": prefill_id,
        "prefill_instance_meta": prefill_meta,
        "prefill_nodes": prefill_nodes,
        "decode_instance_id": decode_id,
        "decode_instance_meta": decode_meta,
        "decode_nodes": decode_nodes,
    }

    all_rows: list[dict[str, Any]] = []
    disagg_rows: list[dict[str, Any]] = []
    decode_alone_rows: list[dict[str, Any]] = []
    prefill_alone_rows: list[dict[str, Any]] = []
    link_id = ""
    prefill_alive = False
    decode_alive = False
    sampler_nodes = sorted(
        {*node_ids_from_instance(prefill_instance), *node_ids_from_instance(decode_instance)}
    )
    sampler = SystemMetricsSampler(
        ExoClient(args.host, args.port, timeout_s=30), sampler_nodes
    )
    sampler.start()
    try:
        logger.info("Creating prefill instance...")
        client.request_json("POST", "/instance", body={"instance": prefill_instance})
        wait_for_instance_ready(client, prefill_id)
        prefill_alive = True
        logger.info("Prefill instance ready")

        if args.compare_baseline:
            time.sleep(2)
            prefill_alone_rows = _run_phase(
                client=client,
                label="prefill_alone",
                pp_tg_pairs=pp_tg_pairs,
                model_id=prefill_full_id,
                prompt_sizer=prefill_prompt_sizer,
                warmup=args.warmup,
                repeat=args.repeat,
                common_meta=common_meta,
                sampler=sampler,
            )
            all_rows.extend(prefill_alone_rows)

        logger.info("Creating decode instance...")
        client.request_json("POST", "/instance", body={"instance": decode_instance})
        wait_for_instance_ready(client, decode_id)
        decode_alive = True
        logger.info("Decode instance ready")

        logger.info("Linking instances (prefill → decode)...")
        _create_instance_link(client, prefill_id, decode_id)
        time.sleep(1)
        links = _list_instance_links(client)
        if not links:
            logger.error("Link did not appear in state.")
            return 1
        link_id = str(links[-1].get("linkId") or links[-1].get("link_id") or "")
        logger.info(f"Link created: {link_id}")
        time.sleep(2)

        disagg_rows = _run_phase(
            client=client,
            label="disaggregated",
            pp_tg_pairs=pp_tg_pairs,
            model_id=decode_full_id,
            prompt_sizer=decode_prompt_sizer,
            warmup=args.warmup,
            repeat=args.repeat,
            common_meta=common_meta,
            sampler=sampler,
        )
        all_rows.extend(disagg_rows)

        if args.compare_baseline:
            logger.info("Removing link and prefill instance to isolate decode_alone.")
            with contextlib.suppress(ExoHttpError):
                if link_id:
                    _delete_instance_link(client, link_id)
                    link_id = ""
            with contextlib.suppress(ExoHttpError):
                client.request_json("DELETE", f"/instance/{prefill_id}")
            wait_for_instance_gone(client, prefill_id)
            prefill_alive = False
            time.sleep(2)

            decode_alone_rows = _run_phase(
                client=client,
                label="decode_alone",
                pp_tg_pairs=pp_tg_pairs,
                model_id=decode_full_id,
                prompt_sizer=decode_prompt_sizer,
                warmup=args.warmup,
                repeat=args.repeat,
                common_meta=common_meta,
                sampler=sampler,
            )
            all_rows.extend(decode_alone_rows)

            _print_diff(disagg_rows, decode_alone_rows, prefill_alone_rows)
    finally:
        sampler.stop()
        with contextlib.suppress(ExoHttpError):
            if link_id:
                _delete_instance_link(client, link_id)
        if decode_alive:
            with contextlib.suppress(ExoHttpError):
                client.request_json("DELETE", f"/instance/{decode_id}")
            wait_for_instance_gone(client, decode_id)
        if prefill_alive:
            with contextlib.suppress(ExoHttpError):
                client.request_json("DELETE", f"/instance/{prefill_id}")
            wait_for_instance_gone(client, prefill_id)
        logger.debug("Deleted both instances")

    if args.stdout:
        json.dump(all_rows, sys.stdout, indent=2, ensure_ascii=False)
    elif args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2, ensure_ascii=False)
        logger.debug(f"\nWrote results JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
