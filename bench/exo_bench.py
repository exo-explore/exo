#!/usr/bin/env python3
# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import time
from collections.abc import Callable
from statistics import mean
from typing import Any
from urllib.parse import urlencode

from loguru import logger
from transformers import AutoTokenizer


class ExoHttpError(RuntimeError):
    def __init__(self, status: int, reason: str, body_preview: str):
        super().__init__(f"HTTP {status} {reason}: {body_preview}")
        self.status = status


class ExoClient:
    def __init__(self, host: str, port: int, timeout_s: float = 600.0):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s

    def request_json(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        if not path.startswith("/"):
            path = "/" + path
        if params:
            path = path + "?" + urlencode(params)

        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)
        try:
            payload: bytes | None = None
            hdrs: dict[str, str] = {"Accept": "application/json"}

            if body is not None:
                payload = json.dumps(body).encode("utf-8")
                hdrs["Content-Type"] = "application/json"
            if headers:
                hdrs.update(headers)

            conn.request(method.upper(), path, body=payload, headers=hdrs)
            resp = conn.getresponse()
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace") if raw else ""

            if resp.status >= 400:
                raise ExoHttpError(resp.status, resp.reason, text[:300])

            if not text:
                return None
            return json.loads(text)
        finally:
            conn.close()

    def post_bench_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request_json("POST", "/bench/chat/completions", body=payload)


def unwrap_instance(instance: dict[str, Any]) -> dict[str, Any]:
    if len(instance) != 1:
        raise KeyError(f"Expected 1 key, got keys={list(instance.keys())}")

    tag = next(iter(instance))
    inner = instance[tag]
    if not isinstance(inner, dict):
        raise TypeError(f"payload for {tag} must be dict, got {type(inner)}")
    return inner


def instance_id_from_instance(instance: dict[str, Any]) -> str:
    inner = unwrap_instance(instance)
    return str(inner["instanceId"])


def nodes_used_in_instance(instance: dict[str, Any]) -> int:
    inner = unwrap_instance(instance)
    return len(inner["shardAssignments"]["nodeToRunner"])


def runner_ids_from_instance(instance: dict[str, Any]) -> list[str]:
    inner = unwrap_instance(instance)
    runner_to_shard = inner["shardAssignments"]["runnerToShard"]
    return list(runner_to_shard.keys())


def runner_ready(runner: dict[str, Any]) -> bool:
    return "RunnerReady" in runner


def runner_failed(runner: dict[str, Any]) -> bool:
    return "RunnerFailed" in runner


def get_runner_failed_message(runner: dict[str, Any]) -> str | None:
    if "RunnerFailed" in runner:
        return runner["RunnerFailed"].get("errorMessage")
    return None


def wait_for_instance_ready(
    client: ExoClient, instance_id: str, timeout: float = 24000.0
) -> None:
    start_time = time.time()
    instance_existed = False
    while time.time() - start_time < timeout:
        state = client.request_json("GET", "/state")
        instances = state.get("instances", {})

        if instance_id not in instances:
            if instance_existed:
                # Instance was deleted after being created - likely due to runner failure
                raise RuntimeError(
                    f"Instance {instance_id} was deleted (runner may have failed)"
                )
            time.sleep(0.1)
            continue

        instance_existed = True
        instance = instances[instance_id]
        runner_ids = runner_ids_from_instance(instance)
        runners = state.get("runners", {})

        # Check for failed runners first
        for rid in runner_ids:
            runner = runners.get(rid, {})
            if runner_failed(runner):
                error_msg = get_runner_failed_message(runner) or "Unknown error"
                raise RuntimeError(f"Runner {rid} failed: {error_msg}")

        if all(runner_ready(runners.get(rid, {})) for rid in runner_ids):
            return

        time.sleep(0.1)

    raise TimeoutError(f"Instance {instance_id} did not become ready within {timeout=}")


def wait_for_instance_gone(
    client: ExoClient, instance_id: str, timeout: float = 3.0
) -> None:
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            client.request_json("GET", f"/instance/{instance_id}")
            time.sleep(0.4)
        except ExoHttpError as e:
            if e.status == 404:
                return

    raise TimeoutError(f"Instance {instance_id} did not get deleted within {timeout=}")


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

    seen: set[int] = set()
    out: list[int] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def resolve_model_short_id(client: ExoClient, model_arg: str) -> tuple[str, str]:
    models = client.request_json("GET", "/models") or {}
    data = models.get("data") or []

    for m in data:
        if m.get("name").lower() == model_arg.lower():
            short_id = str(m["name"])
            full_id = str(m.get("hugging_face_id") or m["name"])
            return short_id, full_id

    for m in data:
        if m.get("hugging_face_id") == model_arg:
            short_id = str(m["name"])
            full_id = str(m["hugging_face_id"])
            return short_id, full_id

    raise ValueError(f"Model not found in /models: {model_arg}")


def placement_filter(instance_meta: str, wanted: str) -> bool:
    s = (instance_meta or "").lower()
    if wanted == "both":
        return ("ring" in s) or ("jaccl" in s)
    return wanted in s


def sharding_filter(sharding: str, wanted: str) -> bool:
    s = (sharding or "").lower()
    if wanted == "both":
        return ("pipeline" in s) or ("tensor" in s)
    return wanted in s


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

    preview = (out.get("choices") or [{}])[0]["message"]["content"][:200]

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

        content = ""
        tok = self.count_fn(content)

        while tok < target:
            content += self.atom
            tok = self.count_fn(content)

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
    ap.add_argument("--host", default=os.environ.get("EXO_HOST", "localhost"))
    ap.add_argument(
        "--port", type=int, default=int(os.environ.get("EXO_PORT", "52415"))
    )
    ap.add_argument("--model", required=True, help="Model short id or huggingface id")
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
        "--max-nodes",
        type=int,
        default=4,
        help="Only consider placements using <= this many nodes.",
    )
    ap.add_argument(
        "--min-nodes",
        type=int,
        default=1,
        help="Only consider placements using >= this many nodes.",
    )
    ap.add_argument(
        "--instance-meta", choices=["ring", "jaccl", "both"], default="both"
    )
    ap.add_argument(
        "--sharding", choices=["pipeline", "tensor", "both"], default="both"
    )
    ap.add_argument(
        "--skip-pipeline-jaccl",
        action="store_true",
        help="Pipeline jaccl is often pointless, skip by default",
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
        "--timeout", type=float, default=600.0, help="HTTP timeout (seconds)."
    )
    ap.add_argument(
        "--json-out",
        default="bench/results.json",
        help="Write raw per-run results JSON to this path.",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="List selected placements and exit."
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

    client = ExoClient(args.host, args.port, timeout_s=args.timeout)
    short_id, full_model_id = resolve_model_short_id(client, args.model)

    previews_resp = client.request_json(
        "GET", "/instance/previews", params={"model_id": full_model_id}
    )
    previews = previews_resp.get("previews") or []

    tokenizer = AutoTokenizer.from_pretrained(
        full_model_id,
        trust_remote_code=True,
    )
    if tokenizer is None:
        raise RuntimeError("[exo-bench] tokenizer load failed")

    try:
        prompt_sizer = PromptSizer(tokenizer)
        logger.debug(f"[exo-bench] loaded tokenizer: {full_model_id} for prompt sizer")
    except Exception:
        logger.error("[exo-bench] tokenizer usable but prompt sizing failed")
        raise

    selected: list[dict[str, Any]] = []
    for p in previews:
        if p.get("error") is not None:
            continue
        if not placement_filter(str(p.get("instance_meta", "")), args.instance_meta):
            continue
        if not sharding_filter(str(p.get("sharding", "")), args.sharding):
            continue

        instance = p.get("instance")
        if not isinstance(instance, dict):
            continue

        n = nodes_used_in_instance(instance)
        # Skip tensor ring single node as it is pointless when pipeline ring
        if n == 1 and (
            (args.sharding == "both" and "tensor" in p.get("sharding", "").lower())
            or (
                args.instance_meta == "both"
                and "jaccl" in p.get("instance_meta", "").lower()
            )
        ):
            continue

        if (
            args.skip_pipeline_jaccl
            and (
                args.instance_meta == "both"
                and "jaccl" in p.get("instance_meta", "").lower()
            )
            and (
                args.sharding == "both" and "pipeline" in p.get("sharding", "").lower()
            )
        ):
            continue

        if args.min_nodes <= n <= args.max_nodes:
            selected.append(p)

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

            for pp in pp_list:
                # if (
                #     pp * n_nodes > 2048
                #     and "ring" in instance_meta.lower()
                #     and "tensor" in sharding.lower()
                # ):
                #     model_card = MODEL_CARDS[short_id]
                #     if model_card.metadata.storage_size > Memory.from_gb(10):
                #         logger.info(
                #             f"Skipping tensor ring as this is too slow for model of size {model_card.metadata.storage_size} on {n_nodes=}"
                #         )
                #         continue
                for tg in tg_list:
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

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2, ensure_ascii=False)
        logger.debug(f"\nWrote results JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
