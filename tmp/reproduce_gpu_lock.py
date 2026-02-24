#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Reproduce GPU lock issue with mlx-community/Llama-3.2-1B-Instruct-4bit.

Starts exo or mlx_lm.server, then sends repeated chat completions
until a request stalls for >5 seconds (indicating a GPU lock).

Usage:
  uv run tmp/reproduce_gpu_lock.py              # use exo (default)
  uv run tmp/reproduce_gpu_lock.py --mlx-lm     # use mlx_lm.server
"""

import argparse
import hashlib
import json
import os
import platform
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"
MODEL_PATH = os.path.expanduser("~/.exo/models/mlx-community--Llama-3.2-1B-Instruct-4bit")
STALL_THRESHOLD_S = 5.0

server_proc = None
base_url = ""


def cleanup(*_):
    if server_proc and server_proc.poll() is None:
        print("\nStopping server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def api_get(path, timeout=30):
    req = urllib.request.Request(f"{base_url}{path}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def api_post(path, body, timeout=300):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{base_url}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def wait_for_api(max_wait=120):
    print("Waiting for API to be ready...", flush=True)
    start = time.time()
    while time.time() - start < max_wait:
        try:
            api_get("/v1/models", timeout=5)
            print("API is ready.", flush=True)
            return
        except Exception:
            time.sleep(2)
    print("ERROR: API did not become ready in time.", flush=True)
    cleanup()


def create_instance(max_wait=120):
    print(f"Waiting for valid placements for {MODEL_ID}...", flush=True)
    start = time.time()
    valid = []
    while time.time() - start < max_wait:
        try:
            previews = api_get(f"/instance/previews?model_id={MODEL_ID}")
            valid = [p for p in previews.get("previews", []) if p.get("error") is None and p.get("instance") is not None]
            if valid:
                break
        except Exception:
            pass
        time.sleep(3)
    if not valid:
        print("ERROR: No valid placements found after waiting.", flush=True)
        cleanup()

    instance = valid[0]["instance"]
    print(f"Creating instance (sharding={valid[0].get('sharding')}, meta={valid[0].get('instance_meta')})...", flush=True)
    resp = api_post("/instance", {"instance": instance})
    print(f"Instance creation requested: {resp.get('message')} (command_id={resp.get('command_id')})", flush=True)
    return instance.get("id") or instance.get("instance_id")


def wait_for_instance(max_wait=120):
    print("Waiting for instance to be ready...", flush=True)
    start = time.time()
    while time.time() - start < max_wait:
        try:
            state = api_get("/state", timeout=10)
            instances = state.get("instances") or state.get("model_instances") or {}
            if instances:
                print(f"Instance ready. ({len(instances)} instance(s) in state)", flush=True)
                return
        except Exception:
            pass
        time.sleep(3)
    print("WARNING: Timed out waiting for instance in state. Proceeding anyway...", flush=True)


TOPICS = [
    "the weather", "cats", "space", "pizza", "music", "ocean", "mountains",
    "robots", "books", "coffee", "trains", "clouds", "birds", "fire",
    "ice cream", "trees", "rivers", "stars", "thunder", "gardens",
]

def send_chat(request_num):
    topic = random.choice(TOPICS)
    nonce = random.randint(1000, 9999)
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": f"Say something about {topic} in one sentence. ({nonce})"}],
        "stream": False,
        "max_tokens": 64,
    }
    start = time.time()
    resp = api_post("/v1/chat/completions", body, timeout=600)
    elapsed = time.time() - start
    return elapsed, resp


def start_exo():
    global server_proc
    machine_id = hashlib.sha256(f"{platform.node()}-{uuid.getnode()}".encode()).hexdigest()[:12]
    namespace = f"gpu-lock-repro-{machine_id}"

    log_file = open("/tmp/exo_gpu_lock_repro.log", "w", buffering=1)
    print(f"\nStarting exo (namespace={namespace})...", flush=True)
    print(f"Log: /tmp/exo_gpu_lock_repro.log", flush=True)
    print(f"  tail -f /tmp/exo_gpu_lock_repro.log  (in another terminal to watch)", flush=True)
    env = {**os.environ, "EXO_LIBP2P_NAMESPACE": namespace, "PYTHONUNBUFFERED": "1"}
    server_proc = subprocess.Popen(
        ["uv", "run", "exo"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    print(f"exo started (pid={server_proc.pid})", flush=True)

    wait_for_api()
    create_instance()
    wait_for_instance()


def start_mlx_lm():
    global server_proc
    log_file = open("/tmp/mlx_lm_gpu_lock_repro.log", "w", buffering=1)
    print(f"\nStarting mlx_lm.server on port 8080...", flush=True)
    print(f"  Model path: {MODEL_PATH}", flush=True)
    print(f"Log: /tmp/mlx_lm_gpu_lock_repro.log", flush=True)
    print(f"  tail -f /tmp/mlx_lm_gpu_lock_repro.log  (in another terminal to watch)", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    server_proc = subprocess.Popen(
        ["uv", "run", "mlx_lm.server", "--model", MODEL_PATH, "--port", "8080"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=os.path.expanduser("~/mlx-lm"),
    )
    print(f"mlx_lm.server started (pid={server_proc.pid})", flush=True)
    wait_for_api()


def chat_loop():
    print("\n" + "-" * 60, flush=True)
    print("Starting chat completion loop. Watching for stalls...", flush=True)
    print("-" * 60 + "\n", flush=True)

    timings = []
    request_num = 0

    while True:
        request_num += 1
        print(f"  [#{request_num}] sending...", end="", flush=True)
        req_start = time.time()

        done_event = threading.Event()
        def print_waiting():
            while not done_event.is_set():
                if done_event.wait(5):
                    break
                elapsed_so_far = time.time() - req_start
                print(f" ({elapsed_so_far:.0f}s)", end="", flush=True)
        watcher = threading.Thread(target=print_waiting, daemon=True)
        watcher.start()

        try:
            elapsed, resp = send_chat(request_num)
        except Exception as e:
            done_event.set()
            print(f" ERROR after {time.time() - req_start:.1f}s: {e}", flush=True)
            time.sleep(2)
            continue
        finally:
            done_event.set()

        timings.append(elapsed)
        content = ""
        try:
            content = resp["choices"][0]["message"]["content"][:80]
        except (KeyError, IndexError):
            content = "<no content>"

        print(f" {elapsed:.2f}s  |  {content}", flush=True)

        if elapsed > STALL_THRESHOLD_S:
            print("\n", flush=True)
            print("!" * 60, flush=True)
            print("!" * 60, flush=True)
            print("!!!", flush=True)
            print(f"!!!  GPU LOCK DETECTED on request #{request_num}", flush=True)
            print(f"!!!  Elapsed: {elapsed:.2f}s (threshold: {STALL_THRESHOLD_S}s)", flush=True)
            print("!!!", flush=True)
            print("!" * 60, flush=True)
            print("!" * 60, flush=True)
            print(f"\nTotal requests sent: {request_num}", flush=True)
            print(f"Average time (all):  {sum(timings) / len(timings):.2f}s", flush=True)
            normal = [t for t in timings if t <= STALL_THRESHOLD_S]
            if normal:
                print(f"Average time (normal): {sum(normal) / len(normal):.2f}s", flush=True)
            print(f"Max time: {max(timings):.2f}s", flush=True)
            print(f"Min time: {min(timings):.2f}s", flush=True)
            print("\nAll timings:", flush=True)
            for i, t in enumerate(timings, 1):
                marker = " <<<< STALL" if t > STALL_THRESHOLD_S else ""
                print(f"  #{i}: {t:.2f}s{marker}", flush=True)
            print(f"\nServer still running (pid={server_proc.pid}). Continuing... Ctrl+C to stop.", flush=True)
            print("-" * 60 + "\n", flush=True)


def main():
    global base_url

    parser = argparse.ArgumentParser(description="Reproduce GPU lock issue")
    parser.add_argument("--mlx-lm", action="store_true", help="Use mlx_lm.server instead of exo")
    parser.add_argument("--port", type=int, default=None, help="Override server port")
    args = parser.parse_args()

    mode = "mlx_lm" if args.mlx_lm else "exo"
    port = args.port or (8080 if args.mlx_lm else 52415)
    base_url = f"http://localhost:{port}"

    print("=" * 60, flush=True)
    print("  GPU Lock Reproduction Script", flush=True)
    print(f"  Mode: {mode}", flush=True)
    print(f"  Model: {MODEL_ID}", flush=True)
    print(f"  API: {base_url}", flush=True)
    print(f"  Stall threshold: {STALL_THRESHOLD_S}s", flush=True)
    print("=" * 60, flush=True)

    if args.mlx_lm:
        start_mlx_lm()
    else:
        start_exo()

    chat_loop()


if __name__ == "__main__":
    main()
