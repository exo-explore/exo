# type: ignore
import argparse
import asyncio
import sys
import termios
import time
import tty

import aiohttp

NUM_REQUESTS = 10
BASE_URL = ""

QUESTIONS = [
    "What is the capital of Australia?",
    "How many bones are in the human body?",
    "What year did World War II end?",
    "What is the speed of light in meters per second?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical formula for water?",
    "How many planets are in our solar system?",
    "What is the largest ocean on Earth?",
    "Who painted the Mona Lisa?",
    "What is the boiling point of water in Celsius?",
]


def write(s: str) -> None:
    sys.stdout.write(s)


# ---------------------------------------------------------------------------
# Model picker (same style as exo_eval)
# ---------------------------------------------------------------------------


def fetch_models() -> list[str]:
    import json
    import urllib.request

    with urllib.request.urlopen(f"{BASE_URL}/state") as resp:
        data = json.loads(resp.read())
    model_ids: set[str] = set()
    for instance in data.get("instances", {}).values():
        for variant in instance.values():
            sa = variant.get("shardAssignments", {})
            model_id = sa.get("modelId")
            if model_id:
                model_ids.add(model_id)
    return sorted(model_ids)


def pick_model() -> str | None:
    models = fetch_models()
    if not models:
        print("No models found.")
        return None

    cursor = 0
    total_lines = len(models) + 4

    def render(first: bool = False) -> None:
        if not first:
            write(f"\033[{total_lines}A")
        write("\033[J")
        write("\033[1mSelect model\033[0m (up/down, enter confirm, q quit)\r\n\r\n")
        for i, model in enumerate(models):
            line = f"  {'>' if i == cursor else ' '} {model}"
            write(f"\033[7m{line}\033[0m\r\n" if i == cursor else f"{line}\r\n")
        write("\r\n")
        sys.stdout.flush()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        write("\033[?25l")
        render(first=True)
        while True:
            ch = sys.stdin.read(1)
            if ch in ("q", "\x03"):
                write("\033[?25h\033[0m\r\n")
                return None
            elif ch in ("\r", "\n"):
                break
            elif ch == "\x1b":
                seq = sys.stdin.read(2)
                if seq == "[A":
                    cursor = (cursor - 1) % len(models)
                elif seq == "[B":
                    cursor = (cursor + 1) % len(models)
            render()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        write(f"\033[{total_lines}A\033[J")  # clear picker UI
        write("\033[?25h\033[0m")
        sys.stdout.flush()

    return models[cursor]


# ---------------------------------------------------------------------------
# Parallel requests
# ---------------------------------------------------------------------------

statuses: list[str] = []
times: list[str] = []
previews: list[str] = []
tokens: list[str] = []
full_responses: list[dict | None] = []
total_lines = 0
start_time: float = 0
selected_model: str = ""


def render_progress(first: bool = False) -> None:
    if not first:
        write(f"\033[{total_lines}A")
    write("\033[J")
    elapsed = time.monotonic() - start_time if start_time else 0
    done = sum(1 for s in statuses if s == "done")
    write(
        f"\033[1m{selected_model}\033[0m  [{done}/{NUM_REQUESTS}]  {elapsed:.1f}s\r\n\r\n"
    )

    for i in range(NUM_REQUESTS):
        q = QUESTIONS[i % len(QUESTIONS)]
        status = statuses[i]
        if status == "pending":
            color = "\033[33m"  # yellow
        elif status == "running":
            color = "\033[36m"  # cyan
        elif status == "done":
            color = "\033[32m"  # green
        else:
            color = "\033[31m"  # red
        write(
            f"  {i:>2}  {color}{status:<8}\033[0m  {times[i]:>6}  {tokens[i]:>5}tok  {q[:40]:<40}  {previews[i][:50]}\r\n"
        )

    write("\r\n")
    sys.stdout.flush()


async def send_request(
    session: aiohttp.ClientSession, i: int, lock: asyncio.Lock
) -> None:
    payload = {
        "model": selected_model,
        "messages": [{"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]}],
        "max_tokens": 1024,
    }
    statuses[i] = "running"
    async with lock:
        render_progress()
    t0 = time.monotonic()
    try:
        async with session.post(
            f"{BASE_URL}/v1/chat/completions", json=payload
        ) as resp:
            data = await resp.json()
            elapsed = time.monotonic() - t0
            full_responses[i] = data
            times[i] = f"{elapsed:.1f}s"
            if resp.status == 200:
                choice = data["choices"][0]
                msg = choice["message"]
                content = msg.get("content", "")
                previews[i] = content[:50].replace("\n", " ") or "(empty)"
                if "usage" in data:
                    tokens[i] = str(data["usage"].get("total_tokens", ""))
                statuses[i] = "done"
            else:
                statuses[i] = f"err:{resp.status}"
                previews[i] = str(data.get("error", {}).get("message", ""))[:50]
    except Exception as e:
        elapsed = time.monotonic() - t0
        times[i] = f"{elapsed:.1f}s"
        statuses[i] = "error"
        previews[i] = str(e)[:50]
    async with lock:
        render_progress()


async def run_requests(print_stdout: bool = False) -> None:
    global start_time, total_lines, statuses, times, previews, tokens, full_responses

    statuses = ["pending"] * NUM_REQUESTS
    times = ["-"] * NUM_REQUESTS
    previews = ["-"] * NUM_REQUESTS
    tokens = ["-"] * NUM_REQUESTS
    full_responses = [None] * NUM_REQUESTS
    total_lines = NUM_REQUESTS + 4

    write("\033[?25l")  # hide cursor
    start_time = time.monotonic()
    render_progress(first=True)
    lock = asyncio.Lock()
    try:
        async with aiohttp.ClientSession() as session:
            tasks = [send_request(session, i, lock) for i in range(NUM_REQUESTS)]
            await asyncio.gather(*tasks)
        total = time.monotonic() - start_time
        write(
            f"\033[1m=== All {NUM_REQUESTS} requests done in {total:.1f}s ===\033[0m\r\n\r\n"
        )

        if print_stdout:
            for i in range(NUM_REQUESTS):
                data = full_responses[i]
                if not data or "choices" not in data:
                    continue
                choice = data["choices"][0]
                msg = choice["message"]
                q = QUESTIONS[i % len(QUESTIONS)]
                write(f"\033[1m--- #{i}: {q} ---\033[0m\r\n")
                if msg.get("reasoning_content"):
                    write(f"\033[2m[Thinking]: {msg['reasoning_content']}\033[0m\r\n")
                write(f"{msg.get('content', '')}\r\n")
                if "usage" in data:
                    u = data["usage"]
                    write(
                        f"\033[2m[Usage: prompt={u.get('prompt_tokens')}, "
                        f"completion={u.get('completion_tokens')}, "
                        f"total={u.get('total_tokens')}]\033[0m\r\n"
                    )
                write("\r\n")
    finally:
        write("\033[?25h")  # show cursor
        sys.stdout.flush()


def main() -> None:
    global selected_model, BASE_URL
    parser = argparse.ArgumentParser(
        description="Send parallel requests to an exo cluster"
    )
    parser.add_argument(
        "--host", required=True, help="Hostname of the exo node (e.g. s1)"
    )
    parser.add_argument("--port", type=int, default=52415, help="Port (default: 52415)")
    parser.add_argument(
        "--stdout", action="store_true", help="Print full responses after completion"
    )
    args = parser.parse_args()
    BASE_URL = f"http://{args.host}:{args.port}"
    model = pick_model()
    if not model:
        return
    selected_model = model
    asyncio.run(run_requests(print_stdout=args.stdout))


if __name__ == "__main__":
    main()
