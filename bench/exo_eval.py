# type: ignore
#!/usr/bin/env python3
"""Quality evaluation for exo — matches Artificial Analysis methodology.

Runs LLM benchmarks against exo's OpenAI-compatible API using the same
prompts, temperature settings, and answer extraction as Artificial Analysis.

Supported benchmarks:
  gpqa_diamond   - Graduate-level science QA (198 questions, 4-choice MC)
  mmlu_pro       - Multi-task language understanding (12K questions, 10-choice MC)
  aime_2024      - Math olympiad 2024 (30 problems, integer answers)
  aime_2025      - Math olympiad 2025 (30 problems, integer answers)
  humaneval      - Python code generation (164 problems, pass@1)
  livecodebench  - Competitive programming (880+ problems, pass@1)

Model configs in eval_configs/models.toml auto-detect reasoning/non-reasoning
settings per model. Override with --reasoning / --no-reasoning.

Usage:
  uv run python exo_eval.py --model <model-id> --tasks gpqa_diamond
  uv run python exo_eval.py --model <model-id> --tasks humaneval,livecodebench --limit 50
  uv run python exo_eval.py --model <model-id> --tasks gpqa_diamond --compare-concurrency 1,4

References:
  https://artificialanalysis.ai/methodology/intelligence-benchmarking
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import random
import re
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
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

# ---------------------------------------------------------------------------
# Artificial Analysis constants
# ---------------------------------------------------------------------------

MAX_RETRIES = 30
DEFAULT_MAX_TOKENS = 16_384
REASONING_MAX_TOKENS = 131_072
TEMPERATURE_NON_REASONING = 0.0
TEMPERATURE_REASONING = 0.6

# MC answer extraction: 8 fallback regex patterns.
# All patterns are tried; the match at the latest text position wins
# (handles models that self-correct during reasoning).
_MC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])"
    ),
    re.compile(r"\\boxed\{[^}]*([A-Z])[^}]*\}"),
    re.compile(r"(?i)answer is ([a-zA-Z])"),
    re.compile(r"(?i)answer is \\\(([a-zA-Z])"),
    re.compile(r"([A-Z])\)\s*[^A-Z]*$"),
    re.compile(r"([A-Z])\s+is\s+the\s+correct\s+answer"),
    re.compile(r"([A-Z])\s*$"),
    re.compile(r"([A-Z])\s*\."),
]

# Code extraction: last ```python ... ``` block (AA regex)
_CODE_BLOCK_RE = re.compile(r"```(?:python|Python)?\s*\n(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# Model config loading
# ---------------------------------------------------------------------------


def load_model_config(model_id: str) -> dict[str, Any] | None:
    """Look up model in eval_configs/models.toml. Returns config dict or None."""
    config_path = Path(__file__).resolve().parent / "eval_configs" / "models.toml"
    if not config_path.exists():
        return None
    with open(config_path, "rb") as f:
        data = tomllib.load(f)
    for entry in data.get("model", []):
        patterns = entry.get("patterns", [])
        if any(p in model_id for p in patterns):
            return entry
    return None


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_mc_answer(text: str, valid_letters: str = "ABCD") -> str | None:
    """Extract MC answer. Last match by text position wins."""
    valid_set = set(valid_letters)
    best: tuple[int, str] | None = None
    for pattern in _MC_PATTERNS:
        for m in pattern.finditer(text):
            letter = m.group(1).upper()
            if letter in valid_set:
                pos = m.start()
                if best is None or pos >= best[0]:
                    best = (pos, letter)
    return best[1] if best else None


def extract_boxed_answer(text: str) -> str | None:
    r"""Extract content from the last \boxed{...}."""
    matches: list[str] = []
    idx = 0
    while True:
        pos = text.find("\\boxed{", idx)
        if pos < 0:
            break
        depth = 0
        i = pos + len("\\boxed{")
        start = i
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                if depth == 0:
                    matches.append(text[start:i])
                    break
                depth -= 1
            i += 1
        idx = i + 1 if i < len(text) else len(text)
    return matches[-1].strip() if matches else None


def extract_code_block(text: str) -> str | None:
    """Extract the last Python code block from markdown response."""
    matches = _CODE_BLOCK_RE.findall(text)
    if matches:
        return matches[-1].strip()
    # Fallback: try raw code after last ```
    lines = text.split("\n")
    backtick_lines = [i for i, line in enumerate(lines) if "```" in line]
    if len(backtick_lines) >= 2:
        return "\n".join(lines[backtick_lines[-2] + 1 : backtick_lines[-1]])
    return None


def check_aime_answer(extracted: str, gold: int) -> bool:
    """Check if extracted AIME answer matches gold integer."""
    try:
        return int(extracted.strip()) == gold
    except ValueError:
        pass
    try:
        from math_verify import parse, verify

        return verify(parse(str(gold)), parse(extracted))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Code execution sandbox
# ---------------------------------------------------------------------------


def execute_code(
    code: str, stdin_input: str | None = None, timeout: float = 10.0
) -> tuple[bool, str]:
    """Execute Python code in subprocess. Returns (passed, stderr_or_stdout)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Execution timed out"
    except Exception as e:
        return False, str(e)


def run_humaneval_test(
    prompt: str, completion: str, test: str, entry_point: str
) -> bool:
    """Execute HumanEval test case. Returns True if all assertions pass."""
    check_program = prompt + completion + "\n" + test + f"\ncheck({entry_point})"
    passed, _ = execute_code(check_program, timeout=10.0)
    return passed


def run_livecodebench_test(
    code: str,
    test_inputs: list[str],
    test_outputs: list[str],
    test_type: str,
    func_name: str | None = None,
) -> bool:
    """Execute LiveCodeBench test case. Returns True if all tests pass."""
    for inp, expected_out in zip(test_inputs, test_outputs, strict=True):
        expected_out = expected_out.strip()
        if test_type == "functional" and func_name:
            # Call function with parsed input, compare output
            check_code = (
                f"{code}\n\n"
                f"import json\n"
                f"args = json.loads({json.dumps(inp)})\n"
                f"if isinstance(args, list):\n"
                f"    result = {func_name}(*args)\n"
                f"else:\n"
                f"    result = {func_name}(args)\n"
                f"expected = json.loads({json.dumps(expected_out)})\n"
                f"assert result == expected, f'Got {{result}}, expected {{expected}}'\n"
            )
            passed, _ = execute_code(check_code, timeout=10.0)
        else:
            # stdin/stdout test
            passed, stdout = execute_code(code, stdin_input=inp, timeout=10.0)
            if passed:
                passed = stdout.strip() == expected_out
        if not passed:
            return False
    return True


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------


@dataclass
class QuestionResult:
    question_id: int
    prompt: str
    response: str
    extracted_answer: str | None
    gold_answer: str
    correct: bool
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    elapsed_s: float = 0.0


@dataclass
class BenchmarkConfig:
    name: str
    description: str
    dataset_name: str
    dataset_config: str | None
    split: str
    kind: str  # "mc", "math", "code"


BENCHMARKS: dict[str, BenchmarkConfig] = {
    "gpqa_diamond": BenchmarkConfig(
        name="gpqa_diamond",
        description="Graduate-level science QA (198 Q, 4-choice MC)",
        dataset_name="Idavidrein/gpqa",
        dataset_config="gpqa_diamond",
        split="train",
        kind="mc",
    ),
    "mmlu_pro": BenchmarkConfig(
        name="mmlu_pro",
        description="Multi-task language understanding (12K Q, 10-choice MC)",
        dataset_name="TIGER-Lab/MMLU-Pro",
        dataset_config=None,
        split="test",
        kind="mc",
    ),
    "aime_2024": BenchmarkConfig(
        name="aime_2024",
        description="Math olympiad 2024 (30 problems, integer answers)",
        dataset_name="HuggingFaceH4/aime_2024",
        dataset_config=None,
        split="train",
        kind="math",
    ),
    "aime_2025": BenchmarkConfig(
        name="aime_2025",
        description="Math olympiad 2025 (30 problems, integer answers)",
        dataset_name="MathArena/aime_2025",
        dataset_config=None,
        split="train",
        kind="math",
    ),
    "humaneval": BenchmarkConfig(
        name="humaneval",
        description="Python code generation (164 problems, pass@1)",
        dataset_name="openai/openai_humaneval",
        dataset_config=None,
        split="test",
        kind="code",
    ),
    "livecodebench": BenchmarkConfig(
        name="livecodebench",
        description="Competitive programming (880+ problems, pass@1)",
        dataset_name="livecodebench/code_generation_lite",
        dataset_config=None,
        split="test",
        kind="code",
    ),
}


# ---------------------------------------------------------------------------
# Prompt formatters
# ---------------------------------------------------------------------------

_GPQA_INSTRUCTION = (
    "Answer the following multiple choice question. "
    "The last line of your response should be in the following format: "
    "'Answer: A/B/C/D' (e.g. 'Answer: A')."
)

_MMLU_PRO_INSTRUCTION = (
    "Answer the following multiple choice question. "
    "The last line of your response should be in the following format: "
    "'Answer: A/B/C/D/E/F/G/H/I/J' (e.g. 'Answer: A')."
)

_AIME_INSTRUCTION = (
    "Solve the following math problem step by step. "
    "Put your answer inside \\boxed{}.\n"
    "Remember to put your answer inside \\boxed{}."
)

_HUMANEVAL_INSTRUCTION = (
    "Complete the following Python function. Return only the function body "
    "inside a ```python code block. Do not include the function signature."
)

# LiveCodeBench: AA uses original prompts without custom system prompts
_LCB_SYSTEM = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests."
)

_LCB_WITH_STARTER = (
    "### Question:\n{question}\n\n"
    "### Format: You will use the following starter code to write the "
    "solution to the problem and enclose your code within delimiters.\n"
    "```python\n{starter_code}\n```\n\n"
    "### Answer: (use the provided format with backticks)\n"
)

_LCB_WITHOUT_STARTER = (
    "### Question:\n{question}\n\n"
    "### Format: Read the inputs from stdin solve the problem and write "
    "the answer to stdout (do not directly test on the sample inputs). "
    "Enclose your code within delimiters as follows. Ensure that when the "
    "python program runs, it reads the inputs, runs the algorithm and "
    "writes output to STDOUT.\n"
    "```python\n# YOUR CODE HERE\n```\n\n"
    "### Answer: (use the provided format with backticks)\n"
)


def format_gpqa_question(doc: dict, idx: int) -> tuple[str, str]:
    """Returns (prompt, correct_letter)."""
    correct = doc["Correct Answer"]
    choices = [
        correct,
        doc["Incorrect Answer 1"],
        doc["Incorrect Answer 2"],
        doc["Incorrect Answer 3"],
    ]
    rng = random.Random(idx)
    order = rng.sample(range(4), 4)
    shuffled = [choices[i] for i in order]
    correct_letter = "ABCD"[order.index(0)]
    choices_text = "\n".join(f"{L}) {shuffled[i]}" for i, L in enumerate("ABCD"))
    return f"{_GPQA_INSTRUCTION}\n\n{doc['Question']}\n\n{choices_text}", correct_letter


def format_mmlu_pro_question(doc: dict) -> tuple[str, str]:
    """Returns (prompt, correct_letter)."""
    options = doc["options"]
    letters = "ABCDEFGHIJ"
    choices_text = "\n".join(f"{letters[i]}) {opt}" for i, opt in enumerate(options))
    return f"{_MMLU_PRO_INSTRUCTION}\n\n{doc['question']}\n\n{choices_text}", doc[
        "answer"
    ]


def format_aime_question(doc: dict) -> tuple[str, int]:
    """Returns (prompt, correct_answer_int)."""
    return f"{_AIME_INSTRUCTION}\n\n{doc['problem']}", int(doc["answer"])


def format_humaneval_question(doc: dict) -> tuple[str, dict]:
    """Returns (prompt, metadata_for_execution)."""
    prompt = f"{_HUMANEVAL_INSTRUCTION}\n\n```python\n{doc['prompt']}```"
    meta = {
        "prompt": doc["prompt"],
        "test": doc["test"],
        "entry_point": doc["entry_point"],
    }
    return prompt, meta


def format_livecodebench_question(doc: dict) -> tuple[str, str | None, dict]:
    """Returns (prompt, system_message, metadata_for_execution)."""
    starter_code = doc.get("starter_code", "")
    question_content = doc["question_content"]

    if starter_code and starter_code.strip():
        user_msg = _LCB_WITH_STARTER.format(
            question=question_content, starter_code=starter_code
        )
    else:
        user_msg = _LCB_WITHOUT_STARTER.format(question=question_content)

    # Parse test cases
    public_tests = (
        json.loads(doc["public_test_cases"])
        if isinstance(doc["public_test_cases"], str)
        else doc["public_test_cases"]
    )
    private_tests = doc.get("private_test_cases", "[]")
    if isinstance(private_tests, str):
        try:
            private_tests = json.loads(private_tests)
        except Exception:
            import base64
            import pickle
            import zlib

            private_tests = json.loads(
                pickle.loads(
                    zlib.decompress(base64.b64decode(private_tests.encode("utf-8")))
                )
            )

    all_tests = public_tests + (
        private_tests if isinstance(private_tests, list) else []
    )
    test_inputs = [t["input"] for t in all_tests]
    test_outputs = [t["output"] for t in all_tests]
    test_type = all_tests[0]["testtype"] if all_tests else "stdin"

    metadata = doc.get("metadata", "{}")
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    func_name = metadata.get("func_name")

    meta = {
        "test_inputs": test_inputs,
        "test_outputs": test_outputs,
        "test_type": test_type,
        "func_name": func_name,
    }
    return user_msg, _LCB_SYSTEM, meta


# ---------------------------------------------------------------------------
# API client with retries
# ---------------------------------------------------------------------------


@dataclass
class ApiResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int


async def _call_api(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float | None,
    system_message: str | None = None,
) -> ApiResult:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    resp = await client.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    if not content or not content.strip():
        raise ValueError("Empty response from model")
    usage = data.get("usage", {})
    details = usage.get("completion_tokens_details", {})
    return ApiResult(
        content=content,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        reasoning_tokens=details.get("reasoning_tokens", 0) if details else 0,
    )


async def call_with_retries(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float | None = None,
    system_message: str | None = None,
) -> ApiResult | None:
    for attempt in range(MAX_RETRIES):
        try:
            return await _call_api(
                client,
                base_url,
                model,
                prompt,
                temperature,
                max_tokens,
                timeout,
                system_message,
            )
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = min(2**attempt, 60)
                logger.warning(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}. Retrying in {wait}s..."
                )
                await asyncio.sleep(wait)
            else:
                logger.error(f"All {MAX_RETRIES} retries exhausted. Last error: {e}")
                return None


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------


async def evaluate_benchmark(
    benchmark_name: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    concurrency: int = 1,
    limit: int | None = None,
    timeout: float | None = None,
) -> list[QuestionResult]:
    """Run a benchmark. Returns per-question results."""
    import datasets

    config = BENCHMARKS[benchmark_name]
    logger.info(f"Loading dataset {config.dataset_name}...")

    try:
        if benchmark_name == "livecodebench":
            ds = datasets.load_dataset(
                "json",
                data_files="hf://datasets/livecodebench/code_generation_lite/*.jsonl",
                split="train",
            )
        else:
            ds = datasets.load_dataset(
                config.dataset_name,
                config.dataset_config,
                split=config.split,
            )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        if "gated" in str(e).lower() or "login" in str(e).lower():
            logger.error("Dataset requires authentication. Run: huggingface-cli login")
        return []

    total = len(ds)
    if limit and limit < total:
        ds = ds.select(range(limit))
        total = limit

    logger.info(
        f"Evaluating {benchmark_name}: {total} questions, concurrency={concurrency}, "
        f"temperature={temperature}, max_tokens={max_tokens}"
    )

    if config.kind == "code":
        logger.warning(
            "Code benchmarks execute model-generated code. Use a sandboxed environment."
        )

    semaphore = asyncio.Semaphore(concurrency)
    results: list[QuestionResult | None] = [None] * total
    completed = 0
    lock = asyncio.Lock()

    async def process_question(
        idx: int, doc: dict, http_client: httpx.AsyncClient
    ) -> None:
        nonlocal completed
        system_msg = None

        if benchmark_name == "gpqa_diamond":
            prompt, gold = format_gpqa_question(doc, idx)
            valid_letters = "ABCD"
        elif benchmark_name == "mmlu_pro":
            prompt, gold = format_mmlu_pro_question(doc)
            valid_letters = "ABCDEFGHIJ"[: len(doc["options"])]
        elif benchmark_name.startswith("aime"):
            prompt, gold_int = format_aime_question(doc)
            gold = str(gold_int)
        elif benchmark_name == "humaneval":
            prompt, exec_meta = format_humaneval_question(doc)
            gold = "pass"
        elif benchmark_name == "livecodebench":
            prompt, system_msg, exec_meta = format_livecodebench_question(doc)
            gold = "pass"
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        async with semaphore:
            t0 = time.monotonic()
            api_result = await call_with_retries(
                http_client,
                base_url,
                model,
                prompt,
                temperature,
                max_tokens,
                timeout,
                system_message=system_msg,
            )
            elapsed = time.monotonic() - t0

        if api_result is None:
            result = QuestionResult(
                question_id=idx,
                prompt=prompt,
                response="",
                extracted_answer=None,
                gold_answer=gold,
                correct=False,
                error="API failure after retries",
                elapsed_s=elapsed,
            )
        else:
            response = api_result.content
            stats = {
                "prompt_tokens": api_result.prompt_tokens,
                "completion_tokens": api_result.completion_tokens,
                "reasoning_tokens": api_result.reasoning_tokens,
                "elapsed_s": elapsed,
            }

            if config.kind == "mc":
                extracted = extract_mc_answer(response, valid_letters)
                result = QuestionResult(
                    question_id=idx,
                    prompt=prompt,
                    response=response,
                    extracted_answer=extracted,
                    gold_answer=gold,
                    correct=(extracted == gold) if extracted else False,
                    **stats,
                )
            elif config.kind == "math":
                extracted = extract_boxed_answer(response)
                correct = (
                    check_aime_answer(extracted, int(gold)) if extracted else False
                )
                result = QuestionResult(
                    question_id=idx,
                    prompt=prompt,
                    response=response,
                    extracted_answer=extracted,
                    gold_answer=gold,
                    correct=correct,
                    **stats,
                )
            elif config.kind == "code":
                code = extract_code_block(response)
                if code is None:
                    result = QuestionResult(
                        question_id=idx,
                        prompt=prompt,
                        response=response,
                        extracted_answer=None,
                        gold_answer=gold,
                        correct=False,
                        error="No code block extracted",
                        **stats,
                    )
                elif benchmark_name == "humaneval":
                    passed = run_humaneval_test(
                        exec_meta["prompt"],
                        code,
                        exec_meta["test"],
                        exec_meta["entry_point"],
                    )
                    result = QuestionResult(
                        question_id=idx,
                        prompt=prompt,
                        response=response,
                        extracted_answer="pass" if passed else "fail",
                        gold_answer=gold,
                        correct=passed,
                        **stats,
                    )
                elif benchmark_name == "livecodebench":
                    passed = run_livecodebench_test(
                        code,
                        exec_meta["test_inputs"],
                        exec_meta["test_outputs"],
                        exec_meta["test_type"],
                        exec_meta["func_name"],
                    )
                    result = QuestionResult(
                        question_id=idx,
                        prompt=prompt,
                        response=response,
                        extracted_answer="pass" if passed else "fail",
                        gold_answer=gold,
                        correct=passed,
                        **stats,
                    )
                else:
                    result = QuestionResult(
                        question_id=idx,
                        prompt=prompt,
                        response=response,
                        extracted_answer=None,
                        gold_answer=gold,
                        correct=False,
                        error="Unknown code benchmark",
                        **stats,
                    )
            else:
                result = QuestionResult(
                    question_id=idx,
                    prompt=prompt,
                    response=response,
                    extracted_answer=None,
                    gold_answer=gold,
                    correct=False,
                    error="Unsupported kind",
                    **stats,
                )

        results[idx] = result

        async with lock:
            completed += 1
            n = completed
        if n % max(1, total // 20) == 0 or n == total:
            correct_so_far = sum(1 for r in results if r is not None and r.correct)
            answered = sum(1 for r in results if r is not None)
            logger.info(
                f"  [{n}/{total}] {correct_so_far}/{answered} correct "
                f"({correct_so_far / max(answered, 1):.1%})"
            )

    async with httpx.AsyncClient() as http_client:
        tasks = [process_question(i, doc, http_client) for i, doc in enumerate(ds)]
        await asyncio.gather(*tasks)

    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------


def print_results(
    benchmark_name: str,
    results: list[QuestionResult],
    concurrency: int | None = None,
) -> dict[str, Any]:
    total = len(results)
    correct = sum(r.correct for r in results)
    errors = sum(1 for r in results if r.error)
    no_extract = sum(1 for r in results if r.extracted_answer is None and not r.error)
    accuracy = correct / max(total, 1)

    total_prompt_tokens = sum(r.prompt_tokens for r in results)
    total_completion_tokens = sum(r.completion_tokens for r in results)
    total_reasoning_tokens = sum(r.reasoning_tokens for r in results)
    total_elapsed = sum(r.elapsed_s for r in results)
    wall_clock = max(r.elapsed_s for r in results) if results else 0.0
    avg_gen_tps = total_completion_tokens / total_elapsed if total_elapsed > 0 else 0.0

    label = f"[c={concurrency}] " if concurrency is not None else ""
    print(f"\n{label}{benchmark_name}: {correct}/{total} ({accuracy:.1%})")
    tok_line = f"  tokens: {total_prompt_tokens:,} prompt + {total_completion_tokens:,} completion"
    if total_reasoning_tokens > 0:
        tok_line += f" ({total_reasoning_tokens:,} reasoning)"
    tok_line += (
        f"  |  avg gen tps: {avg_gen_tps:.1f}"
        f"  |  total time: {total_elapsed:.1f}s  wall clock: {wall_clock:.1f}s"
    )
    print(tok_line)
    if errors:
        print(f"  API errors: {errors}")
    if no_extract:
        print(f"  No answer extracted: {no_extract}")

    return {
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": errors,
        "no_extract": no_extract,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_reasoning_tokens": total_reasoning_tokens,
        "total_elapsed_s": total_elapsed,
        "wall_clock_s": wall_clock,
        "avg_gen_tps": avg_gen_tps,
    }


def print_comparison(
    benchmark_name: str,
    results_by_c: dict[int, list[QuestionResult]],
) -> None:
    levels = sorted(results_by_c.keys())
    print(f"\n{'=' * 70}")
    print(f"COMPARISON: {benchmark_name}")
    print(f"{'=' * 70}")

    header = f"{'Concurrency':<15} {'Accuracy':>10} {'Correct':>10} {'Total':>10} {'Comp Tokens':>12} {'Wall Clock':>12} {'Avg Gen TPS':>12}"
    print(header)
    print("-" * len(header))
    for c in levels:
        r = results_by_c[c]
        correct = sum(q.correct for q in r)
        total = len(r)
        comp_tok = sum(q.completion_tokens for q in r)
        total_elapsed = sum(q.elapsed_s for q in r)
        avg_tps = comp_tok / total_elapsed if total_elapsed > 0 else 0.0
        wall = max(q.elapsed_s for q in r) if r else 0.0
        print(
            f"c={c:<13} {correct / max(total, 1):>10.1%} {correct:>10} {total:>10}"
            f" {comp_tok:>12,} {wall:>11.1f}s {avg_tps:>12.1f}"
        )

    if len(levels) >= 2:
        base_results = results_by_c[levels[0]]
        test_results = results_by_c[levels[-1]]
        changed = sum(
            1
            for br, tr in zip(base_results, test_results, strict=True)
            if br.correct != tr.correct
        )
        total = min(len(base_results), len(test_results))
        print(
            f"\nQuestions with different correctness (c={levels[0]} vs c={levels[-1]}): {changed}/{total}"
        )
        if changed == 0:
            print("Batching produced identical quality.")
        elif changed <= total * 0.01:
            print("Negligible quality difference from batching.")
        else:
            print(
                f"WARNING: {changed / max(total, 1) * 100:.1f}% of questions changed."
            )
    print()


# ---------------------------------------------------------------------------
# Interactive task picker
# ---------------------------------------------------------------------------


def pick_tasks_interactive() -> list[str]:
    import termios
    import tty

    if not sys.stdin.isatty():
        logger.error("No --tasks specified and stdin is not a terminal.")
        return []

    items = [(name, cfg.description) for name, cfg in BENCHMARKS.items()]
    selected: list[bool] = [False] * len(items)
    cursor = 0
    total_lines = len(items) + 4

    def write(s: str) -> None:
        sys.stdout.write(s)

    def render(first: bool = False) -> None:
        if not first:
            write(f"\033[{total_lines}A")
        write("\033[J")
        write(
            "\033[1mSelect benchmarks\033[0m (up/down, space toggle, enter confirm, q quit)\r\n\r\n"
        )
        for i, (name, desc) in enumerate(items):
            marker = ">" if i == cursor else " "
            check = "x" if selected[i] else " "
            line = f"  {marker} [{check}] {name:<17} {desc}"
            write(f"\033[7m{line}\033[0m\r\n" if i == cursor else f"{line}\r\n")
        write(f"\r\n  {sum(selected)} selected\r\n")
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
                return []
            elif ch in ("\r", "\n"):
                break
            elif ch == " ":
                selected[cursor] = not selected[cursor]
            elif ch == "\x1b":
                seq = sys.stdin.read(2)
                if seq == "[A":
                    cursor = (cursor - 1) % len(items)
                elif seq == "[B":
                    cursor = (cursor + 1) % len(items)
            render()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        write("\033[?25h\033[0m\r\n")
        sys.stdout.flush()

    chosen = [name for (name, _), sel in zip(items, selected, strict=True) if sel]
    if chosen:
        logger.info(f"Selected: {', '.join(chosen)}")
    return chosen


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------


def save_results(
    results_dir: str,
    benchmark_name: str,
    model: str,
    concurrency: int,
    results: list[QuestionResult],
    scores: dict[str, Any],
) -> Path:
    out_dir = Path(results_dir) / model.replace("/", "_") / benchmark_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"c{concurrency}_{ts}.json"

    data = {
        "benchmark": benchmark_name,
        "model": model,
        "concurrency": concurrency,
        "scores": scores,
        "results": [
            {
                "question_id": r.question_id,
                "extracted_answer": r.extracted_answer,
                "gold_answer": r.gold_answer,
                "correct": r.correct,
                "error": r.error,
                "response_length": len(r.response),
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "reasoning_tokens": r.reasoning_tokens,
                "elapsed_s": round(r.elapsed_s, 2),
            }
            for r in results
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Results saved to {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_int_list(values: list[str]) -> list[int]:
    items: list[int] = []
    for v in values:
        for part in v.split(","):
            if part.strip():
                items.append(int(part.strip()))
    return items


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="exo-eval",
        description="Quality evaluation for exo — matches Artificial Analysis methodology.",
    )
    add_common_instance_args(ap)

    ap.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated benchmark names. Omit for interactive picker.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max questions per benchmark (for fast iteration).",
    )

    reasoning_group = ap.add_mutually_exclusive_group()
    reasoning_group.add_argument(
        "--reasoning",
        action="store_true",
        default=None,
        help="Force reasoning-model settings (temperature=0.6, max_tokens=65536).",
    )
    reasoning_group.add_argument(
        "--no-reasoning",
        action="store_true",
        default=False,
        help="Force non-reasoning settings (temperature=0, max_tokens=16384).",
    )

    ap.add_argument(
        "--temperature", type=float, default=None, help="Override temperature."
    )
    ap.add_argument(
        "--max-tokens", type=int, default=None, help="Override max output tokens."
    )
    ap.add_argument(
        "--num-concurrent",
        type=int,
        default=1,
        help="Concurrent API requests (default: 1).",
    )
    ap.add_argument(
        "--compare-concurrency",
        nargs="+",
        default=None,
        help="Run at multiple concurrency levels and compare. E.g. --compare-concurrency 1,4",
    )
    ap.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        help="Per-request timeout in seconds (default: no timeout).",
    )
    ap.add_argument(
        "--results-dir",
        default="eval_results",
        help="Directory for result JSON files (default: eval_results).",
    )
    ap.add_argument(
        "--skip-instance-setup",
        action="store_true",
        help="Skip exo instance management (assumes model is already running).",
    )

    args, _ = ap.parse_known_args()

    # Resolve tasks
    if args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        task_names = pick_tasks_interactive()
    if not task_names:
        return 0

    for t in task_names:
        if t not in BENCHMARKS:
            logger.error(f"Unknown benchmark '{t}'. Available: {', '.join(BENCHMARKS)}")
            return 1

    # Instance management
    client = ExoClient(args.host, args.port, timeout_s=args.timeout)
    instance_id: str | None = None

    if not args.skip_instance_setup:
        short_id, full_model_id = resolve_model_short_id(
            client,
            args.model,
            force_download=args.force_download,
        )
        selected = settle_and_fetch_placements(
            client,
            full_model_id,
            args,
            settle_timeout=args.settle_timeout,
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
        preview = selected[0]
        instance = preview["instance"]
        instance_id = instance_id_from_instance(instance)

        logger.info(
            f"PLACEMENT: {preview['sharding']} / {preview['instance_meta']} / "
            f"nodes={nodes_used_in_instance(instance)}"
        )

        settle_deadline = (
            time.monotonic() + args.settle_timeout if args.settle_timeout > 0 else None
        )
        download_duration = run_planning_phase(
            client,
            full_model_id,
            preview,
            args.danger_delete_downloads,
            args.timeout,
            settle_deadline,
        )
        if download_duration is not None:
            logger.info(f"Download: {download_duration:.1f}s")

        client.request_json("POST", "/instance", body={"instance": instance})
        try:
            wait_for_instance_ready(client, instance_id)
        except (RuntimeError, TimeoutError) as e:
            logger.error(f"Failed to initialize: {e}")
            with contextlib.suppress(ExoHttpError):
                client.request_json("DELETE", f"/instance/{instance_id}")
            return 1
        time.sleep(1)
    else:
        full_model_id = args.model

    # Auto-detect reasoning from model config
    model_config = load_model_config(full_model_id)
    if args.reasoning:
        is_reasoning = True
    elif args.no_reasoning:
        is_reasoning = False
    elif model_config is not None:
        is_reasoning = model_config.get("reasoning", False)
        logger.info(
            f"Auto-detected from config: {model_config['name']} → "
            f"{'reasoning' if is_reasoning else 'non-reasoning'}"
        )
    else:
        is_reasoning = False
        logger.warning(
            f"Model '{full_model_id}' not found in eval_configs/models.toml. "
            f"Defaulting to non-reasoning. Use --reasoning to override."
        )

    # Resolve temperature and max_tokens
    if args.temperature is not None:
        temperature = args.temperature
    else:
        temperature = (
            TEMPERATURE_REASONING if is_reasoning else TEMPERATURE_NON_REASONING
        )

    if args.max_tokens is not None:
        max_tokens = args.max_tokens
    else:
        max_tokens = REASONING_MAX_TOKENS if is_reasoning else DEFAULT_MAX_TOKENS

    base_url = f"http://{args.host}:{args.port}"

    logger.info(f"Model: {full_model_id}")
    logger.info(
        f"Settings: temperature={temperature}, max_tokens={max_tokens}, "
        f"reasoning={'yes' if is_reasoning else 'no'}"
    )

    try:
        if args.compare_concurrency:
            concurrency_levels = parse_int_list(args.compare_concurrency)
            for task_name in task_names:
                results_by_c: dict[int, list[QuestionResult]] = {}
                for c in concurrency_levels:
                    logger.info(f"\n{'=' * 50}")
                    logger.info(f"Running {task_name} at concurrency={c}")
                    results = asyncio.run(
                        evaluate_benchmark(
                            task_name,
                            base_url,
                            full_model_id,
                            temperature,
                            max_tokens,
                            concurrency=c,
                            limit=args.limit,
                            timeout=args.request_timeout,
                        )
                    )
                    if results:
                        scores = print_results(task_name, results, concurrency=c)
                        save_results(
                            args.results_dir,
                            task_name,
                            full_model_id,
                            c,
                            results,
                            scores,
                        )
                        results_by_c[c] = results
                if len(results_by_c) >= 2:
                    print_comparison(task_name, results_by_c)
        else:
            for task_name in task_names:
                results = asyncio.run(
                    evaluate_benchmark(
                        task_name,
                        base_url,
                        full_model_id,
                        temperature,
                        max_tokens,
                        concurrency=args.num_concurrent,
                        limit=args.limit,
                        timeout=args.request_timeout,
                    )
                )
                if results:
                    scores = print_results(task_name, results)
                    save_results(
                        args.results_dir,
                        task_name,
                        full_model_id,
                        args.num_concurrent,
                        results,
                        scores,
                    )
    finally:
        if instance_id is not None:
            try:
                client.request_json("DELETE", f"/instance/{instance_id}")
            except ExoHttpError as e:
                if e.status != 404:
                    raise
            wait_for_instance_gone(client, instance_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
