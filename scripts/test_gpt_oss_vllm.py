#!/usr/bin/env python3
"""Test GPT OSS output parsing with vLLM backend.

Run on Spark:
    uv run python scripts/test_gpt_oss_vllm.py
    uv run python scripts/test_gpt_oss_vllm.py --no-prefix-cache
    uv run python scripts/test_gpt_oss_vllm.py --multi-first
"""
import json
import os
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from openai_harmony import (
    HarmonyEncodingName,
    HarmonyError,
    Role,
    StreamableParser,
    load_harmony_encoding,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.llm_engine import LLMEngine

from exo.shared.constants import EXO_MODELS_DIR

MODEL_ID = str(EXO_MODELS_DIR / "openai--gpt-oss-20b")

SYSTEM_PROMPT = "You are a helpful AI assistant. Respond directly and concisely. Do not show your reasoning or thought process. When files are shared with you, analyze them and respond helpfully."

MESSAGES_SINGLE = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Hi!"},
]

MESSAGES_MULTI = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Hi!"},
    {"role": "assistant", "content": "Hello! How can I help you today?"},
    {"role": "user", "content": "A lot."},
]

MESSAGES_LONG = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Hi!"},
    {"role": "assistant", "content": "Hello! How can I help you today?"},
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "And Germany?"},
    {"role": "assistant", "content": "Berlin."},
    {"role": "user", "content": "Tell me a fun fact about Berlin."},
]

MAX_TOKENS = 200

_CHANNEL_TOKEN = 200005
_MESSAGE_TOKEN = 200008


def run_generation(engine, messages: list[dict], label: str, request_id: str):
    tokenizer = engine.get_tokenizer()

    prompt_text: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    tokens_from_encode: list[int] = tokenizer.encode(prompt_text, add_special_tokens=False)
    tokens_from_template: list[int] = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    print(f"\n{'=' * 120}")
    print(f"=== {label} ===")
    print(f"{'=' * 120}")
    print(f"PROMPT TEXT:\n{prompt_text}\n")
    print(f"TOKENS FROM encode() ({len(tokens_from_encode)}): {tokens_from_encode}")
    print(f"TOKENS FROM template  ({len(tokens_from_template)}): {tokens_from_template}")
    if tokens_from_encode != tokens_from_template:
        print("  !! MISMATCH! Diffs:")
        for i, (a, b) in enumerate(zip(tokens_from_encode, tokens_from_template)):
            if a != b:
                print(f"    pos {i}: encode={a} ({tokenizer.decode([a])!r}) vs template={b} ({tokenizer.decode([b])!r})")
        if len(tokens_from_encode) != len(tokens_from_template):
            print(f"    length diff: encode={len(tokens_from_encode)} vs template={len(tokens_from_template)}")
    else:
        print("  (tokens match)")

    prompt_tokens = tokens_from_template

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    parser = StreamableParser(encoding, role=Role.ASSISTANT)

    thinking = False
    current_tool_name: str | None = None
    tool_arg_parts: list[str] = []

    print(f"{'idx':>4} | {'token_id':>8} | {'vllm_text':>20} | {'harm_delta':>25} | {'channel':>12} | {'recipient':>25} | {'yield':>40}")
    print("-" * 160)

    sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    engine.add_request(request_id, {"prompt_token_ids": prompt_tokens}, sampling_params)

    _IDLE, _EXPECT_NAME, _EXPECT_MSG = 0, 1, 2
    header_state = _IDLE

    all_tokens: list[int] = []
    all_yielded: list[str] = []
    token_index = 0
    prev_token_count = 0
    prev_text = ""

    while engine.has_unfinished_requests():
        outputs = engine.step()
        for output in outputs:
            if output.request_id != request_id:
                continue
            completion = output.outputs[0]
            new_tokens = completion.token_ids[prev_token_count:]
            finish_reason = completion.finish_reason
            prev_token_count = len(completion.token_ids)
            prev_text = completion.text

            for j, token_id in enumerate(new_tokens):
                is_last = j == len(new_tokens) - 1
                finished = is_last and finish_reason is not None

                vllm_text = tokenizer.decode([token_id])
                all_tokens.append(token_id)

                if header_state == _EXPECT_MSG and token_id != _MESSAGE_TOKEN:
                    parser.process(_MESSAGE_TOKEN)
                    header_state = _IDLE
                elif header_state == _EXPECT_MSG:
                    header_state = _IDLE
                elif header_state == _EXPECT_NAME:
                    header_state = _EXPECT_MSG

                if token_id == _CHANNEL_TOKEN:
                    header_state = _EXPECT_NAME

                try:
                    parser.process(token_id)
                except HarmonyError as e:
                    print(f"  !! HarmonyError at token {token_index}: {e}")
                    engine.abort_request([request_id])
                    return

                delta = parser.last_content_delta
                ch = parser.current_channel
                recipient = parser.current_recipient

                yielded = ""

                effective_recipient = recipient if (recipient is not None and recipient.startswith("functions.")) else None
                if effective_recipient != current_tool_name:
                    if current_tool_name is not None:
                        tool_name = current_tool_name.removeprefix("functions.")
                        args = "".join(tool_arg_parts).strip()
                        yielded = f"TOOL_CALL({tool_name}, {args!r})"
                        tool_arg_parts = []
                    current_tool_name = effective_recipient

                if current_tool_name is not None:
                    if delta:
                        tool_arg_parts.append(delta)
                    if finished:
                        yielded = f"TOOL_FINISH({json.dumps(''.join(tool_arg_parts))})"
                        tool_arg_parts = []
                else:
                    is_suppressed = ch == "analysis" or (recipient is not None and recipient.startswith("!"))
                    if is_suppressed and not thinking:
                        thinking = True
                    if not is_suppressed and thinking:
                        thinking = False
                    if delta:
                        prefix = "[THINK] " if thinking else ""
                        yielded = f"{prefix}{delta!r}"
                    if finished:
                        yielded += f" [FINISH={finish_reason}]"

                all_yielded.append(yielded)
                print(f"{token_index:4d} | {token_id:8d} | {vllm_text!r:>20} | {delta!r:>25} | {str(ch):>12} | {str(recipient):>25} | {yielded:>40}")
                token_index += 1

            if finish_reason is not None:
                print(f"\nFinish reason: {finish_reason}")

    print(f"\n--- RAW TOKEN IDS ({len(all_tokens)}) ---")
    print(all_tokens)
    print("\n--- FINAL TEXT ---")
    final_text = ""
    for y in all_yielded:
        if y.startswith("[THINK] "):
            continue
        if y.startswith("TOOL_"):
            continue
        clean = y.replace(" [FINISH=stop]", "").replace(" [FINISH=length]", "")
        if clean.startswith("'") and clean.endswith("'") or clean.startswith('"') and clean.endswith('"'):
            final_text += clean[1:-1]
    print(repr(final_text))


def main():
    no_prefix_cache = "--no-prefix-cache" in sys.argv
    multi_first = "--multi-first" in sys.argv

    print(f"Loading vLLM engine from {MODEL_ID}")
    print(f"  prefix_caching: {'DISABLED' if no_prefix_cache else 'DEFAULT'}")
    print(f"  order: {'multi-first' if multi_first else 'single-first'}")

    kwargs = {}
    if no_prefix_cache:
        kwargs["enable_prefix_caching"] = False

    engine_args = EngineArgs(
        model=MODEL_ID,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        load_format="fastsafetensors",
        **kwargs,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    tokenizer = engine.get_tokenizer()
    eos_ids = getattr(tokenizer, "eos_token_id", None)
    print(f"VLLM: Engine loaded. eos_token_id from tokenizer: {eos_ids}")

    if multi_first:
        run_generation(engine, MESSAGES_MULTI, "MULTI TURN: Hi! -> A lot.", "test-multi")
        run_generation(engine, MESSAGES_SINGLE, "SINGLE TURN: Hi!", "test-single")
        run_generation(engine, MESSAGES_LONG, "LONG CONVO: 5 turns", "test-long")
    else:
        run_generation(engine, MESSAGES_SINGLE, "SINGLE TURN: Hi!", "test-single")
        run_generation(engine, MESSAGES_MULTI, "MULTI TURN: Hi! -> A lot.", "test-multi")
        run_generation(engine, MESSAGES_LONG, "LONG CONVO: 5 turns", "test-long")

    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
