#!/usr/bin/env python3
"""Test GPT OSS output parsing with MLX backend.

Run on Mac:
    uv run python scripts/test_gpt_oss_mlx.py

Replicates exo's exact flow:
  mlx_lm.stream_generate -> GenerationResponse -> parse_gpt_oss -> output
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

from exo.shared.constants import EXO_MODELS_DIR
from openai_harmony import (
    HarmonyEncodingName,
    HarmonyError,
    Role,
    StreamableParser,
    load_harmony_encoding,
)

MODEL_PATH = str(EXO_MODELS_DIR / "mlx-community--gpt-oss-20b-MXFP4-Q8")

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


def run_generation(model, tokenizer, messages: list[dict], label: str):
    prompt_text: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_tokens: list[int] = tokenizer.encode(prompt_text, add_special_tokens=False)

    print(f"\n{'=' * 120}")
    print(f"=== {label} ===")
    print(f"{'=' * 120}")
    print(f"PROMPT TEXT:\n{prompt_text}\n")
    print(f"PROMPT TOKENS ({len(prompt_tokens)}): {prompt_tokens}\n")

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    parser = StreamableParser(encoding, role=Role.ASSISTANT)

    thinking = False
    current_tool_name: str | None = None
    tool_arg_parts: list[str] = []

    print(f"{'idx':>4} | {'token_id':>8} | {'mlx_text':>20} | {'harm_delta':>25} | {'channel':>12} | {'recipient':>25} | {'yield':>40}")
    print("-" * 160)

    prompt_array = mx.array(prompt_tokens)

    _CHANNEL_TOKEN = 200005
    _MESSAGE_TOKEN = 200008
    _IDLE, _EXPECT_NAME, _EXPECT_MSG = 0, 1, 2
    header_state = _IDLE

    all_tokens: list[int] = []
    all_yielded: list[str] = []

    for i, out in enumerate(stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_array,
        max_tokens=MAX_TOKENS,
        sampler=make_sampler(temp=0.0),
    )):
        token_id = int(out.token)
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
            print(f"  !! HarmonyError at token {i}: {e}")
            break

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
            if out.finish_reason is not None:
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
            if out.finish_reason is not None:
                yielded += f" [FINISH={out.finish_reason}]"

        all_yielded.append(yielded)
        print(f"{i:4d} | {token_id:8d} | {out.text!r:>20} | {delta!r:>25} | {str(ch):>12} | {str(recipient):>25} | {yielded:>40}")

        if out.finish_reason is not None:
            break

    print(f"\n--- RAW TOKEN IDS ({len(all_tokens)}) ---")
    print(all_tokens)
    print(f"\n--- FINAL TEXT ---")
    text_parts = [y for y in all_yielded if y and not y.startswith("TOOL_") and not y.endswith("]") or "[THINK]" in y]
    final_text = ""
    for y in all_yielded:
        if y.startswith("[THINK] "):
            continue
        if y.startswith("TOOL_"):
            continue
        clean = y.replace(" [FINISH=stop]", "").replace(" [FINISH=length]", "")
        if clean.startswith("'") and clean.endswith("'"):
            final_text += clean[1:-1]
        elif clean.startswith('"') and clean.endswith('"'):
            final_text += clean[1:-1]
    print(repr(final_text))


def main():
    print(f"Loading model from {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)
    print(f"Model loaded. EOS token IDs: {tokenizer.eos_token_ids}")

    run_generation(model, tokenizer, MESSAGES_SINGLE, "SINGLE TURN: Hi!")
    run_generation(model, tokenizer, MESSAGES_MULTI, "MULTI TURN: Hi! -> A lot.")
    run_generation(model, tokenizer, MESSAGES_LONG, "LONG CONVO: 5 turns")


if __name__ == "__main__":
    main()
