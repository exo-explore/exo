import argparse
import json
from pathlib import Path


def recall_prompt() -> str:
    records = [
        f"Record {index:04d}: depot inventory has 17 units reserved and 23 units "
        "available for the next delivery."
        for index in range(1, 7801)
    ]
    records[0] += " START_CODE=copper-41."
    records[3899] += " MIDDLE_CODE=harbor-72."
    records[7799] += " END_CODE=spruce-93."
    return (
        "Read the following records. Return exactly the START_CODE, MIDDLE_CODE "
        "and END_CODE values, in that order, comma separated, without explanation.\n"
        + "\n".join(records)
    )


def generation_prompt() -> str:
    records = [
        f"Record {index:04d}: depot inventory has 17 units reserved and 23 units "
        "available for the next delivery."
        for index in range(7801, 9001)
    ]
    return (
        "Append these records to the previous ledger.\n"
        + "\n".join(records)
        + "\nCorrection: MIDDLE_CODE is now anchor-84. The other two codes are "
        "unchanged. Begin your answer with the three current code values in order. "
        "Then write a complete C# streaming ledger parser with a runnable example, "
        "bounded memory, cancellation, validation of record IDs and numeric "
        "quantities, and aggregation of reserved/available totals. Include focused "
        "tests for malformed records and cancellation. Work from the entire "
        "conversation, not just this last message."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit the Qwen qualification requests."
    )
    parser.add_argument(
        "model",
        choices=["mlx-community/Qwen3.8-27B-8bit", "mlx-community/Qwen3.8-27B-bf16"],
    )
    parser.add_argument("scenario", choices=["sustained", "recall", "generation"])
    parser.add_argument("--previous-answer", type=Path)
    arguments = parser.parse_args()
    if arguments.scenario == "generation" and arguments.previous_answer is None:
        parser.error(
            "generation requires --previous-answer with the exact recall answer"
        )
    if arguments.scenario == "sustained":
        messages = [
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
            {
                "role": "user",
                "content": "Write a complete C# implementation of a high-throughput, "
                "reliable UDP protocol that uses NACK-based retransmission. Include "
                "packet framing, sequence tracking, loss detection, retransmission "
                "queues, bounded memory, cancellation, and a runnable example. "
                "Include every source file and project file without ellipses, then "
                "a complete deterministic packet-loss/reordering simulator with at "
                "least 12 explicitly implemented validation scenarios, and explain "
                "the invariants each scenario checks.",
            },
        ]
    else:
        messages = [{"role": "user", "content": recall_prompt()}]
        if arguments.scenario == "generation":
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": arguments.previous_answer.read_text(
                            encoding="utf-8"
                        ),
                    },
                    {"role": "user", "content": generation_prompt()},
                ]
            )
    request = {
        "model": arguments.model,
        "stream": True,
        "temperature": 0.7 if arguments.scenario == "sustained" else 0,
        "enable_thinking": arguments.scenario != "recall",
        "logprobs": True,
        "top_logprobs": 5,
        "messages": messages,
    }
    if arguments.scenario == "sustained":
        request["reasoning_effort"] = "medium"
    else:
        request["max_tokens"] = 1024 if arguments.scenario == "recall" else 8192
    print(json.dumps(request, ensure_ascii=True))


if __name__ == "__main__":
    main()
