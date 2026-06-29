#!/usr/bin/env python3
"""Generate meaningful bench prompts for exo_models_bench.py.

Usage:
    uv run python gen_bench_prompts.py --tokens 512 1024 2048 4096
    uv run python gen_bench_prompts.py --tokens 512 --out my_prompts.json
    uv run python gen_bench_prompts.py --tokens 512,1024,2048
"""
import argparse
import itertools
import json

# Diverse sentences across domains for realistic token distribution.
# Covers science, history, philosophy, tech, literature — varied vocabulary
# and syntax to exercise the model's attention patterns properly.
_CORPUS = [
    "The mitochondria is the powerhouse of the cell, converting glucose into ATP through a process called oxidative phosphorylation.",
    "Quantum entanglement allows particles to remain correlated regardless of the distance separating them, defying classical intuitions about locality.",
    "The French Revolution began in 1789 and fundamentally transformed European political thought, replacing monarchical authority with popular sovereignty.",
    "Neural networks learn by adjusting weights through backpropagation, minimizing the difference between predicted and actual outputs over many iterations.",
    "Shakespeare's tragedies explore the tension between individual ambition and social order, often culminating in the protagonist's inevitable downfall.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen, sustaining nearly all life on Earth.",
    "The Turing test proposes that a machine exhibits intelligent behavior if its responses are indistinguishable from those of a human evaluator.",
    "Continental drift, proposed by Alfred Wegener in 1912, explains how Earth's landmasses have moved over geological timescales.",
    "Keynesian economics argues that government spending can stimulate aggregate demand during recessions, reducing unemployment and stabilizing output.",
    "The human genome contains approximately three billion base pairs encoding around twenty thousand protein-coding genes.",
    "Rust's ownership model enforces memory safety at compile time, eliminating entire classes of bugs such as dangling pointers and data races.",
    "The Apollo 11 mission landed the first humans on the Moon on July 20, 1969, fulfilling Kennedy's decade-long national commitment.",
    "In thermodynamics, entropy is a measure of disorder in a system; the second law states that total entropy never decreases in an isolated system.",
    "Plato's allegory of the cave illustrates how perception can be mistaken for reality, and how philosophical inquiry can liberate the mind.",
    "Distributed systems must balance consistency, availability, and partition tolerance, as formalized by the CAP theorem.",
    "The Great Wall of China was built over many centuries by successive dynasties to protect against nomadic invasions from the north.",
    "CRISPR-Cas9 enables precise editing of DNA sequences, offering potential cures for genetic diseases and new tools for biological research.",
    "Supply chains became a central focus of economic policy after pandemic disruptions exposed the fragility of just-in-time manufacturing.",
    "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape beyond the event horizon.",
    "The Silk Road connected East Asia to the Mediterranean, facilitating trade in silk, spices, and ideas across thousands of miles.",
    "Transformers revolutionized natural language processing by replacing recurrence with self-attention, enabling efficient parallel training at scale.",
    "Climate change is driven primarily by the accumulation of greenhouse gases, which trap infrared radiation and raise global surface temperatures.",
    "The immune system distinguishes self from non-self through major histocompatibility complex proteins, enabling targeted destruction of pathogens.",
    "Socrates argued that the unexamined life is not worth living, positioning philosophical dialogue as the highest human activity.",
    "Moore's law observed that the number of transistors on a chip doubles roughly every two years, driving decades of exponential performance gains.",
    "The theory of relativity demonstrates that space and time are interwoven into a single continuum, curved by mass and energy.",
    "Democracy requires not only free elections but also an informed citizenry, independent judiciary, and protection of minority rights.",
    "The Amazon rainforest, often called the lungs of the Earth, produces twenty percent of the world's oxygen and hosts ten percent of all species.",
    "Blockchain achieves decentralized consensus through cryptographic hashing and proof-of-work or proof-of-stake validation mechanisms.",
    "Medieval guilds regulated trade and craftsmanship in European cities, establishing standards of quality and controlling entry into skilled professions.",
    "Fermentation allows microorganisms to convert sugars into alcohol and carbon dioxide, a process exploited in breadmaking and brewing for millennia.",
    "The observer effect in quantum mechanics states that measuring a system inevitably disturbs it, making uncertainty fundamental rather than incidental.",
    "Urbanization has accelerated over the past century, with more than half of the world's population now living in cities for the first time in history.",
    "Game theory analyzes strategic interactions where the outcome for each participant depends on the choices of all others involved.",
    "Van Gogh produced over nine hundred paintings in a decade, pioneering post-impressionism through bold color and expressive, swirling brushwork.",
]

# ponytail: rough estimate, avoids needing a tokenizer in this script; real counts verified by exo_models_bench.py
_WORDS_PER_TOKEN = 0.75  # conservative: more words generated than needed, bench script trims


def build_text(target_tokens: int, offset: int = 0) -> str:
    target_words = int(target_tokens / _WORDS_PER_TOKEN)
    start = offset % len(_CORPUS)
    rotated = _CORPUS[start:] + _CORPUS[:start]
    words: list[str] = []
    for sentence in itertools.cycle(rotated):
        words.extend(sentence.split())
        if len(words) >= target_words:
            break
    return " ".join(words)


def parse_token_list(values: list[str]) -> list[int]:
    items: list[int] = []
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if part:
                items.append(int(part))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate meaningful bench prompts and save to JSON."
    )
    ap.add_argument(
        "--tokens",
        nargs="+",
        required=True,
        metavar="N",
        help="Target token counts (space or comma separated). E.g. --tokens 512 1024 2048",
    )
    ap.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of prompts to generate per token size (default: 1)",
    )
    ap.add_argument(
        "--out",
        default="bench_prompts.json",
        help="Output JSON file (default: bench_prompts.json)",
    )
    args = ap.parse_args()

    token_list = parse_token_list(args.tokens)
    if not token_list or any(t <= 0 for t in token_list):
        ap.error("--tokens must be positive integers")
    if args.count <= 0:
        ap.error("--count must be >= 1")

    prompts: dict[str, list[str]] = {}
    for n in sorted(set(token_list)):
        prompts[str(n)] = [build_text(n, i) for i in range(args.count)]
        print(f"  generated {args.count}x ~{n} token prompt")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"prompts": prompts}, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(prompts)} prompts → {args.out}")


if __name__ == "__main__":
    main()
