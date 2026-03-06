# type: ignore
"""Custom process_results for hendrycks_math that uses math_verify.

The upstream hendrycks_math task extracts answers between $ signs, which
breaks for thinking/chat models that use $ throughout their reasoning.
This version uses math_verify.parse() + verify() for robust extraction.
"""

from typing import Dict, List

import datasets
from math_verify import LatexExtractionConfig, parse, verify


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        return {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": _remove_boxed(_last_boxed_only(doc["solution"])),
        }

    return dataset.map(_process_doc)


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    candidate = results[0]
    parsed_candidate = parse(candidate)
    parsed_gold = parse(doc["solution"], extraction_config=[LatexExtractionConfig()])

    retval = 1 if verify(parsed_gold, parsed_candidate) else 0

    return {"exact_match": retval}


def _last_boxed_only(string: str) -> str:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return ""

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return ""
    return string[idx : right_brace_idx + 1]


def _remove_boxed(s: str) -> str:
    if not s:
        return s
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] == left:
            return s[len(left) :]
    left = "\\boxed{"
    if s[: len(left)] == left and s[-1] == "}":
        return s[len(left) : -1]
    return s
