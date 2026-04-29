import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--model-id",
        action="store",
        default=None,
        help="HuggingFace-style model id (e.g. Qwen/Qwen3-0.6B) — must be downloaded",
    )
