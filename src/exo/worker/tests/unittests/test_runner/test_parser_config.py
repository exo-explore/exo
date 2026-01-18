# pyright: reportAny=false

from exo.worker.runner.runner import _get_parser_config


def test_get_parser_config_maps_known_model_ids() -> None:
    cfg = _get_parser_config("some-gpt-oss-harmony-model")
    assert cfg.reasoning_parser_name == "harmony"
    assert cfg.tool_parser_name == "harmony"

    cfg = _get_parser_config("SOLAR-OPEN")
    assert cfg.reasoning_parser_name == "solar_open"
    assert cfg.tool_parser_name == "solar_open"

    cfg = _get_parser_config("qwen3-coder")
    assert cfg.reasoning_parser_name == "qwen3"
    assert cfg.tool_parser_name == "qwen3_coder"

    cfg = _get_parser_config("nemotron-3-nano")
    assert cfg.reasoning_parser_name == "nemotron3_nano"
    assert cfg.tool_parser_name == "nemotron3_nano"

    cfg = _get_parser_config("some-unknown-model")
    assert cfg.reasoning_parser_name is None
    assert cfg.tool_parser_name is None
