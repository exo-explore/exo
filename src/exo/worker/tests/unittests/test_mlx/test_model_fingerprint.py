import json

from exo.worker.engines.mlx.model_fingerprint import (
    is_gemma3,
    is_glm,
    is_kimi_tokenizer_repo,
    load_model_fingerprint,
)


def test_load_model_fingerprint_prefers_text_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "ignored",
                "architectures": ["Ignored"],
                "text_config": {"model_type": "glm", "architectures": ["ChatGLMModel"]},
            }
        )
    )

    fp = load_model_fingerprint(tmp_path)
    assert fp.model_type == "glm"
    assert fp.architectures == ("ChatGLMModel",)
    assert is_glm(fp)


def test_is_gemma3_detects_model_type(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma3"}))
    fp = load_model_fingerprint(tmp_path)
    assert is_gemma3(fp)


def test_is_gemma3_detects_architecture(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Gemma3ForCausalLM"]})
    )
    fp = load_model_fingerprint(tmp_path)
    assert is_gemma3(fp)


def test_is_kimi_tokenizer_repo_detects_custom_tokenizer_file(tmp_path):
    (tmp_path / "tokenization_kimi.py").write_text("# test")
    assert is_kimi_tokenizer_repo(tmp_path)

