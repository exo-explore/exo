import tomllib
from pathlib import Path
from typing import cast

from exo.shared.constants import RESOURCES_DIR

TWO_BIT_MODEL_CARD_FILENAMES = (
    "mlx-community--Qwen2.5-0.5B-Instruct-2bit-mlx.toml",
    "mlx-community--Llama-3-8B-Instruct-262k-2bit.toml",
    "mlx-community--Llama-3-70B-Instruct-Gradient-262k-2bit.toml",
)


def test_two_bit_model_cards_exist_and_parse_valid_toml() -> None:
    """Ensure all introduced 2-bit model cards exist in the inference directory and parse correctly."""
    cards_directory = Path(RESOURCES_DIR) / "inference_model_cards"

    for filename in TWO_BIT_MODEL_CARD_FILENAMES:
        card_path = cards_directory / filename
        assert card_path.is_file(), f"Model card file {filename} does not exist in {cards_directory}"

        raw_content = card_path.read_text(encoding="utf-8")
        parsed_data = tomllib.loads(raw_content)

        assert isinstance(parsed_data, dict)
        assert "model_id" in parsed_data
        assert "n_layers" in parsed_data
        assert "hidden_size" in parsed_data
        assert "storage_size" in parsed_data
        assert parsed_data.get("quantization") == "2bit"
        assert parsed_data["n_layers"] > 0
        assert parsed_data["hidden_size"] > 0
        assert parsed_data["storage_size"]["in_bytes"] > 0


def test_two_bit_storage_size_is_lower_than_four_bit_counterpart() -> None:
    """Verify that 2-bit quantization achieves a significant reduction in storage footprint."""
    cards_directory = Path(RESOURCES_DIR) / "inference_model_cards"

    two_bit_eight_billion_card = cards_directory / "mlx-community--Llama-3-8B-Instruct-262k-2bit.toml"
    four_bit_eight_billion_card = cards_directory / "mlx-community--Meta-Llama-3.1-8B-Instruct-4bit.toml"

    assert two_bit_eight_billion_card.is_file()
    assert four_bit_eight_billion_card.is_file()

    two_bit_data = tomllib.loads(two_bit_eight_billion_card.read_text(encoding="utf-8"))
    four_bit_data = tomllib.loads(four_bit_eight_billion_card.read_text(encoding="utf-8"))

    two_bit_bytes = cast(int, two_bit_data["storage_size"]["in_bytes"])
    four_bit_bytes = cast(int, four_bit_data["storage_size"]["in_bytes"])

    # 2-bit quantization must require substantially less memory than 4-bit
    assert two_bit_bytes < four_bit_bytes
    assert two_bit_bytes < (four_bit_bytes * 0.65)


def test_two_bit_model_cards_validate_with_pydantic_model_card() -> None:
    """Validate that all 2-bit model cards strictly deserialize into the Pydantic ModelCard model."""
    from exo.shared.models.model_cards import ModelCard

    cards_directory = Path(RESOURCES_DIR) / "inference_model_cards"

    for filename in TWO_BIT_MODEL_CARD_FILENAMES:
        card_path = cards_directory / filename
        parsed_dictionary = tomllib.loads(card_path.read_text(encoding="utf-8"))

        validated_card = ModelCard.model_validate(parsed_dictionary)
        assert validated_card.quantization == "2bit"
        assert validated_card.storage_size.in_bytes > 0
        assert validated_card.n_layers > 0
