import pytest

from exo.windows.launcher import (
    exit_status,
    launch_environment,
    memory_field_label,
    parse_memory_limit_megabytes,
    windows_device_choices,
)


def test_device_choices_keep_each_adapter_and_always_offer_cpu() -> None:
    choices = windows_device_choices(
        ["AMD Radeon RX 9700 XT", "NVIDIA GeForce RTX 4090", "Microsoft Basic Display"],
        has_nvidia_gpu=False,
    )
    assert [
        (choice.device_name, choice.label, choice.memory_kind) for choice in choices
    ] == [
        ("AMD", "AMD Radeon RX 9700 XT (VRAM)", "vram"),
        ("CUDA", "NVIDIA GeForce RTX 4090 (VRAM)", "vram"),
        ("CPU", "CPU (RAM)", "ram"),
    ]
    assert memory_field_label(choices[0]) == "VRAM (MB)"
    assert memory_field_label(choices[2]) == "RAM (MB)"


def test_nvml_without_an_adapter_name_adds_cuda() -> None:
    choices = windows_device_choices([], has_nvidia_gpu=True)
    assert [choice.device_name for choice in choices] == ["CUDA", "CPU"]
    assert choices[0].label == "CUDA (VRAM)"


def test_no_accelerator_offers_only_cpu() -> None:
    choices = windows_device_choices(
        ["Microsoft Basic Display Adapter"],
        has_nvidia_gpu=False,
    )
    assert [choice.device_name for choice in choices] == ["CPU"]


def test_launch_environment_sets_device_and_memory_limit() -> None:
    environment = launch_environment(
        {"PATH": "/usr/bin", "OVERRIDE_MEMORY_MB": "1"},
        "AMD",
        14000,
    )
    assert environment["PATH"] == "/usr/bin"
    assert environment["EXO_TINYGRAD_DEVICES"] == "AMD"
    assert environment["OVERRIDE_MEMORY_MB"] == "14000"


def test_blank_memory_limit_drops_a_stale_override() -> None:
    assert parse_memory_limit_megabytes("  ") is None
    environment = launch_environment(
        {"OVERRIDE_MEMORY_MB": "14000"},
        "CPU",
        None,
    )
    assert environment["EXO_TINYGRAD_DEVICES"] == "CPU"
    assert "OVERRIDE_MEMORY_MB" not in environment


def test_memory_limit_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive number of megabytes"):
        launch_environment({}, "CUDA", 0)
    with pytest.raises(ValueError, match="positive number of megabytes"):
        parse_memory_limit_megabytes("0")
    with pytest.raises(ValueError, match="positive number of megabytes"):
        parse_memory_limit_megabytes("nope")


def test_exit_status_keeps_the_import_error() -> None:
    assert exit_status(0, ["ready"]) == "exited 0"
    assert exit_status(1, ["ready", "No module named exo_rs"]) == (
        "exited 1: No module named exo_rs"
    )
