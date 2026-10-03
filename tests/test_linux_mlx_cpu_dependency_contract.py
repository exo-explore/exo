import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_linux_mlx_cpu_extra_uses_a_matching_registry_pair() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    cpu_requirements = pyproject["project"]["optional-dependencies"]["mlx-cpu"]
    mlx_requirement = next(
        requirement
        for requirement in pyproject["project"]["optional-dependencies"]["mlx"]
        if requirement.startswith("mlx==")
    )
    cpu_requirement = next(
        requirement
        for requirement in cpu_requirements
        if requirement.startswith("mlx-cpu==")
    )
    mlx_version = mlx_requirement.removeprefix("mlx==")
    cpu_version = cpu_requirement.removeprefix("mlx-cpu==").split(";", 1)[0]

    assert cpu_version == mlx_version
    linux_sources = [
        source
        for source in pyproject["tool"]["uv"]["sources"]["mlx"]
        if "sys_platform == 'linux'" in source.get("marker", "")
    ]
    assert linux_sources
    assert all(
        "extra == 'mlx-cuda12'" in source["marker"]
        and "extra == 'mlx-cuda13'" in source["marker"]
        for source in linux_sources
    )


if __name__ == "__main__":
    test_linux_mlx_cpu_extra_uses_a_matching_registry_pair()
