import os
import tomllib
from pathlib import Path

import anyio
from loguru import logger
from pydantic import Field, PositiveInt, ValidationError

from exo.utils.pydantic_ext import FrozenModel, TaggedModel


class ApiConfig(FrozenModel):
    host: str = "0.0.0.0"
    port: PositiveInt = 52415


class NodeConfig(TaggedModel):
    """Node configuration from EXO_CONFIG_FILE, loaded once at startup."""

    api: ApiConfig = Field(default_factory=ApiConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> "NodeConfig":
        if path is None:
            from exo.shared.constants import EXO_CONFIG_FILE

            path = EXO_CONFIG_FILE

        if not path.exists():
            return _apply_env_overrides(cls())

        try:
            contents = path.read_text(encoding="utf-8")
            data = tomllib.loads(contents) if contents.strip() else {}
            config = cls.model_validate(data)
        except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as exception:
            raise RuntimeError(f"Failed to read exo config file at {path}") from exception
        except ValidationError as exception:
            raise RuntimeError(f"Invalid exo config file at {path}") from exception

        return _apply_env_overrides(config)

    @classmethod
    async def gather(cls) -> "NodeConfig | None":
        from exo.shared.constants import EXO_CONFIG_FILE

        cfg_file = anyio.Path(EXO_CONFIG_FILE)
        await cfg_file.parent.mkdir(parents=True, exist_ok=True)
        await cfg_file.touch(exist_ok=True)
        try:
            return cls.load(EXO_CONFIG_FILE)
        except RuntimeError as exception:
            logger.opt(exception=exception).warning("Invalid config file, skipping...")
            return None


def _apply_env_overrides(config: NodeConfig) -> NodeConfig:
    return NodeConfig(
        api=ApiConfig(
            host=_env_str("EXO_API_HOST") or config.api.host,
            port=_env_int("EXO_API_PORT") or config.api.port,
        )
    )


def _env_str(name: str) -> str | None:
    return os.getenv(name)


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exception:
        raise RuntimeError(f"{name} must be an integer") from exception
