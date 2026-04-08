from pathlib import Path
from collections.abc import Sequence
import tomlkit
from exo.utils.pydantic_ext import FrozenModel
from typing import Self, Any
from pydantic import Field, BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource, TomlConfigSettingsSource
from exo.shared.types.common import NodeId, ModelId
from exo.shared.types.worker.instances import InstanceId
from exo.shared.constants import EXO_CONFIG_HOME, EXO_DATA_HOME, EXO_CACHE_HOME
from exo.utils.dashboard_path import find_dashboard, find_resources


def default_merge[T: BaseModel](left: T, right: T) -> T:
    if left == right:
        return left
    merged_dict = {}
    for key in type(left).model_fields:
        try:
            merged_dict[key] = getattr(left, key).merge(  # pyright: ignore[reportAny]
                getattr(right, key, None)
            )
        except AttributeError:
            raise NotImplementedError("Cluster Option using default implementation incorrectly")

    return type(left).model_validate(merged_dict)


def _parse_colon_separated_dirs(obj: Any) -> set[Path]:  # pyright: ignore[reportAny]
        if isinstance(obj, (list, set)):
            return set(Path(d).expanduser() for d in obj)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        else:
            return set(Path(d).expanduser() for d in str(obj).split(":"))  # pyright: ignore[reportAny]

class ModelDirsSettings(BaseModel, frozen=True):
    # env: EXO_MODEL_DIRS_DEFAULT prepends to WRITEABLE, defaults to EXO_DATA_HOME/models
    # env: EXO_MODEL_DIRS_WRITEABLE, defaults to []
    writeable: list[Path] = []
    # env: EXO_MODEL_DIRS_READONLY, defaults to []
    readonly: list[Path] = []

    @model_validator(mode="before")
    @classmethod
    def build_defaults(cls, data: Any) -> Any:  # pyright: ignore[reportAny]
        if not isinstance(data, dict):
            return data  # pyright: ignore[reportAny]
        default = Path(data.get("default", EXO_DATA_HOME / "models")).expanduser()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        readonly = _parse_colon_separated_dirs(data.get("readonly", []))  # pyright: ignore[reportUnknownMemberType]
        writeable = _parse_colon_separated_dirs(data.get("writeable", [])).difference(readonly)  # pyright: ignore[reportUnknownMemberType]
        if default not in readonly:
            writeable = [default, *writeable]
        return {**data, "writeable": writeable, "readonly": readonly}  # pyright: ignore[reportUnknownVariableType]

class RuntimeDirsSettings(BaseModel, frozen=True):
    dashboard: Path = Field(default_factory=find_dashboard)
    resources: Path = Field(default_factory=find_resources)
    logs: Path = EXO_CACHE_HOME / "log"
    log_file: str = "latest.log"

    def log_file_path(self):
        return self.logs / self.log_file

# doesnt require merge
class LocalSettings(FrozenModel):
    runtime_dirs: RuntimeDirsSettings
    model_dirs: ModelDirsSettings

class InstanceSettings(FrozenModel):
    # env: EXO_INSTANCE_DEFAULTS_BATCH_CONCURRENCY
    batch_concurrency: int

    def merge(self, other: Self) -> Self:
        return type(self)(batch_concurrency=min(self.batch_concurrency, other.batch_concurrency))

class ClusterSettings(FrozenModel):
    instance_defaults: InstanceSettings = InstanceSettings(batch_concurrency=8)
    model_settings_overrides: dict[ModelId, InstanceSettings] = {}

    def merge(self, other: Self) -> Self:
        return default_merge(self, other)

class SettingsFile(BaseSettings):
    model_config = SettingsConfigDict(
        extra='ignore',
        frozen=True,
        toml_file=EXO_CONFIG_HOME / "config.toml",
        env_prefix="EXO_",
        env_nested_delimiter="_",
        env_ignore_empty=True,
    )

    model_dirs: ModelDirsSettings
    runtime_dirs: RuntimeDirsSettings
    model_settings_overrides: dict[ModelId, InstanceSettings] = {}
    instance_defaults: InstanceSettings

    def get_local(self) -> LocalSettings:
        ...
    def get_cluster(self) -> ClusterSettings:
        ...

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, TomlConfigSettingsSource(settings_cls),)

    def sync(self):
        """nb: only call this once per save"""
        cfg_path = type(self).model_config.get("toml_file", None)
        if isinstance(cfg_path, Sequence):
            cfg_path=cfg_path[0]
        if cfg_path:
            with open(cfg_path, "w") as fp:
                tomlkit.dump(self.model_dump(exclude_defaults=True), fp) # pyright: ignore[reportUnknownMemberType]



class StateSettings(FrozenModel):
    per_node: dict[NodeId, LocalSettings]
    per_instance: dict[InstanceId, InstanceSettings]
    cluster: ClusterSettings
    
    def model_merge_local(self, node_id: NodeId, settings: LocalSettings) -> Self:
        return self.model_copy(update={
            "per_node": {
                **self.per_node,
                node_id: settings
            }
        })

    def model_merge_cluster(self, settings: ClusterSettings) -> Self:
        return self.model_copy(update={
            "cluster": self.cluster.merge(settings)
        })

    def settings_for(self, node_id: NodeId) -> StoredSettings:
        merged = {}
        for key, val in self.cluster.model_dump(exclude_defaults=True).items():  # pyright: ignore[reportAny]
            merged[key] = val

        if (local := self.per_node.get(node_id, None)) is not None:
            for key, val in local.model_dump(exclude_defaults=True).items():  # pyright: ignore[reportAny]
                merged[key] = val

        return StoredSettings.model_validate(merged)

    def sync(self, node_id: NodeId):
        """nb: only call this once per save"""
        toml_file=EXO_CONFIG_HOME / "config.toml"
        with open(toml_file, "w") as fp:
            tomlkit.dump(self.settings_for(node_id).model_dump(exclude_defaults=True), fp) # pyright: ignore[reportUnknownMemberType]

