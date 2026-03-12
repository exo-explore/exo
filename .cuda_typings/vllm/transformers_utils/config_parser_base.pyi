import abc
from abc import ABC, abstractmethod
from pathlib import Path
from transformers import PretrainedConfig as PretrainedConfig

class ConfigParserBase(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]: ...
