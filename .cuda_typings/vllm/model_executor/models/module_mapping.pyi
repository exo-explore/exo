from dataclasses import dataclass, field

@dataclass
class MultiModelKeys:
    language_model: list[str] = field(default_factory=list)
    connector: list[str] = field(default_factory=list)
    tower_model: list[str] = field(default_factory=list)
    generator: list[str] = field(default_factory=list)
    @staticmethod
    def from_string_field(
        language_model: str | list[str] = None,
        connector: str | list[str] = None,
        tower_model: str | list[str] = None,
        generator: str | list[str] = None,
        **kwargs,
    ) -> MultiModelKeys: ...
