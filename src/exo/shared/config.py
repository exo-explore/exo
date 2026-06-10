from exo_rs import LocatorArgs, LocatorConfig

_locator_config: LocatorConfig | None = None


def locator_config() -> LocatorConfig:
    if _locator_config is None:
        raise RuntimeError("Configuration object accessed before loaded")
    return _locator_config


def load_locator_config(args: LocatorArgs):
    global _locator_config
    if _locator_config is not None:
        raise RuntimeError("Configuration already loaded")
    _locator_config = LocatorConfig.resolve(args)
