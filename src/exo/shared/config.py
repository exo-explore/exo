from exo_rs import LocatorConfig

# TODO: for now we are only going to have "load once" globals as our configuration,
#       so "reloading" isn't supported yet; would have to figure out in the future
#       how to create "reactive" configuration that can be reloaded and injected and whatnot
_locator_config: LocatorConfig | None = None


def locator() -> LocatorConfig:
    global _locator_config
    if _locator_config is None:
        _locator_config = LocatorConfig.from_env_only()
    return _locator_config


def load_locator(cfg: LocatorConfig):
    global _locator_config
    _locator_config = cfg
