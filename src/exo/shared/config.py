from exo_rs import LocatorArgs, LocatorConfig

# TODO: for now we are only going to have "load once" globals as our configuration,
#       so "reloading" isn't supported yet; would have to figure out in the future
#       how to create "reactive" configuration that can be reloaded and injected and whatnot
_locator_config: LocatorConfig | None = None


def locator() -> LocatorConfig:
    if _locator_config is None:
        raise RuntimeError("Configuration object accessed before loaded")
    return _locator_config


def load_locator(args: LocatorArgs):
    global _locator_config
    if _locator_config is not None:
        raise RuntimeError("Configuration already loaded")
    _locator_config = LocatorConfig.resolve(args)
