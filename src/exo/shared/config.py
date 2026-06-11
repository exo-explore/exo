from exo_rs import AppSettings, BootstrapSettings

# TODO: for now we are only going to have mutable globals as our configuration,
#       so reactive reload/injection still needs a real design.
_bootstrap_settings: BootstrapSettings | None = None
_app_settings: AppSettings | None = None


def bootstrap() -> BootstrapSettings:
    global _bootstrap_settings
    if _bootstrap_settings is None:
        _bootstrap_settings = BootstrapSettings.from_env_only()
    return _bootstrap_settings


def app() -> AppSettings:
    global _app_settings
    if _app_settings is None:
        _app_settings = AppSettings.from_env_only()
    return _app_settings


def load_bootstrap(settings: BootstrapSettings):
    global _bootstrap_settings
    _bootstrap_settings = settings


def load_app(settings: AppSettings):
    global _app_settings
    _app_settings = settings


def load(bootstrap_settings: BootstrapSettings, app_settings: AppSettings):
    load_bootstrap(bootstrap_settings)
    load_app(app_settings)
