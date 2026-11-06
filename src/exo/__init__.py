from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("exo")
except PackageNotFoundError:
    # Fallback version string when running directly from source tree without installation metadata
    __version__ = "0.3.70"
