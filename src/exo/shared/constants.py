import os
from pathlib import Path

from exo.utils.dashboard_path import find_dashboard, find_resources

# NOTE: I will leave these alone here for now - I don't know how I should port raw constants
EXO_MAX_CHUNK_SIZE = 512 * 1024
EXO_MAX_INSTANCE_RETRIES = 5

# NOTE: I will leave these alone here for now - I don't wanna (or know how to) port them to Rust yet
_RESOURCES_DIR_ENV = os.environ.get("EXO_RESOURCES_DIR", None)
RESOURCES_DIR = (
    find_resources() if _RESOURCES_DIR_ENV is None else Path.home() / _RESOURCES_DIR_ENV
)
_DASHBOARD_DIR_ENV = os.environ.get("EXO_DASHBOARD_DIR", None)
DASHBOARD_DIR = (
    find_dashboard() if _DASHBOARD_DIR_ENV is None else Path.home() / _DASHBOARD_DIR_ENV
)
