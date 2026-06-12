import os
from importlib.metadata import version

# set __version__ and env-var
__version__ = version("exo")
os.environ["EXO_PKG_VERSION"] = __version__
