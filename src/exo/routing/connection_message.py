from exo_net import PyFromSwarm

from exo.utils.pydantic_ext import FrozenModel

"""Serialisable types for Connection Updates/Messages"""


class ConnectionMessage(FrozenModel):
    connected: bool

    @classmethod
    def from_update(cls, update: PyFromSwarm.Connection) -> "ConnectionMessage":
        return cls(connected=update.connected)
