try:
    from exo_rs import FromSwarm
except ImportError:

    class FromSwarm:  # type: ignore
        class Connection:
            connected: bool


from exo.utils.pydantic_ext import FrozenModel

"""Serialisable types for Connection Updates/Messages"""


class ConnectionMessage(FrozenModel):
    connected: bool

    @classmethod
    def from_update(cls, update: FromSwarm.Connection) -> "ConnectionMessage":
        return cls(connected=update.connected)
