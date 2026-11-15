from dataclasses import dataclass


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class Heartbeat:
    pass


class EventDecoder:
    def __init__(self, limit: int = 1024 * 1024):
        self._limit = limit
        self._buffer = bytearray()
        self._data: list[bytes] = []
        self._size = 0

    def feed(self, content: bytes) -> list[str | Heartbeat]:
        self._buffer.extend(content)
        events: list[str | Heartbeat] = []
        while (end := self._buffer.find(b"\n")) >= 0:
            line = bytes(self._buffer[:end]).removesuffix(b"\r")
            del self._buffer[: end + 1]
            if len(line) + self._size > self._limit:
                raise ProtocolError("TensorFold stream event exceeds its size limit")
            if not line:
                if self._data:
                    events.append(b"\n".join(self._data).decode("utf-8"))
                    self._data.clear()
                    self._size = 0
            elif line.startswith(b":"):
                events.append(Heartbeat())
            elif line.startswith(b"data:"):
                data = line[5:].removeprefix(b" ")
                self._data.append(data)
                self._size += len(data) + 1
        if len(self._buffer) + self._size > self._limit:
            raise ProtocolError("TensorFold stream event exceeds its size limit")
        return events

    def finish(self) -> None:
        if self._buffer or self._data:
            raise ProtocolError("TensorFold stream ended inside an event")
