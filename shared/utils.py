from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Type

from exo_pyo3_bindings import Keypair
from filelock import FileLock

from shared.constants import EXO_NODE_ID_KEYPAIR


def ensure_type[T](obj: Any, expected_type: Type[T]) -> T:  # type: ignore
    if not isinstance(obj, expected_type):
        raise TypeError(f"Expected {expected_type}, got {type(obj)}")  # type: ignore
    return obj


# def make_async_iter[T]():
#     """
#     Creates a pair `<async-iter>, <put-to-iter>` of an asynchronous iterator
#     and a synchronous function to put items into that iterator.
#     """
#
#     loop = asyncio.get_event_loop()
#     queue: asyncio.Queue[T] = asyncio.Queue()
#
#     def put(c: ConnectionUpdate) -> None:
#         loop.call_soon_threadsafe(queue.put_nowait, (c,))
#
#     async def get():
#         while True:
#             yield await queue.get()
#
#     return get(), put

def get_node_id_keypair(path: str | bytes | os.PathLike[str] | os.PathLike[bytes] = EXO_NODE_ID_KEYPAIR) -> Keypair:
    """
    Obtains the :class:`Keypair` associated with this node-ID.
    Obtain the :class:`PeerId` by from it.
    """

    def lock_path(path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> Path:
        return Path(str(path) + ".lock")

    # operate with cross-process lock to avoid race conditions
    with FileLock(lock_path(path)):
        with open(path, 'a+b') as f:  # opens in append-mode => starts at EOF
            # if non-zero EOF, then file exists => use to get node-ID
            if f.tell() != 0:
                f.seek(0)  # go to start & read protobuf-encoded bytes
                protobuf_encoded = f.read()

                try:  # if decoded successfully, save & return
                    return Keypair.from_protobuf_encoding(protobuf_encoded)
                except RuntimeError as e:  # on runtime error, assume corrupt file
                    logging.warning(f"Encountered runtime error when trying to get keypair: {e}")

        # if no valid credentials, create new ones and persist
        with open(path, 'w+b') as f:
            keypair = Keypair.generate_ed25519()
            f.write(keypair.to_protobuf_encoding())
            return keypair
