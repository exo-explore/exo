from __future__ import annotations

import logging
import os
from pathlib import Path

from exo_pyo3_bindings import Keypair
from filelock import FileLock

from shared.constants import EXO_NODE_ID_KEYPAIR

"""
This file is responsible for concurrent race-free persistent node-ID retrieval.
"""


def _lock_path(path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> Path:
    return Path(str(path) + ".lock")


def get_node_id_keypair(path: str | bytes | os.PathLike[str] | os.PathLike[bytes] = EXO_NODE_ID_KEYPAIR) -> Keypair:
    """
    Obtains the :class:`Keypair` associated with this node-ID.
    Obtain the :class:`PeerId` by from it.
    """

    # operate with cross-process lock to avoid race conditions
    with FileLock(_lock_path(path)):
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
