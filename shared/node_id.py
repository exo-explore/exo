import logging
from multiprocessing import Lock
from multiprocessing.synchronize import Lock as LockT
from typing import Optional, TypedDict

from exo_pyo3_bindings import Keypair

from shared.constants import EXO_NODE_ID_KEYPAIR

"""
This file is responsible for concurrent race-free persistent node-ID retrieval.
"""

class _NodeIdGlobal(TypedDict):
    file_lock: LockT
    keypair: Optional[Keypair]

_NODE_ID_GLOBAL: _NodeIdGlobal = {
    "file_lock": Lock(),
    "keypair": None,
}

def get_node_id_keypair() -> Keypair:
    """
    Obtains the :class:`Keypair` associated with this node-ID.
    Obtain the :class:`PeerId` by from it.
    """

    # get from memory if we have it => read from file otherwise
    if _NODE_ID_GLOBAL["keypair"] is not None:
        return _NODE_ID_GLOBAL["keypair"]

    # operate with cross-process lock to avoid race conditions
    with _NODE_ID_GLOBAL["file_lock"]:
        with open(EXO_NODE_ID_KEYPAIR, 'a+b') as f: # opens in append-mode => starts at EOF
            # if non-zero EOF, then file exists => use to get node-ID
            if f.tell() != 0:
                f.seek(0) # go to start & read protobuf-encoded bytes
                protobuf_encoded = f.read()

                try: # if decoded successfully, save & return
                    _NODE_ID_GLOBAL["keypair"] = Keypair.from_protobuf_encoding(protobuf_encoded)
                    return _NODE_ID_GLOBAL["keypair"]
                except RuntimeError as e: # on runtime error, assume corrupt file
                    logging.warning(f"Encountered runtime error when trying to get keypair: {e}")

        # if no valid credentials, create new ones and persist
        with open(EXO_NODE_ID_KEYPAIR, 'w+b') as f:
            _NODE_ID_GLOBAL["keypair"] = Keypair.generate_ed25519()
            f.write(_NODE_ID_GLOBAL["keypair"].to_protobuf_encoding())
            return _NODE_ID_GLOBAL["keypair"]