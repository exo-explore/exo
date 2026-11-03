"""This node's id, kept across restarts.

The cluster knows a node by its id, so keeping the id means a node that restarts comes back as
itself: the cluster sees the same device again, not a departed one and a new one.

Two nodes with one id can't both be in a cluster (they never connect to each other, and every
other node drops one of them). So the id file is tied to this machine and to the file's own
location, and a copy of it anywhere else isn't used: a disk cloned or restored onto another Mac,
or an exo folder copied to run a second exo on the same Mac, gets a new id. Only a hash of the
machine's id is stored. One exo at a time uses a given exo folder (its PID file is locked), so
one id file never has two processes.
"""

import ctypes
import hashlib
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from exo.shared.types.common import NodeId
from exo.utils.pydantic_ext import FrozenModel

# What zenoh accepts as an id: lowercase hex, at most 16 bytes, no leading zero
# (rust/networking is_valid_zid)
_VALID_NODE_ID = re.compile(r"[1-9a-f][0-9a-f]{0,31}")
# Changing this gives every node a new id once, as the stored bindings stop matching
_BINDING_SALT = b"exo node id v1"
# How long to wait for macOS to report the hardware UUID
_HOST_UUID_WAIT_SECONDS = 1
_LINUX_MACHINE_ID_FILES = (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id"))


class _StoredNodeId(FrozenModel):
    node_id: str
    # A hash of this machine's id and the file's location (_binding)
    bound_to: str


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


def _macos_hardware_uuid() -> str | None:
    """The Mac's hardware UUID (the one System Information shows), from gethostuuid(2)."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        gethostuuid = libc.gethostuuid
    except (OSError, AttributeError):
        return None
    gethostuuid.argtypes = [ctypes.c_char_p, ctypes.POINTER(_Timespec)]
    gethostuuid.restype = ctypes.c_int
    uuid = ctypes.create_string_buffer(16)
    wait = _Timespec(tv_sec=_HOST_UUID_WAIT_SECONDS, tv_nsec=0)
    result: int = gethostuuid(uuid, ctypes.byref(wait))  # pyright: ignore[reportAny]
    if result != 0:
        return None
    raw = uuid.raw
    return raw.hex() if any(raw) else None


def _linux_machine_id() -> str | None:
    for path in _LINUX_MACHINE_ID_FILES:
        try:
            machine_id = path.read_text().strip()
        except OSError:
            continue
        if machine_id:
            return machine_id
    return None


def machine_id() -> str | None:
    """An id of this machine that a copy of its disk on another machine doesn't share, or None
    if there isn't one."""
    if sys.platform == "darwin":
        return _macos_hardware_uuid()
    if sys.platform == "linux":
        return _linux_machine_id()
    return None


def is_valid_node_id(node_id: str) -> bool:
    return _VALID_NODE_ID.fullmatch(node_id) is not None


def random_node_id() -> NodeId:
    while True:
        node_id = os.urandom(16).hex().lstrip("0")
        if is_valid_node_id(node_id):
            return NodeId(node_id)


def _binding(machine: str, path: Path) -> str:
    """What the id file is tied to: this machine and the file's own location."""
    digest = hashlib.sha256(_BINDING_SALT)
    for part in (machine, str(path.resolve())):
        digest.update(b"\0")
        digest.update(part.encode())
    return digest.hexdigest()


def _read_stored(path: Path, bound_to: str) -> tuple[NodeId | None, str]:
    """The stored id if it can be used here, else None and why not."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None, "none was saved yet"
    except OSError as error:
        return None, f"its file couldn't be read ({error})"
    try:
        stored = _StoredNodeId.model_validate_json(text)
    except ValidationError:
        return None, "its file is unreadable"
    if not is_valid_node_id(stored.node_id):
        return None, "its file is unreadable"
    if stored.bound_to != bound_to:
        return None, (
            "its file was made on another machine or in another folder "
            "(a cloned or restored disk, or a copied exo folder)"
        )
    return NodeId(stored.node_id), ""


def _save(path: Path, node_id: NodeId, bound_to: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(
        os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w"
    ) as file:
        file.write(_StoredNodeId(node_id=node_id, bound_to=bound_to).model_dump_json())
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def load_or_create_node_id(
    path: Path, get_machine_id: Callable[[], str | None] = machine_id
) -> NodeId:
    """This node's id: the one saved at `path` if it was saved on this machine at this path,
    otherwise a new one, saved there for the next start.

    With no machine id to tie the file to, nothing is saved and every start gets a new id."""
    machine = get_machine_id()
    if machine is None:
        node_id = random_node_id()
        logger.info(
            f"Node id {node_id} is new for this run: this machine has no id to tie a saved one to"
        )
        return node_id
    bound_to = _binding(machine, path)
    stored, why_not = _read_stored(path, bound_to)
    if stored is not None:
        logger.info(f"Node id {stored} (kept from earlier runs, in {path})")
        return stored
    node_id = random_node_id()
    try:
        _save(path, node_id, bound_to)
    except OSError as error:
        logger.warning(
            f"Node id {node_id} is new and couldn't be saved to {path} ({error}): "
            "the next start will have another one"
        )
        return node_id
    logger.info(f"Node id {node_id} is new, as {why_not}; saved in {path}")
    return node_id
