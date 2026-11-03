"""A node keeps its id across restarts, but a copy of its id file on another machine or in
another folder doesn't make a second node with the same id."""

import shutil
import stat
import subprocess
import sys
from pathlib import Path

from exo.routing.node_identity import (
    is_valid_node_id,
    load_or_create_node_id,
    machine_id,
)
from exo.routing.router import get_node_zid

THIS_MAC = "5e1f0000a1b2c3d4e5f60718293a4b5c"
OTHER_MAC = "0c1d2e3f405162738495a6b7c8d9eaf1"


def _this_mac() -> str:
    return THIS_MAC


def _other_mac() -> str:
    return OTHER_MAC


def _no_machine_id() -> None:
    return None


def test_a_restarted_node_keeps_its_id(tmp_path: Path):
    path = tmp_path / "node_zid"
    first = get_node_zid(path)
    assert get_node_zid(path) == first
    assert get_node_zid(path) == first


def test_the_id_is_kept_across_processes(tmp_path: Path):
    path = tmp_path / "node_zid"
    program = (
        "import sys; from pathlib import Path; from exo.routing.router import get_node_zid; "
        "print(get_node_zid(Path(sys.argv[1])))"
    )

    def start() -> str:
        return subprocess.run(
            [sys.executable, "-c", program, str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    first = start()
    assert is_valid_node_id(first)
    assert start() == first


def test_ids_are_valid_zenoh_ids(tmp_path: Path):
    for attempt in range(50):
        node_id = load_or_create_node_id(tmp_path / f"node_zid_{attempt}", _this_mac)
        assert is_valid_node_id(node_id)
        assert len(node_id) <= 32
        assert not node_id.startswith("0")


def test_a_disk_cloned_onto_another_machine_gets_a_new_id(tmp_path: Path):
    path = tmp_path / "node_zid"
    original = load_or_create_node_id(path, _this_mac)
    # The same file, read on another Mac (Time Machine restore, cloned disk image)
    clone = load_or_create_node_id(path, _other_mac)
    assert clone != original
    # ...which then keeps its own new id
    assert load_or_create_node_id(path, _other_mac) == clone


def test_an_exo_folder_copied_to_another_folder_gets_a_new_id(tmp_path: Path):
    # e.g. a second exo on the same Mac started from a copy of the first one's EXO_HOME
    first_home = tmp_path / "exo-a"
    second_home = tmp_path / "exo-b"
    first_home.mkdir()
    first = load_or_create_node_id(first_home / "node_zid", _this_mac)
    shutil.copytree(first_home, second_home)

    second = load_or_create_node_id(second_home / "node_zid", _this_mac)
    assert second != first
    assert load_or_create_node_id(first_home / "node_zid", _this_mac) == first
    assert load_or_create_node_id(second_home / "node_zid", _this_mac) == second


def test_separate_exo_folders_have_separate_ids(tmp_path: Path):
    first = load_or_create_node_id(tmp_path / "a" / "node_zid", _this_mac)
    second = load_or_create_node_id(tmp_path / "b" / "node_zid", _this_mac)
    assert first != second


def test_without_a_machine_id_every_start_gets_a_new_id(tmp_path: Path):
    path = tmp_path / "node_zid"
    first = load_or_create_node_id(path, _no_machine_id)
    second = load_or_create_node_id(path, _no_machine_id)
    assert first != second
    assert not path.exists()


def test_an_unreadable_file_is_replaced(tmp_path: Path):
    path = tmp_path / "node_zid"
    for contents in ("", "not json", '{"node_id": "00ab", "bound_to": "x"}', "[1, 2]"):
        path.write_text(contents)
        node_id = load_or_create_node_id(path, _this_mac)
        assert is_valid_node_id(node_id)
        assert load_or_create_node_id(path, _this_mac) == node_id


def test_the_id_file_is_private(tmp_path: Path):
    path = tmp_path / "node_zid"
    load_or_create_node_id(path, _this_mac)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert THIS_MAC not in path.read_text()


def test_this_machine_has_an_id():
    if sys.platform not in ("darwin", "linux"):
        return
    assert machine_id() == machine_id()
    if sys.platform == "darwin":
        assert machine_id() is not None
