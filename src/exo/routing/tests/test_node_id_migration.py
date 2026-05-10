"""Regression tests for the cache→config migration of the node-ID
keypair (Codex P1, PR #16 round 5).

The keypair used to live under ``EXO_CACHE_HOME``, which is subject
to normal cache cleanup (e.g. ``trash ~/.cache/exo``) and would
silently regenerate a new node-ID. The fix relocates the keypair to
``EXO_CONFIG_HOME`` and migrates legacy files transparently.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from exo_pyo3_bindings import Keypair

from exo.routing.router import (
    _migrate_legacy_node_id_keypair,  # pyright: ignore[reportPrivateUsage]
    get_node_id_keypair,
)


def test_legacy_keypair_is_migrated_to_new_location(tmp_path: Path) -> None:
    """Legacy cache-dir keypair must be moved to the new config-dir
    location and the legacy file removed -- so the node retains its
    identity across the upgrade and a future cache wipe doesn't
    resurrect a stale copy."""
    legacy_path = tmp_path / "cache" / "node_id.keypair"
    new_path = tmp_path / "config" / "node_id.keypair"
    legacy_path.parent.mkdir(parents=True)

    keypair = Keypair.generate()
    legacy_bytes = keypair.to_bytes()
    legacy_path.write_bytes(legacy_bytes)

    _migrate_legacy_node_id_keypair(new_path, legacy_path)

    assert new_path.exists(), "migration must place keypair at new location"
    assert new_path.read_bytes() == legacy_bytes, (
        "migration must preserve the byte-for-byte keypair contents "
        "so the node retains its peer ID"
    )
    assert not legacy_path.exists(), (
        "migration must remove the legacy file once the new location "
        "holds the keypair, otherwise a later cache wipe could "
        "resurrect a now-stale copy"
    )


def test_migration_is_idempotent_when_new_location_already_present(
    tmp_path: Path,
) -> None:
    """If the new location already has a keypair, migration must be
    a no-op even when a legacy file exists -- otherwise we'd
    overwrite the (canonical) new keypair with a stale legacy one."""
    legacy_path = tmp_path / "cache" / "node_id.keypair"
    new_path = tmp_path / "config" / "node_id.keypair"
    legacy_path.parent.mkdir(parents=True)
    new_path.parent.mkdir(parents=True)

    canonical = Keypair.generate().to_bytes()
    legacy = Keypair.generate().to_bytes()
    new_path.write_bytes(canonical)
    legacy_path.write_bytes(legacy)

    _migrate_legacy_node_id_keypair(new_path, legacy_path)

    assert new_path.read_bytes() == canonical, (
        "migration must NOT overwrite an existing new-location keypair"
    )
    # We deliberately leave the legacy file alone in this branch:
    # touching it would surprise an operator who is intentionally
    # keeping both copies during an upgrade window.
    assert legacy_path.exists()


def test_migration_skipped_when_no_legacy_file(tmp_path: Path) -> None:
    """Fresh installs must not error when the legacy path is absent."""
    new_path = tmp_path / "config" / "node_id.keypair"
    new_path.parent.mkdir(parents=True)

    _migrate_legacy_node_id_keypair(new_path, tmp_path / "missing.keypair")

    assert not new_path.exists()


def test_get_node_id_keypair_uses_migrated_legacy_keypair(tmp_path: Path) -> None:
    """End-to-end: ``get_node_id_keypair`` must surface the legacy
    keypair bytes when only the legacy path holds a valid file at
    call time, completing the cache→config migration on first use."""
    legacy_path = tmp_path / "cache" / "node_id.keypair"
    new_path = tmp_path / "config" / "node_id.keypair"
    legacy_path.parent.mkdir(parents=True)

    keypair = Keypair.generate()
    expected_bytes = keypair.to_bytes()
    legacy_path.write_bytes(expected_bytes)

    loaded = get_node_id_keypair(path=new_path, legacy_path=legacy_path)

    assert loaded.to_bytes() == expected_bytes
    assert new_path.exists()
    assert not legacy_path.exists()


# ---------------------------------------------------------------------------
# Codex P1 (PR #16 round-(N+2), router.py:297): per-process scoping
# ---------------------------------------------------------------------------
#
# The new same-host multi-node workflow (per-process
# ``--peer-download-port``) requires distinct ``NodeId``s per
# process so peer-discovery's self-skip and routing's unique-NodeId
# invariants hold. ``get_node_id_keypair`` therefore accepts a
# ``process_scope`` argument that is folded into the on-disk
# filename.


def test_distinct_process_scopes_produce_distinct_keypairs(tmp_path: Path) -> None:
    """Two processes that pass different scopes (e.g. distinct
    peer-download ports) MUST end up with different keypair files
    and different on-disk identities; otherwise two same-host
    nodes would race on the same NodeId."""
    base_path = tmp_path / "config" / "node_id.keypair"

    keypair_a = get_node_id_keypair(
        path=base_path, legacy_path=None, process_scope=52416
    )
    keypair_b = get_node_id_keypair(
        path=base_path, legacy_path=None, process_scope=52417
    )

    assert keypair_a.to_bytes() != keypair_b.to_bytes(), (
        "distinct process scopes must yield distinct keypairs so "
        "same-host multi-node deployments don't share a NodeId"
    )

    scoped_a = base_path.parent / "node_id.52416.keypair"
    scoped_b = base_path.parent / "node_id.52417.keypair"
    assert scoped_a.exists()
    assert scoped_b.exists()
    assert scoped_a.read_bytes() != scoped_b.read_bytes()


def test_same_process_scope_is_stable_across_calls(tmp_path: Path) -> None:
    """Per-process scoping must remain *persistent*: the same
    process (same scope) must load the same keypair on subsequent
    calls -- otherwise restart would silently churn NodeIds."""
    base_path = tmp_path / "config" / "node_id.keypair"

    first = get_node_id_keypair(path=base_path, legacy_path=None, process_scope=52416)
    second = get_node_id_keypair(path=base_path, legacy_path=None, process_scope=52416)

    assert first.to_bytes() == second.to_bytes()


def test_migration_runs_inside_file_lock(tmp_path: Path) -> None:
    """Codex P2 (PR #16 round-(N+2), router.py:322): the legacy
    migration must execute *inside* ``FileLock`` so two processes
    booting concurrently can't both pass the existence check, race
    each other into divergent in-memory keypairs, and end up with
    mismatched identities for the same on-disk file.

    We assert this structurally by hooking ``_migrate_legacy_node_id_keypair``
    and ``filelock.FileLock`` and verifying the lock is acquired
    *before* the migrator is called. A pre-lock migration would
    show ``migrate_called=True`` while the lock is still
    ``unacquired``."""
    import exo.routing.router as router_mod

    legacy_path = tmp_path / "cache" / "node_id.keypair"
    base_path = tmp_path / "config" / "node_id.keypair"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_bytes(Keypair.generate().to_bytes())

    lock_state: dict[str, bool] = {"acquired": False, "acquired_before_migrate": False}

    # We hook ``router_mod.FileLock`` (the symbol the production
    # code dereferences) with a thin wrapper class. The wrapper
    # delegates to the real ``FileLock`` instance but flips the
    # ``acquired`` flag on entry, which the migrator hook below
    # then snapshots. This keeps the type of ``FileLock`` intact
    # while letting us observe acquire-vs-migrate ordering.
    real_filelock = router_mod.FileLock

    class _ObservingFileLock:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._inner = real_filelock(*args, **kwargs)  # pyright: ignore[reportArgumentType]

        def __enter__(self) -> object:
            lock_state["acquired"] = True
            return self._inner.__enter__()

        def __exit__(self, *exc: object) -> object:
            return self._inner.__exit__(*exc)  # pyright: ignore[reportArgumentType]

    original_migrate = router_mod._migrate_legacy_node_id_keypair  # pyright: ignore[reportPrivateUsage]

    def _track_migrate(new_path: Path, legacy: Path) -> None:
        lock_state["acquired_before_migrate"] = lock_state["acquired"]
        original_migrate(new_path, legacy)

    router_mod.FileLock = _ObservingFileLock
    router_mod._migrate_legacy_node_id_keypair = _track_migrate  # pyright: ignore[reportPrivateUsage]
    try:
        _ = get_node_id_keypair(path=base_path, legacy_path=legacy_path)
    finally:
        router_mod.FileLock = real_filelock
        router_mod._migrate_legacy_node_id_keypair = original_migrate  # pyright: ignore[reportPrivateUsage]

    assert lock_state["acquired_before_migrate"] is True, (
        "legacy migration must run INSIDE the FileLock to prevent a "
        "concurrent-startup race on the on-disk keypair"
    )


class TestNodeIdKeypairScope:
    """Codex P1 (PR #16 round-(N+3), main.py:74): the node-ID keypair
    scope MUST account for every distinguishable per-process port,
    not just ``--peer-download-port``. With peer-download disabled
    the operator can legitimately keep the default
    ``peer_download_port`` (no socket bind), so the previous
    peer-only scope let two same-host processes share an identity.
    """

    def _build_args(
        self,
        *,
        libp2p_port: int = 0,
        api_port: int = 52415,
        peer_download_port: int = 52416,
        no_downloads: bool = False,
        no_peer_download: bool = False,
        spawn_api: bool = False,
    ):  # noqa: ANN202
        from exo.main import Args

        return Args(
            libp2p_port=libp2p_port,
            api_port=api_port,
            peer_download_port=peer_download_port,
            no_downloads=no_downloads,
            no_peer_download=no_peer_download,
            spawn_api=spawn_api,
        )

    def test_distinct_libp2p_ports_yield_distinct_scopes(self) -> None:
        from exo.main import (
            _node_id_keypair_scope,  # pyright: ignore[reportPrivateUsage]
        )

        scope_a = _node_id_keypair_scope(self._build_args(libp2p_port=4001))
        scope_b = _node_id_keypair_scope(self._build_args(libp2p_port=4002))
        assert scope_a != scope_b

    def test_distinct_api_ports_yield_distinct_scopes(self) -> None:
        from exo.main import (
            _node_id_keypair_scope,  # pyright: ignore[reportPrivateUsage]
        )

        scope_a = _node_id_keypair_scope(self._build_args(api_port=52415))
        scope_b = _node_id_keypair_scope(self._build_args(api_port=52416))
        assert scope_a != scope_b

    def test_distinct_peer_download_ports_yield_distinct_scopes(self) -> None:
        from exo.main import (
            _node_id_keypair_scope,  # pyright: ignore[reportPrivateUsage]
        )

        scope_a = _node_id_keypair_scope(self._build_args(peer_download_port=52416))
        scope_b = _node_id_keypair_scope(self._build_args(peer_download_port=52417))
        assert scope_a != scope_b

    def test_disabled_peer_download_with_same_default_port_still_isolates(
        self,
    ) -> None:
        """The original Codex P1 (round-(N+3)) regression: with
        ``--no-peer-download`` two processes can both keep
        ``peer_download_port=52416``. They MUST still get distinct
        scopes when *some* other port differs (here, libp2p).
        Pre-fix the scope was just ``peer_download_port`` and these
        two configs collided on the same keypair."""
        from exo.main import (
            _node_id_keypair_scope,  # pyright: ignore[reportPrivateUsage]
        )

        process_one = self._build_args(
            libp2p_port=4001,
            no_peer_download=True,
            peer_download_port=52416,
        )
        process_two = self._build_args(
            libp2p_port=4002,
            no_peer_download=True,
            peer_download_port=52416,
        )
        assert _node_id_keypair_scope(process_one) != _node_id_keypair_scope(
            process_two
        )

    def test_identical_args_yield_identical_scope(self) -> None:
        """Stability invariant: the same configuration on a single
        process across restarts must hash to the same scope so the
        node retains its identity across restarts."""
        from exo.main import (
            _node_id_keypair_scope,  # pyright: ignore[reportPrivateUsage]
        )

        args = self._build_args(
            libp2p_port=4001, api_port=52415, peer_download_port=52416
        )
        assert _node_id_keypair_scope(args) == _node_id_keypair_scope(args)

    def test_libp2p_port_zero_uses_pid_for_per_process_isolation(self) -> None:
        """Codex P1 (PR #16 round-(N+8), main.py:457): with
        ``--libp2p-port 0`` the configured port is the literal ``0``
        even though each process binds a different ephemeral port at
        runtime. Without per-process discrimination two same-host
        worker-only processes (no API, no peer download) sharing the
        default ``peer_download_port`` and ``api_port`` would collide
        on the same scoped keypair. The scope must therefore fold in
        ``os.getpid()`` (or another guaranteed per-process
        discriminator) when ``libp2p_port == 0``."""
        import os

        from exo.main import (
            _node_id_keypair_scope,  # pyright: ignore[reportPrivateUsage]
        )

        scope = _node_id_keypair_scope(
            self._build_args(
                libp2p_port=0,
                api_port=52415,
                peer_download_port=52416,
                no_peer_download=True,
                spawn_api=False,
            )
        )

        assert f"pid-{os.getpid()}" in scope, (
            f"libp2p_port=0 must mix in os.getpid() to discriminate "
            f"same-host processes binding ephemeral libp2p ports; "
            f"got scope={scope!r}"
        )

    def test_libp2p_port_zero_in_two_processes_yield_distinct_scopes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: simulate two same-host processes both binding
        ``libp2p_port=0`` and otherwise default ports. Pre-fix they
        collided on a single keypair file; post-fix the scopes
        differ because each carries its own PID."""
        import os

        from exo.main import (
            _node_id_keypair_scope,  # pyright: ignore[reportPrivateUsage]
        )

        # Process A: real PID
        scope_a = _node_id_keypair_scope(
            self._build_args(
                libp2p_port=0,
                api_port=52415,
                peer_download_port=52416,
                no_peer_download=True,
                spawn_api=False,
            )
        )

        # Process B: simulate a different PID via monkeypatch
        real_pid = os.getpid()
        monkeypatch.setattr(os, "getpid", lambda: real_pid + 1)
        scope_b = _node_id_keypair_scope(
            self._build_args(
                libp2p_port=0,
                api_port=52415,
                peer_download_port=52416,
                no_peer_download=True,
                spawn_api=False,
            )
        )

        assert scope_a != scope_b, (
            "two same-host processes both binding libp2p_port=0 with "
            "identical api/peer ports must produce distinct keypair "
            "scopes; otherwise they load the same on-disk keypair "
            "and collide on NodeId, breaking routing/election "
            f"invariants. scope_a={scope_a!r} scope_b={scope_b!r}"
        )


def test_legacy_migration_serialized_across_process_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex P1 (PR #16 round-(N+13), router.py:359): legacy
    adoption MUST be serialized across all ``process_scope`` values,
    even when the per-scope ``resolved_path`` lock differs and the
    cross-device byte-copy fallback path is taken inside
    ``_migrate_legacy_node_id_keypair``.

    Pre-fix this test produces two identical scoped keypairs (both
    matching the legacy bytes), simulating two same-host processes
    racing legacy adoption: each acquires its own per-scope lock,
    both fall through to the byte-copy branch, both read the same
    legacy bytes, and both end up writing those bytes to their own
    scoped file -- duplicate ``NodeId`` despite distinct scopes.

    Post-fix the migrator is wrapped in a second FileLock keyed on
    the legacy path. The first scope wins adoption and unlinks the
    legacy file; the second scope's migrator no-ops on the absent
    legacy and generates a fresh keypair, so the two scopes diverge
    as required by the per-process isolation invariant.

    We simulate the cross-device fallback by monkey-patching
    ``Path.replace`` to raise ``OSError`` (the same trigger that
    fires on Linux when ``XDG_*`` dirs span filesystems). The
    serialization invariant is asserted by also blocking the byte
    copy with a ``threading.Event`` so two threads must contend on
    the legacy lock; only one thread should observe the legacy
    file present at copy time.
    """
    import threading

    import exo.routing.router as router_mod

    legacy_path = tmp_path / "cache" / "node_id.keypair"
    base_path = tmp_path / "config" / "node_id.keypair"
    legacy_path.parent.mkdir(parents=True)
    base_path.parent.mkdir(parents=True)

    legacy_bytes = Keypair.generate().to_bytes()
    legacy_path.write_bytes(legacy_bytes)

    # Force the cross-device fallback so the migrator goes through
    # the read_bytes/write_bytes/unlink sequence (the path Codex
    # flagged as racy).
    real_replace = Path.replace

    def _force_cross_device(self: Path, target: object) -> object:  # noqa: ANN001
        if Path(self) == legacy_path:
            raise OSError("simulated cross-device link error")
        return real_replace(self, target)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(Path, "replace", _force_cross_device)

    # Pause inside the byte-copy branch so two threads pile up on
    # the legacy lock while one thread holds it. Without the legacy
    # lock both threads would observe the legacy file present at
    # this point and both would proceed to write_bytes/unlink.
    in_copy = threading.Event()
    release_copy = threading.Event()
    real_write_bytes = Path.write_bytes

    def _slow_write_bytes(self: Path, data: bytes) -> int:
        if self.parent == base_path.parent:
            in_copy.set()
            release_copy.wait(timeout=5.0)
        return real_write_bytes(self, data)

    monkeypatch.setattr(Path, "write_bytes", _slow_write_bytes)

    keypairs: dict[int, Keypair] = {}

    def _run(scope: int) -> None:
        keypairs[scope] = router_mod.get_node_id_keypair(
            path=base_path, legacy_path=legacy_path, process_scope=scope
        )

    thread_a = threading.Thread(target=_run, args=(52416,), daemon=True)
    thread_b = threading.Thread(target=_run, args=(52417,), daemon=True)
    thread_a.start()
    in_copy.wait(timeout=5.0)
    # While thread_a is paused inside the byte copy holding the
    # legacy lock, thread_b should be blocked on the legacy lock --
    # NOT racing through its own byte copy of the same legacy file.
    thread_b.start()
    # Give thread_b a moment to attempt acquiring the legacy lock
    # so we can assert it did not slip through.
    thread_b.join(timeout=0.2)
    assert thread_b.is_alive(), (
        "second scope must be blocked on the legacy lock while the "
        "first scope is mid-copy; if this fails, both scopes will "
        "duplicate the legacy NodeId via the byte-copy race"
    )
    release_copy.set()
    thread_a.join(timeout=5.0)
    thread_b.join(timeout=5.0)
    assert not thread_a.is_alive() and not thread_b.is_alive()

    scope_a_bytes = keypairs[52416].to_bytes()
    scope_b_bytes = keypairs[52417].to_bytes()
    assert scope_a_bytes != scope_b_bytes, (
        "concurrent legacy adoption across distinct process_scope "
        "values must NOT produce duplicate keypairs; the legacy "
        "lock should let exactly one scope adopt the legacy bytes "
        "while the other generates a fresh identity"
    )
    # Exactly one scoped file should match the legacy bytes (the
    # winner of adoption); the other was generated fresh.
    scoped_a = base_path.parent / "node_id.52416.keypair"
    scoped_b = base_path.parent / "node_id.52417.keypair"
    matches = sum(
        1 for p in (scoped_a, scoped_b) if p.exists() and p.read_bytes() == legacy_bytes
    )
    assert matches == 1, (
        f"exactly one scope must have adopted the legacy bytes; "
        f"matches={matches} indicates the cross-device race fired"
    )
    assert not legacy_path.exists(), "legacy file must be unlinked after adoption"


def test_legacy_migration_adopts_into_scoped_path(tmp_path: Path) -> None:
    """When a process passes a scope and a legacy unscoped keypair
    exists, the legacy bytes must be adopted into the scoped path.
    This is the upgrade-time behaviour: the first process to boot
    after the upgrade keeps the operator's existing identity; later
    processes (different scopes) start with fresh identities, which
    is exactly what per-process isolation requires."""
    legacy_path = tmp_path / "cache" / "node_id.keypair"
    base_path = tmp_path / "config" / "node_id.keypair"
    legacy_path.parent.mkdir(parents=True)

    expected_bytes = Keypair.generate().to_bytes()
    legacy_path.write_bytes(expected_bytes)

    loaded = get_node_id_keypair(
        path=base_path, legacy_path=legacy_path, process_scope=52416
    )

    scoped = base_path.parent / "node_id.52416.keypair"
    assert loaded.to_bytes() == expected_bytes
    assert scoped.exists(), "legacy bytes must land at the scoped path"
    assert scoped.read_bytes() == expected_bytes
    assert not legacy_path.exists()
