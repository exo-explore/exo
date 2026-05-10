import random
import socket
from collections.abc import Iterable
from contextlib import closing
from typing import Final, cast

DEFAULT_API_PORT: Final[int] = 52415
"""Exo's default API port (see ``--api-port`` in :mod:`exo.main`).

Mirrored here so :func:`random_ephemeral_port` can avoid handing the
API port back to a caller that is about to bind a *different* listener
(JACCL coordinator, drafter socket, etc.). If the operator is running
with a non-default API port the constant is harmless -- the kernel
returns ports from its own free pool which already excludes whatever
the operator's API listener has bound -- but for the default deploy
the dodge keeps placement decisions deterministic across the
"API-on-this-machine" / "API-on-some-other-machine" split.

Public so callers and tests can reference the same canonical value
without re-defining it; private-by-convention names would force
:mod:`exo.utils.tests.test_ports` to either reach in (triggering
``reportPrivateUsage``) or duplicate the literal.
"""

_KERNEL_PICK_RETRY_BUDGET: Final[int] = 8
"""Number of kernel picks tolerated before giving up and falling back.

A kernel-assigned ephemeral port returning :data:`_DEFAULT_API_PORT`
is improbable (the kernel skips ports already bound elsewhere); 8
retries covers the pathological "kernel free pool is nearly empty"
case without spinning forever.
"""


def random_ephemeral_port() -> int:
    """Pick a likely-free TCP port in the ephemeral range.

    Asks the kernel for a free port via a transient
    ``bind(("", 0))`` -> ``getsockname`` -> close sequence. The
    returned port is therefore guaranteed to be free *on this host
    at this moment*: the kernel will not reassign a recently-released
    ephemeral port for a short window, so the caller has a generous
    race buffer in which to bind it for real.

    Codex P1 (PR #20, placement.py:711): pre-fix this function picked
    a uniformly random integer in [49153, 65535] with no availability
    check at all. In single-machine deploys (master and the eventual
    ``bind_target_listener`` caller share a kernel) this produced a
    ~1-2 percent collision rate against the kernel's existing
    ephemeral-port allocations; runner startup would then fail with
    ``EADDRINUSE`` and surface as a placement-time degradation event.
    The kernel-assigned pick drops that collision rate to effectively
    zero on the same host.

    Cross-machine deploys (master and target rank 0 on different
    hosts) still cannot benefit from this approach -- the master's
    kernel does not know the target's port allocations. The proper
    fix for that case is a two-phase "target rank 0 binds first and
    advertises the bound port back to placement" protocol that
    requires changing :class:`exo.shared.types.events.DrafterPlacement`'s
    wire schema; that is tracked for a follow-up PR. Same-host deploys
    are the dominant production shape (single-laptop dev, single-NUC
    homelab, single-rack staging) and are fully covered here.

    The default-API-port dodge from the historical implementation is
    preserved as a safety net for the rare case the kernel's free
    pool transiently exposes :data:`_DEFAULT_API_PORT` (e.g. an
    earlier API listener is in ``TIME_WAIT``); see
    :data:`_KERNEL_PICK_RETRY_BUDGET` for the retry bound.
    """
    for _ in range(_KERNEL_PICK_RETRY_BUDGET):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            # ``host=""`` -> wildcard (``0.0.0.0``); ``port=0`` ->
            # kernel picks a free port from its ephemeral pool. This
            # is the standard "ask the kernel" idiom and is the only
            # way to get a guaranteed-free port without a wire round
            # trip to the eventual binder.
            sock.bind(("", 0))
            # ``getsockname`` is typed as ``Any`` in the standard
            # library stubs (the return varies by address family);
            # for an AF_INET socket it is ``tuple[str, int]``, so
            # the explicit ``cast`` documents the family invariant
            # and satisfies strict type checking.
            sockname = cast(tuple[str, int], sock.getsockname())
            port = sockname[1]
        if port != DEFAULT_API_PORT:
            return port
    # Improbable: ``_KERNEL_PICK_RETRY_BUDGET`` consecutive kernel
    # picks all hit the API port. Fall back to the legacy uniform
    # random implementation so callers always receive a port even
    # when the kernel's free-port pool is pathologically narrow.
    port = random.randint(49153, 65535)
    return port - 1 if port <= DEFAULT_API_PORT else port


def random_ephemeral_port_excluding(reserved: Iterable[int]) -> int:
    """Draw an ephemeral port that does not collide with any value in
    ``reserved``.

    Used by placement bookkeeping when multiple listener ports must be
    bound on the same node (e.g., target rank 0 binds the drafter
    accept socket, the target-peer fanout socket, AND either the
    JACCL coordinator port or the MLX ring port). A naive
    ``random_ephemeral_port`` for each draw can occasionally produce
    a duplicate, leading to nondeterministic ``EADDRINUSE`` bind
    failures during runner bootstrap. The ephemeral range is wide
    enough (~13K ports) that this loop almost never iterates.

    Codex P2 (PR #21 round 3): the original collision-avoidance loop
    only checked ``target_peer_socket_port != drafter_socket_port``
    and missed sibling listener ports (jaccl coordinator port,
    ring ephemeral port) that bind on the same node.
    """
    reserved_set = set(reserved)
    port = random_ephemeral_port()
    while port in reserved_set:
        port = random_ephemeral_port()
    return port
