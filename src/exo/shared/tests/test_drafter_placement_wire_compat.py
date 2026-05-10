"""Wire-schema compatibility tests for :class:`DrafterPlacement`.

Codex P1 (PR #21 round-(N+9), instances.py:97):
``DrafterPlacement.target_peer_socket_port`` must round-trip through
pubsub ``model_validate_json`` even for legacy/historical payloads
that pre-date the field. Pubsub-based events (commands, state
broadcasts) deserialise via Pydantic ``model_validate_json``, so any
required field on a previously serialisable model breaks instance and
state replay during a rolling upgrade or when an older event stream
is replayed against newer code.
"""

from __future__ import annotations

import json

from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.worker.instances import DrafterPlacement
from exo.shared.types.worker.runners import RunnerId


class TestDrafterPlacementBackwardCompat:
    """Ensure ``DrafterPlacement`` accepts pre-fanout legacy payloads.

    Pre-fix ``target_peer_socket_port`` was required, so a JSON payload
    produced by an older node (or replayed from a stored event stream)
    that omits the field would fail Pydantic validation and abort
    state replay. The field must be optional with a safe default to
    keep mixed-version clusters and historical replay working.
    """

    def test_legacy_payload_without_target_peer_port_validates(self) -> None:
        legacy_payload = {
            "drafter_node_id": "node-drafter",
            "drafter_runner_id": "runner-drafter",
            "drafter_model_id": "mlx-community/test-drafter",
            "drafter_rank": 1,
            "drafter_socket_host": "169.254.0.10",
            "drafter_socket_port": 60001,
        }
        placement = DrafterPlacement.model_validate(legacy_payload)
        assert placement.target_peer_socket_port is None
        assert placement.target_peer_hosts_by_rank == {}

    def test_legacy_json_string_validates(self) -> None:
        """End-to-end JSON path: pubsub uses ``model_validate_json``."""
        legacy_json = json.dumps(
            {
                "drafter_node_id": "node-drafter",
                "drafter_runner_id": "runner-drafter",
                "drafter_model_id": "mlx-community/test-drafter",
                "drafter_rank": 1,
                "drafter_socket_host": "169.254.0.10",
                "drafter_socket_port": 60001,
            }
        )
        placement = DrafterPlacement.model_validate_json(legacy_json)
        assert placement.target_peer_socket_port is None

    def test_modern_payload_round_trips(self) -> None:
        modern = DrafterPlacement(
            drafter_node_id=NodeId("node-drafter"),
            drafter_runner_id=RunnerId("runner-drafter"),
            drafter_model_id=ModelId("mlx-community/test-drafter"),
            drafter_rank=2,
            drafter_socket_host="169.254.0.10",
            drafter_socket_port=60001,
            target_peer_socket_port=60002,
            target_peer_hosts_by_rank={"1": "169.254.0.20"},
        )
        round_tripped = DrafterPlacement.model_validate_json(modern.model_dump_json())
        assert round_tripped == modern
        assert round_tripped.target_peer_socket_port == 60002

    def test_explicit_none_target_peer_port_accepted(self) -> None:
        """A new placement that explicitly omits the fanout port (e.g. a
        single-rank target asymmetric instance) must validate and stay
        ``None`` so downstream code can detect the legacy/no-fanout
        case uniformly.
        """
        placement = DrafterPlacement(
            drafter_node_id=NodeId("node-drafter"),
            drafter_runner_id=RunnerId("runner-drafter"),
            drafter_model_id=ModelId("mlx-community/test-drafter"),
            drafter_rank=1,
            drafter_socket_host="169.254.0.10",
            drafter_socket_port=60001,
        )
        assert placement.target_peer_socket_port is None
        round_tripped = DrafterPlacement.model_validate_json(
            placement.model_dump_json()
        )
        assert round_tripped.target_peer_socket_port is None

    def test_field_range_constraints_still_enforced(self) -> None:
        """Optional must not relax the port range. Out-of-range still
        errors so a malformed payload (port <= 0 or > 65535) is
        rejected at the boundary instead of producing a bad bind
        attempt at runtime.
        """
        from pydantic import ValidationError

        bad_ports: list[int] = [0, 65536, -1]
        base_payload: dict[str, object] = {
            "drafter_node_id": "node-drafter",
            "drafter_runner_id": "runner-drafter",
            "drafter_model_id": "mlx-community/test-drafter",
            "drafter_rank": 1,
            "drafter_socket_host": "169.254.0.10",
            "drafter_socket_port": 60001,
        }
        for bad_port in bad_ports:
            payload: dict[str, object] = {
                **base_payload,
                "target_peer_socket_port": bad_port,
            }
            try:
                DrafterPlacement.model_validate(payload)
            except ValidationError:
                continue
            raise AssertionError(
                f"out-of-range target_peer_socket_port={bad_port!r} "
                f"unexpectedly validated"
            )
