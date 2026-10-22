import time
from collections.abc import Sequence

from exo.shared.types.volunteer_grid import (
    VolunteerNodeProfile,
    VolunteerNodeStatus,
)


class VolunteerNodeHealthMonitor:
    """Monitors volunteer edge node health, latency, reliability scores, and timeouts."""

    def __init__(self, default_heartbeat_timeout_seconds: float = 30.0) -> None:
        self._nodes: dict[str, VolunteerNodeProfile] = {}
        self._heartbeat_timeout = default_heartbeat_timeout_seconds

    def register_node(self, profile: VolunteerNodeProfile) -> None:
        """Registers or updates a volunteer node profile."""
        self._nodes[profile.node_id] = profile

    def get_node(self, node_id: str) -> VolunteerNodeProfile | None:
        return self._nodes.get(node_id)

    def record_heartbeat(self, node_id: str) -> None:
        """Records an active heartbeat signal from the node."""
        node = self._nodes.get(node_id)
        if node is None:
            return
        new_status: VolunteerNodeStatus = "healthy" if node.consecutive_timeouts < 2 else "straggler"
        self._nodes[node_id] = node.model_copy(
            update={
                "last_heartbeat_timestamp": time.time(),
                "status": new_status,
            }
        )

    def record_job_success(self, node_id: str, elapsed_seconds: float, tokens_processed: int) -> None:
        """Updates node reliability score and throughput upon successful task completion."""
        node = self._nodes.get(node_id)
        if node is None:
            return

        tps_sample = (tokens_processed / elapsed_seconds) if elapsed_seconds > 0 else node.estimated_tps
        # EWMA update of throughput: 70% historical + 30% current
        updated_tps = (0.7 * node.estimated_tps) + (0.3 * tps_sample)
        # Reward reliability up to max 1.0
        updated_score = min(1.0, node.reliability_score + 0.1)

        self._nodes[node_id] = node.model_copy(
            update={
                "estimated_tps": updated_tps,
                "reliability_score": updated_score,
                "status": "healthy",
                "consecutive_timeouts": 0,
                "total_jobs_completed": node.total_jobs_completed + 1,
                "last_heartbeat_timestamp": time.time(),
            }
        )

    def record_job_timeout(self, node_id: str) -> None:
        """Penalizes node upon timeout or straggling failure."""
        node = self._nodes.get(node_id)
        if node is None:
            return

        timeouts = node.consecutive_timeouts + 1
        updated_score = max(0.0, node.reliability_score * 0.65)

        if timeouts >= 4:
            new_status: VolunteerNodeStatus = "offline"
        elif timeouts >= 2:
            new_status = "straggler"
        else:
            new_status = "healthy"

        self._nodes[node_id] = node.model_copy(
            update={
                "consecutive_timeouts": timeouts,
                "reliability_score": updated_score,
                "status": new_status,
            }
        )

    def evaluate_grid_health(self) -> None:
        """Evaluates all registered nodes and transitions dead nodes to offline status."""
        now = time.time()
        for node_id, node in list(self._nodes.items()):
            if (now - node.last_heartbeat_timestamp) > self._heartbeat_timeout:
                self._nodes[node_id] = node.model_copy(
                    update={"status": "offline", "reliability_score": 0.0}
                )

    def get_ranked_available_nodes(self) -> Sequence[VolunteerNodeProfile]:
        """Returns online nodes ranked by reliability score and estimated throughput."""
        self.evaluate_grid_health()
        available = [n for n in self._nodes.values() if n.is_available_for_dispatch]
        # Rank primarily by reliability score, secondarily by estimated TPS
        return sorted(available, key=lambda n: (n.reliability_score, n.estimated_tps), reverse=True)
