import json
from datetime import datetime, timezone
from pathlib import Path

from anyio import BrokenResourceError, ClosedResourceError
from loguru import logger

from exo.shared.constants import EXO_LOG_DIR
from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import RunnerFailed, RunnerId
from exo.utils.channels import Sender


def _pending_runner_failure_dir() -> Path:
    return EXO_LOG_DIR / "pending_runner_failures"


def persist_runner_failure(
    bound_instance: BoundInstance,
    *,
    error_message: str,
    source: str,
) -> Path:
    failure_dir = _pending_runner_failure_dir()
    failure_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    runner_id = str(bound_instance.bound_runner_id)
    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "runner_id": runner_id,
        "instance_id": str(bound_instance.instance.instance_id),
        "node_id": str(bound_instance.bound_node_id),
        "model_id": str(bound_instance.instance.shard_assignments.model_id),
        "error_message": error_message,
        "source": source,
    }

    final_path = failure_dir / f"runner_failure_{timestamp}_{runner_id[:8]}.json"
    temp_path = final_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, indent=2))
    temp_path.replace(final_path)
    logger.warning("Persisted terminal runner failure to {}", final_path)
    return final_path


async def replay_persisted_runner_failures(event_sender: Sender[Event]) -> int:
    failure_dir = _pending_runner_failure_dir()
    if not failure_dir.exists():
        return 0

    replayed = 0
    for path in sorted(failure_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
            runner_id = RunnerId(payload["runner_id"])
            error_message = str(payload["error_message"])
            await event_sender.send(
                RunnerStatusUpdated(
                    runner_id=runner_id,
                    runner_status=RunnerFailed(error_message=error_message),
                )
            )
            path.unlink()
            replayed += 1
        except (ClosedResourceError, BrokenResourceError):
            break
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            invalid_path = path.with_suffix(".invalid")
            path.replace(invalid_path)
            logger.warning(
                "Invalid persisted runner failure record {}: {}",
                invalid_path,
                exc,
            )

    return replayed
