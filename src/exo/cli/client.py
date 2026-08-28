"""HTTP client for talking to a running exo cluster."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


class ExoClientError(Exception):
    """Raised when the exo API returns an error."""

    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"HTTP {status}: {detail}")


class ExoClient:
    """Synchronous HTTP client for the exo cluster management API."""

    def __init__(self, host: str = "localhost", port: int = 52415) -> None:
        self.base_url = f"http://{host}:{port}"

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body else None
        headers = {"Content-Type": "application/json"} if data else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            try:
                error_body = json.loads(exc.read())
                detail = error_body.get("error", {}).get("message", str(error_body))
            except Exception:
                detail = exc.reason
            raise ExoClientError(exc.code, detail) from exc
        except urllib.error.URLError as exc:
            print(
                f"Error: cannot connect to exo at {self.base_url}\n"
                f"Is the cluster running? ({exc.reason})",
                file=sys.stderr,
            )
            sys.exit(1)

    def get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, body)

    def delete(self, path: str) -> dict[str, Any]:
        return self._request("DELETE", path)

    # ----- High-level methods -----

    def health(self) -> dict[str, Any]:
        return self.get("/v1/cluster/health")

    def overview(self) -> dict[str, Any]:
        return self.get("/v1/cluster")

    def nodes(self) -> dict[str, Any]:
        return self.get("/v1/cluster/nodes")

    def node(self, node_id: str) -> dict[str, Any]:
        return self.get(f"/v1/cluster/nodes/{node_id}")

    def models(self) -> dict[str, Any]:
        return self.get("/v1/cluster/models")

    def model_status(self, model_id: str) -> dict[str, Any]:
        return self.get(f"/v1/cluster/models/{model_id}/status")

    def load_model(
        self,
        model_id: str,
        *,
        min_nodes: int = 1,
        sharding: str = "auto",
    ) -> dict[str, Any]:
        return self.post(
            "/v1/cluster/models/load",
            {
                "model_id": model_id,
                "min_nodes": min_nodes,
                "preferred_sharding": sharding,
            },
        )

    def unload_model(self, model_id: str) -> dict[str, Any]:
        return self.delete(f"/v1/cluster/models/{model_id}")

    def swap_model(
        self,
        unload: str,
        load: str,
        *,
        min_nodes: int = 1,
        sharding: str = "auto",
    ) -> dict[str, Any]:
        return self.post(
            "/v1/cluster/models/swap",
            {
                "unload_model_id": unload,
                "load_model_id": load,
                "min_nodes": min_nodes,
                "preferred_sharding": sharding,
            },
        )

    def wait_for_model(
        self,
        model_id: str,
        *,
        poll_interval: float = 5.0,
        timeout: float = 600.0,
        on_progress: Any | None = None,
    ) -> dict[str, Any]:
        """Block until a model is ready or timeout is reached.

        Args:
            model_id: Model name to poll.
            poll_interval: Seconds between polls.
            timeout: Max seconds to wait before raising.
            on_progress: Optional callback(status_dict) called each poll.

        Returns:
            The final status dict when ready=True.
        """
        start = time.monotonic()
        while True:
            status = self.model_status(model_id)
            if on_progress:
                on_progress(status)
            if status.get("ready"):
                return status
            if status.get("status") == "failed":
                raise ExoClientError(500, f"Model {model_id} failed to load")
            elapsed = time.monotonic() - start
            if elapsed > timeout:
                raise ExoClientError(
                    408,
                    f"Timed out after {timeout:.0f}s waiting for {model_id}",
                )
            time.sleep(poll_interval)
