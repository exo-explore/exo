"""Marker-driven test framework for exo integration tests.

Test authors declare requirements via markers:

    @pytest.mark.cluster(count=2, thunderbolt='a2a')
    @pytest.mark.instance('mlx-community/Llama-3.2-1B-Instruct-4bit',
                          sharding='tensor', comm='jaccl')
    def test_jaccl_inference(session):
        resp = session.chat('What is 2+2?')
        assert '4' in resp

The `session` fixture reads the markers, deploys the cluster, places the
instance, and provides a `Session` object. All cluster/instance orchestration
lives in `exo_tools.harness`; this module is purely the pytest-facing layer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from exo_tools.client import ExoClient
from exo_tools.cluster import (
    Chip,
    ClusterInfo,
    EcoSession,
    Thunderbolt,
    make_client_from_url,
)
from exo_tools.harness import Comm, Sharding

from exo.api.types.api import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
)

DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def _extract_content(resp: ChatCompletionResponse) -> str:
    """Extract plain-text content from a non-streaming chat completion."""
    choice = resp.choices[0]
    if not isinstance(choice, ChatCompletionChoice):
        raise RuntimeError(
            f"Expected non-streaming choice, got {type(choice).__name__}"
        )
    content = choice.message.content
    if not isinstance(content, str):
        raise RuntimeError(f"Expected string content, got {type(content).__name__}")
    return content


@dataclass(frozen=True)
class ClusterSpec:
    count: int = 1
    thunderbolt: Thunderbolt | None = None
    min_memory_gb: float | None = None
    chip: Chip | None = None


@dataclass(frozen=True)
class InstanceSpec:
    model_id: str
    sharding: Sharding = Sharding.PIPELINE
    comm: Comm = Comm.RING
    min_nodes: int = 1


def parse_cluster_marker(marker) -> ClusterSpec:
    if marker is None:
        return ClusterSpec()
    return ClusterSpec(
        count=marker.kwargs.get("count", 1),
        thunderbolt=marker.kwargs.get("thunderbolt"),
        min_memory_gb=marker.kwargs.get("min_memory"),
        chip=marker.kwargs.get("chip"),
    )


def parse_instance_marker(marker) -> InstanceSpec | None:
    if marker is None:
        return None
    if not marker.args:
        raise ValueError(
            "@pytest.mark.instance requires a positional model_id argument"
        )
    return InstanceSpec(
        model_id=marker.args[0],
        sharding=marker.kwargs.get("sharding", Sharding.PIPELINE),
        comm=marker.kwargs.get("comm", Comm.RING),
        min_nodes=marker.kwargs.get("min_nodes", 1),
    )


@dataclass
class Session:
    cluster: ClusterInfo
    eco: EcoSession
    instance_spec: InstanceSpec | None = None
    instance_id: str | None = None
    _stopped_hosts: set[str] = field(default_factory=set)

    @property
    def client(self) -> ExoClient:
        for host in self.cluster.hosts:
            if host not in self._stopped_hosts:
                return make_client_from_url(self.cluster.api_endpoints[host])
        return self.cluster.make_client()

    @property
    def state(self) -> dict[str, Any]:
        return self.client.request_json("GET", "/state") or {}

    @property
    def instances(self) -> dict[str, Any]:
        return self.state.get("instances", {})

    # ---- Inference ----

    def chat(self, prompt: str, max_tokens: int = 100) -> str:
        resp = self.chat_raw(prompt, max_tokens=max_tokens)
        return _extract_content(resp)

    def chat_raw(self, prompt: str, **kwargs: Any) -> ChatCompletionResponse:
        if not self.instance_spec:
            raise RuntimeError(
                "No instance placed; add @pytest.mark.instance to the test"
            )
        max_tokens = kwargs.pop("max_tokens", 100)
        request = ChatCompletionRequest.model_validate(
            {
                "model": self.instance_spec.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                **kwargs,
            }
        )
        return self._post_chat(request)

    def multi_turn(self, messages: list[dict[str, str]], max_tokens: int = 100) -> str:
        if not self.instance_spec:
            raise RuntimeError(
                "No instance placed; add @pytest.mark.instance to the test"
            )
        request = ChatCompletionRequest.model_validate(
            {
                "model": self.instance_spec.model_id,
                "messages": messages,
                "max_tokens": max_tokens,
            }
        )
        return _extract_content(self._post_chat(request))

    def _post_chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raw = self.client.request_json(
            "POST",
            "/v1/chat/completions",
            body=request.model_dump(exclude_none=True),
        )
        return ChatCompletionResponse.model_validate(raw)

    def disconnect_node(self, index: int) -> None:
        """Stop exo on a node and wait for the cluster to observe the disconnect."""
        host = self.cluster.hosts[index]
        self.eco.stop([host], keep=True)
        self._stopped_hosts.add(host)

    def reconnect_node(self, index: int) -> None:
        """Restart a previously disconnected node into the existing namespace."""
        host = self.cluster.hosts[index]
        self.eco.start_hosts([host], namespace=self.cluster.namespace)
        self._stopped_hosts.discard(host)

    def wait_ready(
        self, expected_nodes: int | None = None, timeout: float = 60
    ) -> None:
        """Wait until the cluster has exactly `expected_nodes` visible and reporting memory.

        Defaults to the count of non-stopped hosts. Use this after
        `disconnect_node` / `reconnect_node` to wait for the cluster to settle.
        """
        if expected_nodes is None:
            expected_nodes = len(self.cluster.hosts) - len(self._stopped_hosts)
        start = time.time()
        while time.time() - start < timeout:
            try:
                state = self.state
                identities = len(state.get("nodeIdentities", {}))
                memory = len(state.get("nodeMemory", {}))
                if identities == expected_nodes and memory == expected_nodes:
                    return
            except Exception:
                pass
            time.sleep(2.0)
        raise TimeoutError(
            f"Cluster did not reach exactly {expected_nodes} ready nodes within {timeout}s"
        )
