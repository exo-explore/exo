import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import httpx
import tomlkit

from exo.api.types import PlacementPreviewResponse
from exo.download.hub import RepoFile
from exo.download.store import Manifest, write_manifest
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId
from exo.shared.types.worker.instances import Instance
from exo.shared.types.worker.shards import Sharding
from exo.testkit.cluster import Cluster, Node, NoPlacementError, StubModel

MODEL = ModelId("exo-integration/tensorfold-protocol-fixture")
REPOSITORY = Path(__file__).resolve().parents[3]
FIXTURE_BINARY = (
    REPOSITORY / "src/exo/worker/engines/tensorfold/tests/fixture_server.py"
)


class NativeCluster(Cluster):
    def __init__(self, size: int, root: Path):
        super().__init__(size, root, models=(StubModel(model_id=MODEL),))

    def _prepare(self, node: Node) -> None:
        super()._prepare(node)
        card = (
            self.models[0]
            .card()
            .model_copy(
                update={
                    "backends": [
                        Backend(
                            "TensorFoldMetal"
                            if sys.platform == "darwin"
                            else "TensorFoldCuda"
                        )
                    ],
                    "model_type": "qwen3",
                    "family": "qwen3",
                    "quantization": "4bit",
                    "tensorfold_format": "mlx-q4g64",
                    "capabilities": ["text", "thinking", "thinking_toggle"],
                    "trust_remote_code": False,
                    "supports_tensor": False,
                    "context_length": 4096,
                }
            )
        )
        document = card.model_dump(
            mode="json", exclude_none=True, exclude={"is_custom"}
        )
        (node.home / "custom_model_cards" / f"{MODEL.normalize()}.toml").write_text(
            tomlkit.dumps(document)  # pyright: ignore[reportUnknownMemberType]
        )
        model_dir = node.home / "models" / MODEL.normalize()
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3",
                    "architectures": ["Qwen3ForCausalLM"],
                    "num_hidden_layers": 8,
                    "hidden_size": 1024,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 8,
                    "vocab_size": 512,
                    "max_position_embeddings": 4096,
                    "quantization": {"bits": 4, "group_size": 64},
                }
            )
        )
        (model_dir / "tokenizer.json").write_text("{}")
        write_manifest(
            model_dir,
            Manifest(
                repo=MODEL,
                commit="protocol-fixture-only",
                files=[
                    RepoFile(path=path.name, size=path.stat().st_size)
                    for path in model_dir.iterdir()
                    if path.name
                    in {"config.json", "tokenizer.json", "model.safetensors"}
                ],
                complete=True,
            ),
        )

    def _environment(self, node: Node) -> dict[str, str]:
        environment = super()._environment(node)
        environment.update(
            {
                "PYTHONPATH": os.pathsep.join(
                    [str(REPOSITORY), str(REPOSITORY / "src")]
                ),
                "PATH": os.pathsep.join(
                    [str(Path(sys.executable).parent), os.environ.get("PATH", "")]
                ),
                "EXO_TENSORFOLD_BINARY": str(FIXTURE_BINARY),
                "EXO_TENSORFOLD_FIXTURE_RECORD": str(node.home / "native-record.jsonl"),
                "EXO_TENSORFOLD_STARTUP_TIMEOUT": "15",
                "EXO_TENSORFOLD_SHUTDOWN_TIMEOUT": "2",
                "EXO_TENSORFOLD_REQUEST_TIMEOUT": "15",
            }
        )
        return environment

    def _spawn(self, node: Node) -> None:
        self.network.set_node_id(node.name, node.node_id)
        self._write_network_table()
        arguments = [
            sys.executable,
            "-m",
            "exo.testkit.tensorfold_node",
            node.node_id,
            "--namespace",
            self.namespace,
            "--api-port",
            str(node.api_port),
            "--zenoh-port",
            str(node.zenoh_port),
            "--discovery-port",
            str(node.discovery_port),
            "--offline",
        ]
        with node.output_path.open("ab") as output:
            node.process = subprocess.Popen(
                arguments,
                env=self._environment(node),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=node.home,
            )
        node.status = "running"
        node.started_at = time.monotonic()
        self.note(
            f"native fixture node {node.name} started, process {node.process.pid}"
        )

    def preview(
        self,
        on: tuple[int, ...],
        *,
        model: ModelId = MODEL,
        sharding: Sharding = Sharding.Pipeline,
        via: int = 0,
    ) -> Instance:
        wanted = {self.nodes[index].node_id for index in on}
        response = self._http.get(
            self.nodes[via].url + "/instance/previews",
            params=[("model_id", model), *(("node_ids", node) for node in wanted)],
            timeout=30,
        )
        response.raise_for_status()
        previews = PlacementPreviewResponse.model_validate_json(response.content)
        for preview in previews.previews:
            if (
                preview.instance is not None
                and preview.instance_meta.value == "TensorFold"
                and preview.sharding == sharding
                and set(preview.instance.shard_assignments.node_to_runner) == wanted
            ):
                return preview.instance.model_copy(
                    update={"context_length": 4096, "parallel": 2}
                )
        raise NoPlacementError(f"No TensorFold placement: {previews.model_dump_json()}")

    def native_events(self, node: int = 0) -> list[dict[str, Any]]:
        destination = self.nodes[node].home / "native-record.jsonl"
        if not destination.exists():
            return []
        return [
            cast(dict[str, Any], json.loads(line))
            for line in destination.read_text().splitlines()
            if line
        ]

    def request(
        self, prompt: str, *, via: int = 0, **options: object
    ) -> httpx.Response:
        return httpx.post(
            self.nodes[via].url + "/v1/chat/completions",
            json={
                "model": str(MODEL),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8,
                "temperature": 0,
                **options,
            },
            timeout=30,
            trust_env=False,
        )
