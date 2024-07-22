from exo.orchestration import Node
from prometheus_client import start_http_server, Counter, Histogram
import json
from typing import List

# Create metrics to track time spent and requests made.
PROCESS_PROMPT_COUNTER = Counter('process_prompt_total', 'Total number of prompts processed', ['node_id'])
PROCESS_TENSOR_COUNTER = Counter('process_tensor_total', 'Total number of tensors processed', ['node_id'])
PROCESS_TENSOR_TIME = Histogram('process_tensor_seconds', 'Time spent processing tensor', ['node_id'])

def start_metrics_server(node: Node, port: int):
    start_http_server(port)

    def _on_opaque_status(request_id, opaque_status: str):
        status_data = json.loads(opaque_status)
        type = status_data.get("type", "")
        node_id = status_data.get("node_id", "")
        if type != "node_status": return
        status = status_data.get("status", "")

        if status == "end_process_prompt":
            PROCESS_PROMPT_COUNTER.labels(node_id=node_id).inc()
        elif status == "end_process_tensor":
            elapsed_time_ns = status_data.get("elapsed_time_ns", 0)
            PROCESS_TENSOR_COUNTER.labels(node_id=node_id).inc()
            PROCESS_TENSOR_TIME.labels(node_id=node_id).observe(elapsed_time_ns / 1e9)  # Convert ns to seconds

    node.on_opaque_status.register("stats").on_next(_on_opaque_status)