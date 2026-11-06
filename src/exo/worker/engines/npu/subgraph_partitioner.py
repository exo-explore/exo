from collections.abc import Sequence

from exo.shared.types.npu_hal import (
    NPU_FUSIBLE_OPERATORS,
    ComputationNode,
    FusedSubgraphDescriptor,
    NpuBackendType,
)


class DynamicSubgraphPartitioner:
    """Analyzes linear or directed computation graphs to segment dense operators into fused NPU kernels."""

    def partition_graph(
        self,
        nodes: Sequence[ComputationNode],
        preferred_backend: NpuBackendType = "directml",
    ) -> list[FusedSubgraphDescriptor]:
        """Segments a chain of computation nodes into fused NPU subgraphs and host CPU fallback regions."""
        if not nodes:
            return []

        subgraphs: list[FusedSubgraphDescriptor] = []
        current_cluster: list[ComputationNode] = []
        current_is_npu: bool | None = None
        cluster_idx = 0

        # Build global consumer map to track which node outputs are needed later
        all_outputs_consumed: set[str] = set()
        for i, _node in enumerate(nodes):
            for subsequent_node in nodes[i + 1 :]:
                for inp in subsequent_node.inputs:
                    all_outputs_consumed.add(inp)

        def flush_cluster() -> None:
            nonlocal cluster_idx, current_cluster, current_is_npu
            if not current_cluster:
                return

            cluster_idx += 1
            is_npu = current_is_npu if current_is_npu is not None else False
            backend: NpuBackendType = preferred_backend if is_npu else "cpu_fallback"

            cluster_produced_outputs: set[str] = set()
            for node in current_cluster:
                cluster_produced_outputs.update(node.outputs)

            # Subgraph inputs: any node input that was NOT produced inside this cluster
            subgraph_inputs: list[str] = []
            for node in current_cluster:
                for inp in node.inputs:
                    if inp not in cluster_produced_outputs and inp not in subgraph_inputs:
                        subgraph_inputs.append(inp)

            # Subgraph outputs: any output produced inside cluster that is either consumed downstream
            # or is produced by the final node in the cluster
            subgraph_outputs: list[str] = []
            for node in current_cluster:
                for out in node.outputs:
                    if (out in all_outputs_consumed or node == current_cluster[-1]) and (
                        out not in subgraph_outputs
                    ):
                        subgraph_outputs.append(out)

            subgraph_id = f"subgraph_{cluster_idx}_{'npu' if is_npu else 'cpu'}"
            descriptor = FusedSubgraphDescriptor(
                subgraph_id=subgraph_id,
                target_backend=backend,
                nodes=tuple(current_cluster),
                input_names=tuple(subgraph_inputs),
                output_names=tuple(subgraph_outputs),
                is_fused_npu=is_npu,
            )
            subgraphs.append(descriptor)
            current_cluster = []
            current_is_npu = None

        for node in nodes:
            node_can_run_on_npu = node.is_npu_supported and (node.op_type in NPU_FUSIBLE_OPERATORS)

            if current_is_npu is None:
                current_is_npu = node_can_run_on_npu
                current_cluster.append(node)
            elif current_is_npu == node_can_run_on_npu:
                current_cluster.append(node)
            else:
                flush_cluster()
                current_is_npu = node_can_run_on_npu
                current_cluster.append(node)

        flush_cluster()
        return subgraphs
