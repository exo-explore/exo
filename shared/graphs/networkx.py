from dataclasses import dataclass
from typing import Mapping, Set

import rustworkx as rx
from pydantic import TypeAdapter

from shared.types.graphs.common import (
    Edge,
    EdgeData,
    EdgeIdT,
    EdgeTypeT,
    MutableGraphProtocol,
    Vertex,
    VertexData,
    VertexIdT,
    VertexTypeT,
)


@dataclass(frozen=True)
class _VertexWrapper[VertexTypeT, VertexIdT]:
    """Internal wrapper to store vertex ID alongside vertex data."""

    vertex_id: VertexIdT
    vertex_data: VertexData[VertexTypeT]


@dataclass(frozen=True)
class _EdgeWrapper[EdgeTypeT, EdgeIdT]:
    """Internal wrapper to store edge ID alongside edge data."""

    edge_id: EdgeIdT
    edge_data: EdgeData[EdgeTypeT]


class NetworkXGraph(MutableGraphProtocol[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT]):
    edge_base: TypeAdapter[EdgeTypeT]
    vertex_base: TypeAdapter[VertexTypeT]

    _graph: rx.PyDiGraph[
        _VertexWrapper[VertexTypeT, VertexIdT], _EdgeWrapper[EdgeTypeT, EdgeIdT]
    ]
    _vertex_id_to_index: dict[VertexIdT, int]
    _edge_id_to_endpoints: dict[EdgeIdT, tuple[int, int]]

    def __init__(
        self, edge_base: TypeAdapter[EdgeTypeT], vertex_base: TypeAdapter[VertexTypeT]
    ) -> None:
        self.edge_base = edge_base
        self.vertex_base = vertex_base
        self._graph = rx.PyDiGraph()
        self._vertex_id_to_index = {}
        self._edge_id_to_endpoints = {}

    ###
    # GraphProtocol methods
    ###

    def list_edges(self) -> Set[EdgeIdT]:
        return {edge.edge_id for edge in self._graph.edges()}

    def list_vertices(self) -> Set[VertexIdT]:
        return {node.vertex_id for node in self._graph.nodes()}

    def get_vertices_from_edges(
        self, edges: Set[EdgeIdT]
    ) -> Mapping[EdgeIdT, Set[VertexIdT]]:
        result: dict[EdgeIdT, Set[VertexIdT]] = {}

        for edge_id in edges:
            if edge_id in self._edge_id_to_endpoints:
                u_idx, v_idx = self._edge_id_to_endpoints[edge_id]
                u_data = self._graph.get_node_data(u_idx)
                v_data = self._graph.get_node_data(v_idx)
                result[edge_id] = {u_data.vertex_id, v_data.vertex_id}

        return result

    def get_edges_from_vertices(
        self, vertices: Set[VertexIdT]
    ) -> Mapping[VertexIdT, Set[EdgeIdT]]:
        result: dict[VertexIdT, Set[EdgeIdT]] = {}

        for vertex_id in vertices:
            if vertex_id in self._vertex_id_to_index:
                vertex_idx = self._vertex_id_to_index[vertex_id]
                edge_ids: Set[EdgeIdT] = set()

                # Get outgoing edges
                for _, _, edge_data in self._graph.out_edges(vertex_idx):
                    edge_ids.add(edge_data.edge_id)

                # Get incoming edges
                for _, _, edge_data in self._graph.in_edges(vertex_idx):
                    edge_ids.add(edge_data.edge_id)

                result[vertex_id] = edge_ids

        return result

    def get_edge_data(
        self, edges: Set[EdgeIdT]
    ) -> Mapping[EdgeIdT, EdgeData[EdgeTypeT]]:
        result: dict[EdgeIdT, EdgeData[EdgeTypeT]] = {}

        for edge_id in edges:
            if edge_id in self._edge_id_to_endpoints:
                u_idx, v_idx = self._edge_id_to_endpoints[edge_id]
                edge_wrapper = self._graph.get_edge_data(u_idx, v_idx)
                result[edge_id] = edge_wrapper.edge_data

        return result

    def get_vertex_data(
        self, vertices: Set[VertexIdT]
    ) -> Mapping[VertexIdT, VertexData[VertexTypeT]]:
        result: dict[VertexIdT, VertexData[VertexTypeT]] = {}

        for vertex_id in vertices:
            if vertex_id in self._vertex_id_to_index:
                vertex_idx = self._vertex_id_to_index[vertex_id]
                vertex_wrapper = self._graph.get_node_data(vertex_idx)
                result[vertex_id] = vertex_wrapper.vertex_data

        return result

    ###
    # MutableGraphProtocol methods
    ###

    def check_edges_exists(self, edge_id: EdgeIdT) -> bool:
        return edge_id in self._edge_id_to_endpoints

    def check_vertex_exists(self, vertex_id: VertexIdT) -> bool:
        return vertex_id in self._vertex_id_to_index

    def _add_edge(self, edge_id: EdgeIdT, edge_data: EdgeData[EdgeTypeT]) -> None:
        # This internal method is not used in favor of a safer `attach_edge` implementation.
        raise NotImplementedError(
            "Use attach_edge to add edges. The internal _add_edge protocol method is flawed."
        )

    def _add_vertex(
        self, vertex_id: VertexIdT, vertex_data: VertexData[VertexTypeT]
    ) -> None:
        if vertex_id not in self._vertex_id_to_index:
            wrapper = _VertexWrapper(vertex_id=vertex_id, vertex_data=vertex_data)
            idx = self._graph.add_node(wrapper)
            self._vertex_id_to_index[vertex_id] = idx

    def _remove_edge(self, edge_id: EdgeIdT) -> None:
        if edge_id in self._edge_id_to_endpoints:
            u_idx, v_idx = self._edge_id_to_endpoints[edge_id]
            self._graph.remove_edge(u_idx, v_idx)
            del self._edge_id_to_endpoints[edge_id]
        else:
            raise ValueError(f"Edge with id {edge_id} not found.")

    def _remove_vertex(self, vertex_id: VertexIdT) -> None:
        if vertex_id in self._vertex_id_to_index:
            vertex_idx = self._vertex_id_to_index[vertex_id]

            # Remove any edges connected to this vertex from our mapping
            edges_to_remove: list[EdgeIdT] = []
            for edge_id, (u_idx, v_idx) in self._edge_id_to_endpoints.items():
                if u_idx == vertex_idx or v_idx == vertex_idx:
                    edges_to_remove.append(edge_id)

            for edge_id in edges_to_remove:
                del self._edge_id_to_endpoints[edge_id]

            # Remove the vertex from the graph
            self._graph.remove_node(vertex_idx)
            del self._vertex_id_to_index[vertex_id]
        else:
            raise ValueError(f"Vertex with id {vertex_id} not found.")

    def attach_edge(
        self,
        edge: Edge[EdgeTypeT, EdgeIdT, VertexIdT],
        extra_vertex: Vertex[VertexTypeT, EdgeIdT, VertexIdT] | None = None,
    ) -> None:
        """
        Attaches an edge to the graph, overriding the default protocol implementation.

        This implementation corrects a flaw in the protocol's `_add_edge`
        signature and provides more intuitive behavior when connecting existing vertices.
        """
        base_vertex_id, target_vertex_id = edge.edge_vertices

        if not self.check_vertex_exists(base_vertex_id):
            raise ValueError(f"Base vertex {base_vertex_id} does not exist.")

        target_vertex_exists = self.check_vertex_exists(target_vertex_id)

        if not target_vertex_exists:
            if extra_vertex is None:
                raise ValueError(
                    f"Target vertex {target_vertex_id} does not exist and no `extra_vertex` was provided."
                )
            if extra_vertex.vertex_id != target_vertex_id:
                raise ValueError(
                    f"The ID of `extra_vertex` ({extra_vertex.vertex_id}) does not match "
                    f"the target vertex ID of the edge ({target_vertex_id})."
                )
            self._add_vertex(extra_vertex.vertex_id, extra_vertex.vertex_data)
        elif extra_vertex is not None:
            raise ValueError(
                f"Target vertex {target_vertex_id} already exists, but `extra_vertex` was provided."
            )

        # Get the internal indices
        base_idx = self._vertex_id_to_index[base_vertex_id]
        target_idx = self._vertex_id_to_index[target_vertex_id]

        # Create edge wrapper and add to graph
        edge_wrapper = _EdgeWrapper(edge_id=edge.edge_id, edge_data=edge.edge_data)
        self._graph.add_edge(base_idx, target_idx, edge_wrapper)

        # Store the mapping
        self._edge_id_to_endpoints[edge.edge_id] = (base_idx, target_idx)
