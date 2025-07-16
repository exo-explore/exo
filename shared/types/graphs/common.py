from collections.abc import Mapping
from typing import Callable, Generic, Protocol, Set, Tuple, TypeVar, overload

from pydantic import BaseModel

from shared.types.common import NewUUID

EdgeTypeT = TypeVar("EdgeTypeT", covariant=True)
VertexTypeT = TypeVar("VertexTypeT", covariant=True)
EdgeIdT = TypeVar("EdgeIdT", bound=NewUUID)
VertexIdT = TypeVar("VertexIdT", bound=NewUUID)


class VertexData(BaseModel, Generic[VertexTypeT]):
    vertex_type: VertexTypeT


class EdgeData(BaseModel, Generic[EdgeTypeT]):
    edge_type: EdgeTypeT


class BaseEdge(BaseModel, Generic[EdgeTypeT, EdgeIdT, VertexIdT]):
    edge_vertices: Tuple[VertexIdT, VertexIdT]
    edge_data: EdgeData[EdgeTypeT]


class BaseVertex(BaseModel, Generic[VertexTypeT, EdgeIdT]):
    vertex_data: VertexData[VertexTypeT]


class Vertex(
    BaseVertex[VertexTypeT, EdgeIdT], Generic[VertexTypeT, EdgeIdT, VertexIdT]
):
    vertex_id: VertexIdT


class Edge(
    BaseEdge[EdgeTypeT, EdgeIdT, VertexIdT], Generic[EdgeTypeT, EdgeIdT, VertexIdT]
):
    edge_id: EdgeIdT


class GraphData(BaseModel, Generic[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT]):
    edges: Mapping[EdgeIdT, EdgeData[EdgeTypeT]] = {}
    vertices: Mapping[VertexIdT, VertexData[VertexTypeT]] = {}


class GraphProtocol(Protocol, Generic[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT]):
    def list_edges(self) -> Set[EdgeIdT]: ...
    def list_vertices(self) -> Set[VertexIdT]: ...
    def get_vertices_from_edges(
        self, edges: Set[EdgeIdT]
    ) -> Mapping[EdgeIdT, Set[VertexIdT]]: ...
    def get_edges_from_vertices(
        self, vertices: Set[VertexIdT]
    ) -> Mapping[VertexIdT, Set[EdgeIdT]]: ...
    def get_edge_data(
        self, edges: Set[EdgeIdT]
    ) -> Mapping[EdgeIdT, EdgeData[EdgeTypeT]]: ...
    def get_vertex_data(
        self, vertices: Set[VertexIdT]
    ) -> Mapping[VertexIdT, VertexData[VertexTypeT]]: ...


class MutableGraphProtocol(GraphProtocol[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT]):
    def check_edges_exists(self, edge_id: EdgeIdT) -> bool: ...
    def check_vertex_exists(self, vertex_id: VertexIdT) -> bool: ...
    def _add_edge(self, edge_id: EdgeIdT, edge_data: EdgeData[EdgeTypeT]) -> None: ...
    def _add_vertex(
        self, vertex_id: VertexIdT, vertex_data: VertexData[VertexTypeT]
    ) -> None: ...
    def _remove_edge(self, edge_id: EdgeIdT) -> None: ...
    def _remove_vertex(self, vertex_id: VertexIdT) -> None: ...
    ###
    @overload
    def attach_edge(self, edge: Edge[EdgeTypeT, EdgeIdT, VertexIdT]) -> None: ...
    @overload
    def attach_edge(
        self,
        edge: Edge[EdgeTypeT, EdgeIdT, VertexIdT],
        extra_vertex: Vertex[VertexTypeT, EdgeIdT, VertexIdT],
    ) -> None: ...
    def attach_edge(
        self,
        edge: Edge[EdgeTypeT, EdgeIdT, VertexIdT],
        extra_vertex: Vertex[VertexTypeT, EdgeIdT, VertexIdT] | None = None,
    ) -> None:
        base_vertex = edge.edge_vertices[0]
        target_vertex = edge.edge_vertices[1]
        base_vertex_exists = self.check_vertex_exists(base_vertex)
        target_vertex_exists = self.check_vertex_exists(target_vertex)

        if not base_vertex_exists:
            raise ValueError("Base Vertex Does Not Exist")

        match (target_vertex_exists, extra_vertex is not None):
            case (True, False):
                raise ValueError("New Vertex Already Exists")
            case (False, True):
                if extra_vertex is None:
                    raise ValueError("BUG: Extra Vertex Must Be Provided")
                self._add_vertex(extra_vertex.vertex_id, extra_vertex.vertex_data)
            case (False, False):
                raise ValueError(
                    "New Vertex Must Be Provided For Non-Existent Target Vertex"
                )
            case (True, True):
                raise ValueError("New Vertex Already Exists")

        self._add_edge(edge.edge_id, edge.edge_data)


class BaseGraph(
    Generic[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT],
    MutableGraphProtocol[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT],
):
    graph_data: GraphData[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT] = GraphData[
        EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT
    ]()


# the first element in the return value is the filtered graph; the second is the
# (possibly empty) set of sub-graphs that were detached during filtering.
def filter_by_edge_data(
    graph: BaseGraph[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT],
    keep: VertexIdT,
    predicate: Callable[[EdgeData[EdgeTypeT]], bool],
) -> Tuple[
    BaseGraph[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT],
    Set[BaseGraph[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT]],
]: ...


# the first element in the return value is the filtered graph; the second is the
# (possibly empty) set of sub-graphs that were detached during filtering.
def filter_by_vertex_data(
    graph: BaseGraph[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT],
    keep: VertexIdT,
    predicate: Callable[[VertexData[VertexTypeT]], bool],
) -> Tuple[
    BaseGraph[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT],
    Set[BaseGraph[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT]],
]: ...


def map_vertices_onto_graph(
    vertices: Mapping[VertexIdT, VertexData[VertexTypeT]],
    graph: BaseGraph[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT],
) -> Tuple[BaseGraph[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT], Set[VertexIdT]]: ...


def map_edges_onto_graph(
    edges: Mapping[EdgeIdT, EdgeData[EdgeTypeT]],
    graph: BaseGraph[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT],
) -> Tuple[BaseGraph[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT], Set[EdgeIdT]]: ...


def split_graph_by_edge(
    graph: BaseGraph[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT],
    edge: EdgeIdT,
    keep: VertexIdT,
) -> Tuple[
    BaseGraph[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT],
    Set[BaseGraph[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT]],
]: ...


def merge_graphs_by_edge(
    graphs: Set[BaseGraph[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT]],
    edge: EdgeIdT,
    keep: VertexIdT,
) -> Tuple[BaseGraph[EdgeTypeT, VertexTypeT, EdgeIdT, VertexIdT], Set[EdgeIdT]]: ...
