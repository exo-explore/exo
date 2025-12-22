from exo.routing.connection_message import ConnectionMessage
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, TopologyEdgeCreated
from exo.shared.types.state import State
from exo.shared.types.topology import Connection


def check_connections(
    local_id: NodeId, msg: ConnectionMessage, state: State
) -> list[Event]:
    remote_id = msg.node_id
    sockets = msg.ips
    if (
        not state.topology.contains_node(remote_id)
        or remote_id not in state.node_profiles
    ):
        return []

    out: list[Event] = []
    conns = list(state.topology.list_connections())
    for iface in state.node_profiles[remote_id].network_interfaces:
        if sockets is None:
            continue
        for sock in sockets:
            if iface.ip_address == sock.ip:
                conn = Connection(source_id=local_id, sink_id=remote_id, sink_addr=sock)
                if state.topology.contains_connection(conn):
                    conns.remove(conn)
                    continue
                out.append(TopologyEdgeCreated(edge=conn))

    return out
