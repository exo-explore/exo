from exo.master.main import orphaned_download_node_ids
from exo.shared.tests.conftest import get_pipeline_shard_metadata
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.state import State
from exo.shared.types.worker.downloads import DownloadPending
from exo.worker.tests.constants import MODEL_A_ID


def test_orphaned_download_node_ids_excludes_connected_nodes() -> None:
    retired_node = NodeId("retired-node")
    connected_node = NodeId("connected-node")
    topology = Topology()
    topology.add_node(connected_node)
    shard = get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0, world_size=1)

    state = State(
        topology=topology,
        downloads={
            retired_node: [DownloadPending(node_id=retired_node, shard_metadata=shard)],
            connected_node: [
                DownloadPending(node_id=connected_node, shard_metadata=shard)
            ],
        },
    )

    assert orphaned_download_node_ids(state) == {retired_node}
