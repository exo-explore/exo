from exo.utils.profilers.rdma_probe_main import build_two_rank_ibv_devs


def test_build_two_rank_ibv_devs_uses_local_iface_per_rank():
    assert build_two_rank_ibv_devs(
        source_iface="rdma_source", sink_iface="rdma_sink"
    ) == [
        [None, "rdma_source"],
        ["rdma_sink", None],
    ]
