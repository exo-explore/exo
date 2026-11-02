"""Unit tests for RDMA physical connection validation with ibv_devinfo (TODO #28)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.shared.apply import apply
from exo.shared.types.common import NodeId
from exo.shared.types.events import IndexedEvent, NodeGatheredInfo
from exo.shared.types.profiling import RdmaDeviceInfo, RdmaPortInfo
from exo.shared.types.state import State
from exo.utils.info_gatherer.info_gatherer import RdmaCtlStatus
from exo.utils.info_gatherer.rdma_info import (
    gather_ibv_devinfo,
    is_rdma_device_active,
    parse_ibv_devinfo_output,
    validate_rdma_connections,
)

SAMPLE_IBV_INFINIBAND_ACTIVE = """
hca_id:\tmlx5_0
\ttransport:\t\t\tInfiniBand (IB)
\tfw_ver:\t\t\t\t16.35.1012
\tnode_guid:\t\t\t0c42:a103:006d:8412
\tsys_image_guid:\t\t\t0c42:a103:006d:8412
\tvendor_id:\t\t\t0x02c9
\tvendor_part_id:\t\t\t4119
\thw_ver:\t\t\t\t0x0
\tboard_id:\t\t\tMT_0000000010
\tphys_port_cnt:\t\t\t1
\t\tport:\t1
\t\t\tstate:\t\t\tPORT_ACTIVE (4)
\t\t\tmax_mtu:\t\t4096 (5)
\t\t\tactive_mtu:\t\t4096 (5)
\t\t\tsm_lid:\t\t\t1
\t\t\tport_lid:\t\t12
\t\t\tport_lmc:\t\t0x00
\t\t\tlink_layer:\t\tInfiniBand
\t\t\tphysical_state:\t\tLinkUp (5)
\t\t\tactive_width:\t\t4X (2)
\t\t\tactive_speed:\t\t25.0 Gbps (EDR) (8)
"""

SAMPLE_IBV_ROCE_ACTIVE = """
hca_id:\tmlx5_1
\ttransport:\t\t\tInfiniBand (IB)
\tfw_ver:\t\t\t\t20.31.1014
\tnode_guid:\t\t\te41d:2d03:002b:3e40
\tphys_port_cnt:\t\t\t1
\t\tport:\t1
\t\t\tstate:\t\t\tPORT_ACTIVE (4)
\t\t\tmax_mtu:\t\t4096 (5)
\t\t\tactive_mtu:\t\t1024 (3)
\t\t\tsm_lid:\t\t\t0
\t\t\tport_lid:\t\t0
\t\t\tport_lmc:\t\t0x00
\t\t\tlink_layer:\t\tEthernet
\t\t\tphysical_state:\t\tLinkUp (5)
\t\t\tactive_width:\t\t4X (2)
\t\t\tactive_speed:\t\t25.0 Gbps (EDR) (8)
"""

SAMPLE_IBV_MULTI_PORT = """
hca_id:\tmlx5_0
\ttransport:\t\t\tInfiniBand (IB)
\tfw_ver:\t\t\t\t16.35.1012
\tnode_guid:\t\t\t0c42:a103:006d:8412
\tphys_port_cnt:\t\t\t2
\t\tport:\t1
\t\t\tstate:\t\t\tPORT_ACTIVE (4)
\t\t\tphysical_state:\t\tLinkUp (5)
\t\t\tlink_layer:\t\tInfiniBand
\t\t\tactive_width:\t\t4X (2)
\t\t\tactive_speed:\t\t25.0 Gbps (EDR) (8)
\t\tport:\t2
\t\t\tstate:\t\t\tPORT_DOWN (1)
\t\t\tphysical_state:\t\tDisabled (1)
\t\t\tlink_layer:\t\tInfiniBand
"""

SAMPLE_IBV_MULTI_HCA = """
hca_id:\tmlx5_0
\ttransport:\t\t\tInfiniBand (IB)
\tfw_ver:\t\t\t\t16.35.1012
\tphys_port_cnt:\t\t\t1
\t\tport:\t1
\t\t\tstate:\t\t\tPORT_ACTIVE (4)
\t\t\tphysical_state:\t\tLinkUp (5)
hca_id:\tmlx5_1
\ttransport:\t\t\tInfiniBand (IB)
\tfw_ver:\t\t\t\t20.31.1014
\tphys_port_cnt:\t\t\t1
\t\tport:\t1
\t\t\tstate:\t\t\tPORT_DOWN (1)
\t\t\tphysical_state:\t\tDisabled (1)
"""

SAMPLE_IBV_ALL_DOWN = """
hca_id:\tmlx5_0
\ttransport:\t\t\tInfiniBand (IB)
\tfw_ver:\t\t\t\t16.35.1012
\tphys_port_cnt:\t\t\t1
\t\tport:\t1
\t\t\tstate:\t\t\tPORT_DOWN (1)
\t\t\tphysical_state:\t\tDisabled (1)
"""


def test_parse_ibv_devinfo_infiniband_active():
    devices = parse_ibv_devinfo_output(SAMPLE_IBV_INFINIBAND_ACTIVE)
    assert len(devices) == 1
    dev = devices[0]
    assert dev.hca_id == "mlx5_0"
    assert dev.transport == "InfiniBand (IB)"
    assert dev.fw_ver == "16.35.1012"
    assert dev.node_guid == "0c42:a103:006d:8412"
    assert len(dev.ports) == 1

    port = dev.ports[0]
    assert port.port == 1
    assert "PORT_ACTIVE" in port.state
    assert "LinkUp" in port.physical_state
    assert port.link_layer == "InfiniBand"
    assert "25.0 Gbps" in port.active_speed
    assert "4X" in port.active_width

    assert is_rdma_device_active(dev) is True
    assert validate_rdma_connections(devices) is True


def test_parse_ibv_devinfo_roce_active():
    devices = parse_ibv_devinfo_output(SAMPLE_IBV_ROCE_ACTIVE)
    assert len(devices) == 1
    dev = devices[0]
    assert dev.hca_id == "mlx5_1"
    assert len(dev.ports) == 1
    port = dev.ports[0]
    assert port.link_layer == "Ethernet"
    assert "PORT_ACTIVE" in port.state
    assert "LinkUp" in port.physical_state

    assert is_rdma_device_active(dev) is True
    assert validate_rdma_connections(devices) is True


def test_parse_ibv_devinfo_multi_port():
    devices = parse_ibv_devinfo_output(SAMPLE_IBV_MULTI_PORT)
    assert len(devices) == 1
    dev = devices[0]
    assert len(dev.ports) == 2
    assert dev.ports[0].port == 1
    assert "PORT_ACTIVE" in dev.ports[0].state
    assert "LinkUp" in dev.ports[0].physical_state

    assert dev.ports[1].port == 2
    assert "PORT_DOWN" in dev.ports[1].state
    assert "Disabled" in dev.ports[1].physical_state

    assert is_rdma_device_active(dev) is True
    assert validate_rdma_connections(devices) is True


def test_parse_ibv_devinfo_multi_hca():
    devices = parse_ibv_devinfo_output(SAMPLE_IBV_MULTI_HCA)
    assert len(devices) == 2
    assert devices[0].hca_id == "mlx5_0"
    assert devices[1].hca_id == "mlx5_1"

    assert is_rdma_device_active(devices[0]) is True
    assert is_rdma_device_active(devices[1]) is False
    assert validate_rdma_connections(devices) is True


def test_parse_ibv_devinfo_all_ports_down():
    devices = parse_ibv_devinfo_output(SAMPLE_IBV_ALL_DOWN)
    assert len(devices) == 1
    assert is_rdma_device_active(devices[0]) is False
    assert validate_rdma_connections(devices) is False


def test_parse_ibv_devinfo_empty_or_no_devices():
    assert parse_ibv_devinfo_output("") == []
    assert parse_ibv_devinfo_output("No IB devices found") == []
    assert parse_ibv_devinfo_output("Failed to get IB devices list") == []
    assert validate_rdma_connections([]) is False


@pytest.mark.asyncio
async def test_gather_ibv_devinfo_success():
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = SAMPLE_IBV_INFINIBAND_ACTIVE.encode("utf-8")

    with (
        patch("shutil.which", return_value="/usr/bin/ibv_devinfo"),
        patch("anyio.run_process", new_callable=AsyncMock, return_value=mock_proc),
    ):
        is_active, devices = await gather_ibv_devinfo()
        assert is_active is True
        assert len(devices) == 1
        assert devices[0].hca_id == "mlx5_0"


@pytest.mark.asyncio
async def test_gather_ibv_devinfo_tool_missing():
    with patch("shutil.which", return_value=None):
        is_active, devices = await gather_ibv_devinfo()
        assert is_active is False
        assert devices == []


@pytest.mark.asyncio
async def test_gather_ibv_devinfo_command_error():
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stdout = b"Failed to query devices"

    with (
        patch("shutil.which", return_value="/usr/bin/ibv_devinfo"),
        patch("anyio.run_process", new_callable=AsyncMock, return_value=mock_proc),
    ):
        is_active, devices = await gather_ibv_devinfo()
        assert is_active is False
        assert devices == []


@pytest.mark.asyncio
async def test_rdma_ctl_status_gather_linux():
    sample_devices = [
        RdmaDeviceInfo(
            hca_id="mlx5_0",
            transport="InfiniBand (IB)",
            ports=[
                RdmaPortInfo(
                    port=1,
                    state="PORT_ACTIVE (4)",
                    physical_state="LinkUp (5)",
                )
            ],
        )
    ]

    with (
        patch("exo.utils.info_gatherer.info_gatherer.IS_DARWIN", False),
        patch("shutil.which", return_value="/usr/bin/ibv_devinfo"),
        patch(
            "exo.utils.info_gatherer.info_gatherer.gather_ibv_devinfo",
            new_callable=AsyncMock,
            return_value=(True, sample_devices),
        ),
    ):
        status = await RdmaCtlStatus.gather()
        assert status is not None
        assert status.enabled is True
        assert len(status.devices) == 1
        assert status.devices[0].hca_id == "mlx5_0"


@pytest.mark.asyncio
async def test_rdma_ctl_status_gather_darwin():
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = b"RDMA is currently enabled on this system"

    with (
        patch("exo.utils.info_gatherer.info_gatherer.IS_DARWIN", True),
        patch("shutil.which", return_value="/usr/local/bin/rdma_ctl"),
        patch("anyio.run_process", new_callable=AsyncMock, return_value=mock_proc),
    ):
        status = await RdmaCtlStatus.gather()
        assert status is not None
        assert status.enabled is True
        assert status.devices == []


def test_apply_rdma_ctl_status_with_devices():
    state = State()
    node_id = NodeId("test-node-1")
    sample_devices = [
        RdmaDeviceInfo(
            hca_id="mlx5_0",
            transport="InfiniBand (IB)",
            ports=[
                RdmaPortInfo(
                    port=1,
                    state="PORT_ACTIVE (4)",
                    physical_state="LinkUp (5)",
                )
            ],
        )
    ]

    event = NodeGatheredInfo(
        node_id=node_id,
        when="2026-10-24T00:00:00Z",
        info=RdmaCtlStatus(enabled=True, devices=sample_devices),
    )

    new_state = apply(state, IndexedEvent(idx=0, event=event))
    assert node_id in new_state.node_rdma_ctl
    node_rdma = new_state.node_rdma_ctl[node_id]
    assert node_rdma.enabled is True
    assert len(node_rdma.devices) == 1
    assert node_rdma.devices[0].hca_id == "mlx5_0"
