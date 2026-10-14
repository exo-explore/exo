import sys

import pytest

from exo.shared.types.thunderbolt import (
    ThunderboltConnectivity,
    ThunderboltConnectivityData,
)
from exo.utils.info_gatherer.info_gatherer import (
    _gather_iface_map,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.anyio
@pytest.mark.skipif(
    sys.platform != "darwin", reason="Thunderbolt info can only be gathered on macos"
)
async def test_tb_parsing():
    data = await ThunderboltConnectivity.gather()
    ifaces = await _gather_iface_map()
    assert ifaces
    assert data
    for datum in data:
        datum.ident(ifaces)
        datum.conn()


def test_conn_resolves_peer_through_intermediate_hub() -> None:
    """A TB hub between two Macs hides the peer one level deeper.

    Reproduces the iVANKY Fusiondock Ultra topology observed on the
    wc-bmbp <-> wc-smbp link, where ``system_profiler`` reports the dock at
    the first ``_items`` level and the peer Mac nested inside it. The parser
    must walk past the dock and surface the peer's domain UUID, otherwise
    half the RDMA mesh stays invisible to the placement engine.
    """
    payload = {
        "domain_uuid_key": "DCA2B6F5-1C58-4589-8DA8-90B9326462D6",
        "receptacle_1_tag": {
            "receptacle_id_key": "2",
            "current_speed_key": "80 Gb/s",
        },
        "_items": [
            {
                "_name": "iVANKY Fusiondock Ultra",
                "_items": [
                    {
                        "_name": "MacBook Pro",
                        "domain_uuid_key": "F74D8F9B-DCDF-40D4-A428-3A3674BCB3F4",
                    }
                ],
            }
        ],
    }
    datum = ThunderboltConnectivityData.model_validate(payload)
    conn = datum.conn()
    assert conn is not None
    assert conn.source_uuid == "DCA2B6F5-1C58-4589-8DA8-90B9326462D6"
    assert conn.sink_uuid == "F74D8F9B-DCDF-40D4-A428-3A3674BCB3F4"


def test_conn_returns_first_peer_for_direct_link() -> None:
    """A direct cable still surfaces the peer at the first level."""
    payload = {
        "domain_uuid_key": "EA94B959-A0C4-1111-1111-111111111111",
        "_items": [
            {
                "_name": "MacBook Pro",
                "domain_uuid_key": "D02B9C20-7504-2222-2222-222222222222",
            }
        ],
    }
    datum = ThunderboltConnectivityData.model_validate(payload)
    conn = datum.conn()
    assert conn is not None
    assert conn.sink_uuid == "D02B9C20-7504-2222-2222-222222222222"


def test_conn_returns_none_when_no_peer_present() -> None:
    """Empty ``_items`` (e.g. unconnected receptacle) yields no edge."""
    payload: dict[str, object] = {
        "domain_uuid_key": "AAA",
        "_items": [],
    }
    datum = ThunderboltConnectivityData.model_validate(payload)
    assert datum.conn() is None
