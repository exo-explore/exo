import pytest
import sys
from exo.shared.types.thunderbolt import (
    TBConnectivity,
)
from exo.utils.info_gatherer.info_gatherer import _gather_iface_map  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
@pytest.mark.skipif(sys.platform != "darwin", reason="TB info can only be gathered on macos")
async def test_tb_parsing():
    data = await TBConnectivity.gather()
    ifaces = await _gather_iface_map()
    assert ifaces
    assert data
    for datum in data:
        datum.ident(ifaces)
        datum.conn()
