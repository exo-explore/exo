from exo.utils.info_gatherer.system_info import (
    _parse_networksetup_ports,
    get_heuristic_interface_type,
)


def test_parse_networksetup_ports():
    sample_output = """Hardware Port: Wi-Fi
Device: en0
Ethernet Address: aa:bb:cc:dd:ee:01

Hardware Port: Ethernet
Device: en1
Ethernet Address: aa:bb:cc:dd:ee:02

Hardware Port: Thunderbolt Bridge
Device: bridge0
Ethernet Address: aa:bb:cc:dd:ee:03

Hardware Port: Thunderbolt 5 Port 1
Device: en4
Ethernet Address: aa:bb:cc:dd:ee:04

Hardware Port: Virtual Adapter
Device: en5
Ethernet Address: aa:bb:cc:dd:ee:05
"""
    parsed = _parse_networksetup_ports(sample_output)

    assert parsed.get("en0") == "wifi"
    assert parsed.get("en1") == "ethernet"
    assert parsed.get("bridge0") == "thunderbolt"
    assert parsed.get("en4") == "tb5"
    # en5 with unknown hardware port type falls back to maybe_ethernet
    assert parsed.get("en5") == "maybe_ethernet"


def test_heuristic_interface_type():
    # Thunderbolt & TB5
    assert get_heuristic_interface_type("tb0") == "thunderbolt"
    assert get_heuristic_interface_type("thunderbolt0") == "thunderbolt"
    assert get_heuristic_interface_type("tb5_0") == "tb5"
    assert get_heuristic_interface_type("ten0") == "thunderbolt"

    # RDMA / InfiniBand
    assert get_heuristic_interface_type("ib0") == "rdma"
    assert get_heuristic_interface_type("mlx5_core0") == "rdma"

    # Ethernet
    assert get_heuristic_interface_type("eth0") == "ethernet"
    assert get_heuristic_interface_type("eno1") == "ethernet"
    assert get_heuristic_interface_type("ens33") == "ethernet"
    assert get_heuristic_interface_type("enp5s0") == "ethernet"

    # WiFi
    assert get_heuristic_interface_type("wlan0") == "wifi"
    assert get_heuristic_interface_type("wl0") == "wifi"
    assert get_heuristic_interface_type("wifi0") == "wifi"

    # Virtual & loopback
    assert get_heuristic_interface_type("docker0") == "virtual"
    assert get_heuristic_interface_type("veth9812") == "virtual"
    assert get_heuristic_interface_type("br-1234") == "virtual"
    assert get_heuristic_interface_type("tun0") == "virtual"
    assert get_heuristic_interface_type("tap0") == "virtual"
    assert get_heuristic_interface_type("lo") == "virtual"

    # Unknown
    assert get_heuristic_interface_type("custom_dummy0") == "unknown"
