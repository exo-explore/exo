from exo.utils.info_gatherer.system_info import (
    _active_speeds_from_stats,  # pyright: ignore[reportPrivateUsage]
    _parse_supported_media_mbps,  # pyright: ignore[reportPrivateUsage]
)


class _FakeStat:
    def __init__(self, speed: int) -> None:
        self.speed = speed


def test_active_speeds_from_stats_keeps_only_known_speeds() -> None:
    stats = {
        "en0": _FakeStat(speed=1000),
        # 0 is psutil's "couldn't determine this" sentinel - common for
        # Wi-Fi, especially on macOS - and should read as unknown, not 0.
        "en1": _FakeStat(speed=0),
        "lo0": _FakeStat(speed=0),
    }

    assert _active_speeds_from_stats(stats) == {"en0": 1000}


def test_active_speeds_from_stats_empty() -> None:
    assert _active_speeds_from_stats({}) == {}


_IFCONFIG_VERBOSE_SAMPLE = """\
lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
	inet 127.0.0.1 netmask 0xff000000
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
	options=400<CHANNEL_IO>
	ether ac:de:48:00:11:22
	inet 192.168.1.5 netmask 0xffffff00 broadcast 192.168.1.255
	media: autoselect (1000baseT <full-duplex>)
	status: active
	supported media:
		media autoselect
		media 1000baseT mediaopt full-duplex
		media 1000baseT mediaopt half-duplex
		media 100baseTX mediaopt full-duplex
		media 100baseTX mediaopt half-duplex
		media 10baseT/UTP mediaopt full-duplex
		media 10baseT/UTP mediaopt half-duplex
		media none
en5: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
	ether 12:34:56:78:9a:bc
	media: autoselect
	status: inactive
awdl0: flags=8943<UP,BROADCAST,RUNNING,PROMISC,SIMPLEX,MULTICAST> mtu 1484
	ether 22:33:44:55:66:77
"""


def test_parse_supported_media_mbps_takes_the_fastest_recognised_media() -> None:
    speeds = _parse_supported_media_mbps(_IFCONFIG_VERBOSE_SAMPLE)

    assert speeds == {"en0": 1000}


def test_parse_supported_media_mbps_ignores_interfaces_without_the_block() -> None:
    speeds = _parse_supported_media_mbps(_IFCONFIG_VERBOSE_SAMPLE)

    assert "lo0" not in speeds
    assert "en5" not in speeds
    assert "awdl0" not in speeds


def test_parse_supported_media_mbps_empty_input() -> None:
    assert _parse_supported_media_mbps("") == {}


def test_parse_supported_media_mbps_unrecognised_media_type_is_dropped() -> None:
    output = """\
en9: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
	media: autoselect
	supported media:
		media autoselect
		media SomeFutureMediaType mediaopt full-duplex
"""

    assert _parse_supported_media_mbps(output) == {}
