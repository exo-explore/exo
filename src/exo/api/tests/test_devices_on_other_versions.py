from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.routing.wire import WIRE_VERSION, DeviceOnOtherVersion, build_id


def _client(devices: list[DeviceOnOtherVersion]) -> TestClient:
    api = object.__new__(API)
    api.app = FastAPI()
    api._devices_on_other_versions = lambda: devices  # pyright: ignore[reportPrivateUsage]
    api.app.get("/v1/devices-on-other-versions")(api.get_devices_on_other_versions)
    return TestClient(api.app)


def test_lists_each_device_that_can_not_join_once() -> None:
    response = _client(
        [
            DeviceOnOtherVersion(
                addresses=["fe80::1", "fe80::2"],
                reason="wire_version",
                wire_version=WIRE_VERSION + 1,
            ),
            DeviceOnOtherVersion(
                addresses=["fe80::3"], reason="build", wire_version=None
            ),
            DeviceOnOtherVersion(
                addresses=["fe80::4"], reason="older_exo", wire_version=None
            ),
        ]
    ).get("/v1/devices-on-other-versions")
    assert response.status_code == 200
    assert response.json() == {
        "wireVersion": WIRE_VERSION,
        "buildId": build_id(),
        "devices": [
            # One device, heard on two of its interfaces
            {
                "addresses": ["fe80::1", "fe80::2"],
                "reason": "wire_version",
                "wireVersion": WIRE_VERSION + 1,
            },
            {"addresses": ["fe80::3"], "reason": "build", "wireVersion": None},
            {"addresses": ["fe80::4"], "reason": "older_exo", "wireVersion": None},
        ],
    }


def test_is_empty_when_every_device_can_join() -> None:
    response = _client([]).get("/v1/devices-on-other-versions")
    assert response.json() == {
        "wireVersion": WIRE_VERSION,
        "buildId": build_id(),
        "devices": [],
    }
