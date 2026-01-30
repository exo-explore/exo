import time
from typing import cast

import anyio
import httpx
from exo.shared.logging import logger
from pydantic.v1 import BaseModel

LATENCY_PING_COUNT = 5
BANDWIDTH_TEST_DURATION_S = 0.5
UPLOAD_BUFFER_SIZE = 256 * 1024 * 1024


class ConnectionProfile(BaseModel):
    latency_ms: float
    upload_mbps: float
    download_mbps: float


async def measure_latency(target_ip: str, port: int = 52415) -> float:
    if ":" in target_ip:
        url = f"http://[{target_ip}]:{port}/node_id"
    else:
        url = f"http://{target_ip}:{port}/node_id"

    rtts: list[float] = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        for _ in range(LATENCY_PING_COUNT):
            try:
                start = time.perf_counter()
                response = await client.get(url)
                end = time.perf_counter()

                if response.status_code == 200:
                    rtts.append((end - start) * 1000)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
                logger.debug(f"Latency ping failed: {e}")

    if not rtts:
        raise ConnectionError(f"Failed to measure latency to {target_ip}:{port}")

    return sum(rtts) / len(rtts)


async def measure_bandwidth(target_ip: str, port: int = 52415) -> tuple[float, float]:
    if ":" in target_ip:
        base_url = f"http://[{target_ip}]:{port}"
    else:
        base_url = f"http://{target_ip}:{port}"

    upload_url = f"{base_url}/bandwidth_test/upload"
    download_url = f"{base_url}/bandwidth_test/download"

    upload_mbps = 0.0
    download_mbps = 0.0

    upload_buffer = b"X" * UPLOAD_BUFFER_SIZE

    timeout = httpx.Timeout(timeout=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            with anyio.fail_after(10.0):
                response = await client.post(upload_url, content=upload_buffer)
                if response.status_code == 200:
                    data = cast(dict[str, float], response.json())
                    bytes_received = data["bytes_received"]
                    duration = data["duration_s"]
                    if duration > 0:
                        upload_mbps = (bytes_received * 8 / duration) / 1_000_000
                        logger.debug(
                            f"Upload: {bytes_received / 1_000_000:.1f}MB in "
                            f"{duration:.3f}s = {upload_mbps:.1f} Mbps"
                        )
        except (TimeoutError, httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
            logger.debug(f"Upload test failed: {e}")

        try:
            bytes_downloaded = 0
            start = time.perf_counter()

            with anyio.move_on_after(BANDWIDTH_TEST_DURATION_S):
                async with client.stream("GET", download_url) as response:
                    if response.status_code == 200:
                        async for chunk in response.aiter_bytes():
                            bytes_downloaded += len(chunk)

            duration = time.perf_counter() - start
            if duration > 0 and bytes_downloaded > 0:
                download_mbps = (bytes_downloaded * 8 / duration) / 1_000_000
                logger.debug(
                    f"Download: {bytes_downloaded / 1_000_000:.1f}MB in "
                    f"{duration:.3f}s = {download_mbps:.1f} Mbps"
                )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
            logger.debug(f"Download test failed: {e}")

    if upload_mbps == 0.0 and download_mbps == 0.0:
        raise ConnectionError(f"Failed to measure bandwidth to {target_ip}:{port}")

    return upload_mbps, download_mbps


async def profile_connection(target_ip: str, port: int = 52415) -> ConnectionProfile:
    logger.debug(f"Profiling connection to {target_ip}:{port}")

    latency_ms = await measure_latency(target_ip, port)
    logger.debug(f"Measured latency to {target_ip}: {latency_ms:.2f}ms")

    upload_mbps, download_mbps = await measure_bandwidth(target_ip, port)
    logger.debug(
        f"Measured bandwidth to {target_ip}: "
        f"upload={upload_mbps:.1f}Mbps, download={download_mbps:.1f}Mbps"
    )

    return ConnectionProfile(
        latency_ms=latency_ms,
        upload_mbps=upload_mbps,
        download_mbps=download_mbps,
    )
