import time

import anyio
import httpx
from anyio.abc import SocketStream
from exo.shared.logging import logger
from pydantic.v1 import BaseModel

LATENCY_PING_COUNT = 5
BANDWIDTH_TEST_DURATION_S = 0.5
BANDWIDTH_TEST_PORT_OFFSET = 1  # API port + 1


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


async def _measure_upload_tcp(stream: SocketStream, duration: float) -> float:
    """Send data for duration seconds, return Mbps."""
    chunk = b"X" * (1024 * 1024)  # 1MB
    bytes_sent = 0
    start = time.perf_counter()
    deadline = start + duration

    while time.perf_counter() < deadline:
        await stream.send(chunk)
        bytes_sent += len(chunk)

    elapsed = time.perf_counter() - start
    return (bytes_sent * 8 / elapsed) / 1_000_000 if elapsed > 0 else 0.0


async def _measure_download_tcp(stream: SocketStream, duration: float) -> float:
    """Receive data for duration seconds, return Mbps."""
    bytes_received = 0
    start = time.perf_counter()

    with anyio.move_on_after(duration):
        while True:
            data = await stream.receive(1024 * 1024)
            if not data:
                break
            bytes_received += len(data)

    elapsed = time.perf_counter() - start
    return (bytes_received * 8 / elapsed) / 1_000_000 if elapsed > 0 else 0.0


async def measure_bandwidth_tcp(target_ip: str, port: int) -> tuple[float, float]:
    """Measure bandwidth using raw TCP like iperf."""
    upload_mbps = 0.0
    download_mbps = 0.0

    try:
        async with await anyio.connect_tcp(target_ip, port) as stream:
            # Protocol: send 'U' for upload test, 'D' for download test
            # Upload: client sends, server receives
            await stream.send(b"U")
            upload_mbps = await _measure_upload_tcp(stream, BANDWIDTH_TEST_DURATION_S)
            await stream.send(b"DONE")
            logger.debug(f"Upload: {upload_mbps:.1f} Mbps")
    except Exception as e:
        logger.debug(f"Upload TCP test failed: {e}")

    try:
        async with await anyio.connect_tcp(target_ip, port) as stream:
            # Download: client receives, server sends
            await stream.send(b"D")
            download_mbps = await _measure_download_tcp(stream, BANDWIDTH_TEST_DURATION_S)
            logger.debug(f"Download: {download_mbps:.1f} Mbps")
    except Exception as e:
        logger.debug(f"Download TCP test failed: {e}")

    return upload_mbps, download_mbps


async def profile_connection(target_ip: str, port: int = 52415) -> ConnectionProfile:
    logger.debug(f"Profiling connection to {target_ip}:{port}")

    latency_ms = await measure_latency(target_ip, port)
    logger.debug(f"Measured latency to {target_ip}: {latency_ms:.2f}ms")

    bandwidth_port = port + BANDWIDTH_TEST_PORT_OFFSET
    upload_mbps, download_mbps = await measure_bandwidth_tcp(target_ip, bandwidth_port)
    logger.debug(
        f"Measured bandwidth to {target_ip}: "
        f"upload={upload_mbps:.1f}Mbps, download={download_mbps:.1f}Mbps"
    )

    if upload_mbps == 0.0 and download_mbps == 0.0:
        raise ConnectionError(f"Failed to measure bandwidth to {target_ip}:{bandwidth_port}")

    return ConnectionProfile(
        latency_ms=latency_ms,
        upload_mbps=upload_mbps,
        download_mbps=download_mbps,
    )
