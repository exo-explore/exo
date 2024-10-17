# Note: this test is for MacOS
# The results are not very good between 2 MacBook Pro's:
# INFO:__main__:Test 1: Write: 60.72 ms, Read: 59.08 ms, Total RTT: 119.80 ms, Timestamp: 1713, Time diff: 0 ms

import asyncio
import time
import struct
import argparse
import logging
from typing import Any

# For the server
from bless import BlessServer
from bless.backends.characteristic import GATTCharacteristicProperties, GATTAttributePermissions

# For the client
from bleak import BleakClient, BleakScanner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVICE_UUID = "A07498CA-AD5B-474E-940D-16F1FBE7E8CD"
CHAR_UUID = "51FF12BB-3ED8-46E5-B4F9-D64E2FEC021B"
CONN_PARAMS_SERVICE_UUID = "1234A00C-0000-1000-8000-00805F9B34FB"
CONN_PARAMS_CHAR_UUID = "1234A00D-0000-1000-8000-00805F9B34FB"

# Define the desired connection interval (7.5ms)
CONN_INTERVAL_MIN = 6  # 7.5ms in 1.25ms units
CONN_INTERVAL_MAX = 6
CONN_LATENCY = 0
SUPERVISION_TIMEOUT = 100  # 1 second

def read_request(characteristic):
    return characteristic.value

def write_request(characteristic, value):
    characteristic.value = value
    if value == b"ping":
        characteristic.value = b"pong"

async def run_server(loop):
    server = BlessServer(name="Latency Test Server", loop=loop)
    server.read_request_func = read_request
    server.write_request_func = write_request

    await server.add_new_service(SERVICE_UUID)

    # Main characteristic for ping-pong (read and write)
    char_flags = GATTCharacteristicProperties.read | GATTCharacteristicProperties.write
    permissions = GATTAttributePermissions.readable | GATTAttributePermissions.writeable
    await server.add_new_characteristic(
        SERVICE_UUID, CHAR_UUID, char_flags, None, permissions
    )

    # Add new service and characteristic for connection parameters (read-only)
    await server.add_new_service(CONN_PARAMS_SERVICE_UUID)
    conn_params = struct.pack("<HHHH", CONN_INTERVAL_MIN, CONN_INTERVAL_MAX, CONN_LATENCY, SUPERVISION_TIMEOUT)
    conn_params_flags = GATTCharacteristicProperties.read
    conn_params_permissions = GATTAttributePermissions.readable
    await server.add_new_characteristic(
        CONN_PARAMS_SERVICE_UUID, CONN_PARAMS_CHAR_UUID, conn_params_flags, conn_params, conn_params_permissions
    )

    await server.start()
    logger.info("Server started. Use the UUID of this device when running the client.")

    await asyncio.Event().wait()  # Run forever

async def run_client(server_uuid):
    logger.info(f"Connecting to server with UUID: {server_uuid}")
    async with BleakClient(server_uuid) as client:
        logger.info("Connected")

        # Read connection parameters
        try:
            conn_params = await client.read_gatt_char(CONN_PARAMS_CHAR_UUID)
            interval_min, interval_max, latency, timeout = struct.unpack("<HHHH", conn_params)
            logger.info(f"Connection parameters: Interval min: {interval_min * 1.25}ms, "
                        f"Interval max: {interval_max * 1.25}ms, Latency: {latency}, "
                        f"Timeout: {timeout * 10}ms")
        except Exception as e:
            logger.warning(f"Failed to read connection parameters: {e}")

        # Proceed with latency test
        num_tests = 50
        rtts = []
        last_timestamp = 0

        for i in range(num_tests):
            start_time = time.perf_counter()

            # Write operation
            await client.write_gatt_char(CHAR_UUID, b"ping")
            write_time = time.perf_counter()

            # Read operation
            response = await client.read_gatt_char(CHAR_UUID)
            end_time = time.perf_counter()

            write_latency = (write_time - start_time) * 1000
            read_latency = (end_time - write_time) * 1000
            total_rtt = (end_time - start_time) * 1000

            # Calculate timestamp (13-bit millisecond resolution as per BLE-MIDI spec)
            timestamp = int((start_time * 1000) % 8192)

            # Calculate time difference from last timestamp
            if last_timestamp:
                time_diff = (timestamp - last_timestamp) % 8192
            else:
                time_diff = 0

            last_timestamp = timestamp

            rtts.append(total_rtt)
            logger.info(f"Test {i+1}: Write: {write_latency:.2f} ms, Read: {read_latency:.2f} ms, "
                        f"Total RTT: {total_rtt:.2f} ms, Timestamp: {timestamp}, Time diff: {time_diff} ms")

            await asyncio.sleep(0.01)  # Small delay between tests

        average_rtt = sum(rtts) / num_tests
        median_rtt = sorted(rtts)[num_tests // 2]
        min_rtt = min(rtts)
        max_rtt = max(rtts)
        logger.info(f"\nAverage RTT: {average_rtt:.2f} ms")
        logger.info(f"Median RTT: {median_rtt:.2f} ms")
        logger.info(f"Min RTT: {min_rtt:.2f} ms")
        logger.info(f"Max RTT: {max_rtt:.2f} ms")

async def discover_devices():
    logger.info("Scanning for BLE devices...")
    devices = await BleakScanner.discover()
    for d in devices:
        logger.info(f"Found device: {d.name} (UUID: {d.address})")
    return devices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLE Latency Test")
    parser.add_argument("mode", choices=["server", "client", "scan"], help="Run as server, client, or scan for devices")
    parser.add_argument("--uuid", help="Server's UUID (required for client mode)")
    args = parser.parse_args()

    if args.mode == "server":
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_server(loop))
    elif args.mode == "client":
        if not args.uuid:
            logger.error("Error: Server UUID is required for client mode.")
            exit(1)
        asyncio.run(run_client(args.uuid))
    elif args.mode == "scan":
        asyncio.run(discover_devices())