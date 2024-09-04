import asyncio
import time
import argparse
import logging
from typing import Any

# For the server
from bless import (
    BlessServer,
    BlessGATTCharacteristic,
    GATTCharacteristicProperties,
    GATTAttributePermissions
)

# For the client
from bleak import BleakClient, BleakScanner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVICE_UUID = "A07498CA-AD5B-474E-940D-16F1FBE7E8CD"
CHAR_UUID = "51FF12BB-3ED8-46E5-B4F9-D64E2FEC021B"

def read_request(characteristic: BlessGATTCharacteristic, **kwargs) -> bytearray:
    return characteristic.value

def write_request(characteristic: BlessGATTCharacteristic, value: Any, **kwargs):
    characteristic.value = value
    if value == b"ping":
        characteristic.value = b"pong"

async def run_server(loop):
    server = BlessServer(name="Latency Test Server", loop=loop)
    server.read_request_func = read_request
    server.write_request_func = write_request

    await server.add_new_service(SERVICE_UUID)
    char_flags = (
        GATTCharacteristicProperties.read
        | GATTCharacteristicProperties.write
        | GATTCharacteristicProperties.indicate
    )
    permissions = GATTAttributePermissions.readable | GATTAttributePermissions.writeable
    await server.add_new_characteristic(
        SERVICE_UUID, CHAR_UUID, char_flags, None, permissions
    )

    await server.start()
    logger.info("Server started. Use the UUID of this device when running the client.")
    await asyncio.Event().wait()  # Run forever

async def run_client(server_uuid):
    logger.info(f"Connecting to server with UUID: {server_uuid}")
    async with BleakClient(server_uuid) as client:
        logger.info("Connected")

        # Explore services and characteristics
        for service in client.services:
            logger.info(f"Service: {service.uuid}")
            for char in service.characteristics:
                logger.info(f"  Characteristic: {char.uuid}")
                logger.info(f"    Properties: {', '.join(char.properties)}")
                try:
                    value = await client.read_gatt_char(char.uuid)
                    logger.info(f"    Value: {value}")
                except Exception as e:
                    logger.info(f"    Could not read value: {e}")

        # Proceed with latency test
        num_tests = 50
        total_rtt = 0
        rtts = []

        for i in range(num_tests):
            start_write = time.perf_counter()
            await client.write_gatt_char(CHAR_UUID, b"ping")
            end_write = time.perf_counter()

            start_read = time.perf_counter()
            response = await client.read_gatt_char(CHAR_UUID)
            end_read = time.perf_counter()

            write_time = (end_write - start_write) * 1000
            read_time = (end_read - start_read) * 1000
            total_time = (end_read - start_write) * 1000

            rtts.append(total_time)
            logger.info(f"Test {i+1}: Write: {write_time:.2f} ms, Read: {read_time:.2f} ms, Total: {total_time:.2f} ms")

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