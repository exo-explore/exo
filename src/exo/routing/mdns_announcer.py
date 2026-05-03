import argparse
import contextlib
import random
import socket
import string
import struct
import sys
import time
from typing import final


def _dns_qname(name: bytes) -> bytes:
    return b"".join(bytes([len(part)]) + part for part in name.split(b".")) + b"\0"


def _build_response_packet(node_id: str, ip_address: str, libp2p_port: int) -> bytes:
    service_name = b"_p2p._udp.local"
    peer_name = (
        "".join(random.choice(string.ascii_letters + string.digits) for _ in range(32))
        + "._p2p._udp.local"
    ).encode()
    txt_record = f"dnsaddr=/ip4/{ip_address}/tcp/{libp2p_port}/p2p/{node_id}".encode()

    peer_qname = _dns_qname(peer_name)
    packet = bytearray()
    packet += struct.pack("!HHHHHH", 0, 0x8400, 0, 1, 0, 1)
    packet += _dns_qname(service_name)
    packet += struct.pack("!HHI", 12, 1, 120)
    packet += struct.pack("!H", len(peer_qname))
    packet += peer_qname
    packet += peer_qname
    packet += struct.pack("!HHI", 16, 1, 120)
    packet += struct.pack("!H", len(txt_record) + 1)
    packet += bytes([len(txt_record)])
    packet += txt_record
    return bytes(packet)


@final
class Args(argparse.Namespace):
    node_id: str
    ip_address: str
    libp2p_port: int
    broadcast_address: str | None
    count: int

    @staticmethod
    def parse() -> "Args":
        parser = argparse.ArgumentParser()
        parser.add_argument("--node-id", required=True)
        parser.add_argument("--ip-address", required=True)
        parser.add_argument("--libp2p-port", required=True, type=int)
        parser.add_argument("--broadcast-address")
        parser.add_argument("--count", default=0, type=int)
        return parser.parse_args(namespace=Args())


def main() -> None:
    args = Args.parse()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    with contextlib.suppress(OSError):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((args.ip_address, 0))

    sent_count = 0
    while True:
        packet = _build_response_packet(
            args.node_id, args.ip_address, args.libp2p_port
        )
        errors: list[str] = []
        destinations: list[tuple[str, int]] = []
        if args.broadcast_address is not None:
            destinations.append((args.broadcast_address, 5353))
        destinations.extend([("255.255.255.255", 5353), ("224.0.0.251", 5353)])
        sent = False
        for destination in destinations:
            try:
                sock.sendto(packet, destination)
                sent = True
            except OSError as err:
                errors.append(f"{destination}: {err}")
        if not sent:
            print(
                f"mDNS announcer send failed: {'; '.join(errors)}",
                file=sys.stderr,
                flush=True,
            )
        sent_count += 1
        if args.count > 0 and sent_count >= args.count:
            return
        time.sleep(1.0 if sent_count < 60 else 10.0)


if __name__ == "__main__":
    main()
