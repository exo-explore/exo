#!/usr/bin/env python3
import argparse
import socket
import sys


def main() -> int:
    p = argparse.ArgumentParser(
        description="IPv6 UDP client with optional explicit source bind"
    )
    p.add_argument("--dest", required=True, help="Destination IPv6 address")
    p.add_argument("--port", type=int, default=45679, help="Destination UDP port")
    p.add_argument("--source", help="Optional source IPv6 address to bind to")
    p.add_argument("--message", default="hello", help="Payload to send")
    p.add_argument(
        "--timeout", type=float, default=5.0, help="Receive timeout in seconds"
    )
    args = p.parse_args()

    s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
    s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
    s.settimeout(args.timeout)

    if args.source:
        s.bind((args.source, 0, 0, 0))

    print(f"local-before-send={s.getsockname()}")
    s.sendto(args.message.encode(), (args.dest, args.port, 0, 0))
    print(f"sent to=[{args.dest}]:{args.port}")
    print(f"local-after-send={s.getsockname()}")

    data, peer = s.recvfrom(65535)
    print(f"from={peer} data={data!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
