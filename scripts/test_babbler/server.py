#!/usr/bin/env python3
import argparse
import socket
import sys


def main() -> int:
    p = argparse.ArgumentParser(
        description="IPv6 UDP server bound to a specific local address"
    )
    p.add_argument(
        "--bind", required=True, help="Local IPv6 address to bind to, e.g. fde0:..."
    )
    p.add_argument("--port", type=int, default=45679, help="UDP port to listen on")
    p.add_argument("--reply", default="ok", help="Reply prefix")
    args = p.parse_args()

    s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
    s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
    s.bind((args.bind, args.port, 0, 0))

    print(f"listening on [{args.bind}]:{args.port}")
    print(f"sockname={s.getsockname()}")

    data, peer = s.recvfrom(65535)
    print(f"from={peer} data={data!r}")

    out = args.reply.encode() + b":" + data
    s.sendto(out, peer)
    print(f"sent={out!r} to={peer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
