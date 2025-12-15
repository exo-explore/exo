from exo_pyo3_bindings import RustNetworkingHandle, Keypair
from asyncio import run


async def main():
    nh = await RustNetworkingHandle.create(Keypair.generate_ed25519(), "mdns_example")
    recv = await nh.get_connection_receiver()
    while True:
        cm = await recv.receive()
        print(
            f"Endpoint({cm.endpoint_id}, reachable={list(map(lambda it: it.ip_addr(), cm.current_transport_addrs)) if cm.current_transport_addrs is not None else None})"
        )


if __name__ == "__main__":
    run(main())
