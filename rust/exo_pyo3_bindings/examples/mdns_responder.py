from exo_pyo3_bindings import RustNetworkingHandle, Keypair
from asyncio import run

async def main():
    nh = await RustNetworkingHandle.create(Keypair.generate_ed25519(), "mdns_example")
    recv = await nh.get_connection_receiver()
    while True:
        print(await recv.receive())


if __name__ == "__main__":
    run(main())
