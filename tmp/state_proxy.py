import anyio
from exo_net import StateProxy


async def main():
    sp = await StateProxy.init()
    while True:
        data = await sp.snapshot()
        if data != "{}":
            print(data)
        await anyio.sleep(1)


if __name__ == "__main__":
    anyio.run(main)
