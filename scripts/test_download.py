from worker.download.download_utils import *

async def main():
    meta = await file_meta(
        'mlx-community/DeepSeek-R1-4bit',
        revision='main',
        path='config.json',
        redirected_location=None,
    )
    print(meta)

asyncio.run(main())