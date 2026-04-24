import argparse
import logging
from contextlib import asynccontextmanager
from typing import cast

import httpx
import uvicorn
from fastapi import FastAPI

from exo.reasoning_proxy.cache import ReasoningCache
from exo.reasoning_proxy.registry import DialectRegistry
from exo.reasoning_proxy.routes import register_routes

logger = logging.getLogger("exo.reasoning_proxy")


def build_app(upstream: str) -> FastAPI:
    client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0))
    cache = ReasoningCache()
    registry = DialectRegistry(upstream=upstream, client=client)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await registry.refresh()
        try:
            yield
        finally:
            await client.aclose()

    app = FastAPI(lifespan=lifespan, title="exo-reasoning-proxy")
    register_routes(
        app=app,
        client=client,
        upstream=upstream.rstrip("/"),
        cache=cache,
        registry=registry,
    )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(prog="exo-reasoning-proxy")
    _ = parser.add_argument("--upstream", default="http://localhost:52415")
    _ = parser.add_argument("--host", default="127.0.0.1")
    _ = parser.add_argument("--port", type=int, default=52416)
    _ = parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    verbose = cast(int, args.verbose)
    upstream = cast(str, args.upstream)
    host = cast(str, args.host)
    port = cast(int, args.port)

    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    logger.info("Starting exo-reasoning-proxy on %s:%d → %s", host, port, upstream)

    app = build_app(upstream=upstream)
    uvicorn.run(app, host=host, port=port, log_level=level)


if __name__ == "__main__":
    main()
