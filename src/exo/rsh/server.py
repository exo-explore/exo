"""RSH Server - runs on each Exo node to accept remote execution requests."""

import asyncio
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from hypercorn.asyncio import serve
from hypercorn.config import Config
from loguru import logger
from pydantic import BaseModel

RSH_PORT = 52416


class ExecuteRequest(BaseModel):
    """Request to execute a command."""

    command: list[str]
    cwd: Optional[str] = None
    env: Optional[dict[str, str]] = None


class ExecuteResponse(BaseModel):
    """Response from command execution."""

    exit_code: int
    stdout: str
    stderr: str


def create_rsh_app() -> FastAPI:
    """Create the RSH FastAPI application."""
    app = FastAPI(title="Exo RSH Server")

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}

    @app.post("/execute")
    async def execute(request: ExecuteRequest) -> ExecuteResponse:
        """Execute a command and return the result."""
        cmd_str = " ".join(request.command)
        logger.info(f"RSH executing: {cmd_str}")

        try:
            # Build environment
            import os

            env = os.environ.copy()
            if request.env:
                env.update(request.env)

            # Check if command contains shell metacharacters (semicolons, pipes, etc.)
            # If so, run through shell. mpirun sends complex commands like:
            # "VAR=value;export VAR;/path/to/prted --args"
            needs_shell = any(c in cmd_str for c in ";|&$`")

            if needs_shell:
                # Run through shell
                process = await asyncio.create_subprocess_shell(
                    cmd_str,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=request.cwd,
                    env=env,
                )
            else:
                # Execute directly
                process = await asyncio.create_subprocess_exec(
                    *request.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=request.cwd,
                    env=env,
                )

            stdout, stderr = await process.communicate()
            exit_code = process.returncode or 0

            logger.info(f"RSH command completed with exit code {exit_code}")

            return ExecuteResponse(
                exit_code=exit_code,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
            )

        except FileNotFoundError as e:
            logger.error(f"RSH command not found: {e}")
            return ExecuteResponse(
                exit_code=127,
                stdout="",
                stderr=f"Command not found: {request.command[0]}",
            )
        except Exception as e:
            logger.error(f"RSH execution error: {e}")
            return ExecuteResponse(
                exit_code=1,
                stdout="",
                stderr=str(e),
            )

    @app.post("/execute_streaming")
    async def execute_streaming(request: ExecuteRequest):
        """Execute a command and stream the output."""
        logger.info(f"RSH streaming execute: {' '.join(request.command)}")

        async def stream_output():
            try:
                env = None
                if request.env:
                    import os

                    env = os.environ.copy()
                    env.update(request.env)

                process = await asyncio.create_subprocess_exec(
                    *request.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=request.cwd,
                    env=env,
                )

                if process.stdout:
                    async for line in process.stdout:
                        yield line

                await process.wait()

            except Exception as e:
                yield f"Error: {e}\n".encode()

        return StreamingResponse(
            stream_output(),
            media_type="application/octet-stream",
        )

    return app


async def run_rsh_server(port: int = RSH_PORT):
    """Run the RSH server."""
    import anyio

    app = create_rsh_app()
    config = Config()
    config.bind = [f"0.0.0.0:{port}"]
    config.accesslog = None  # Disable access logs for cleaner output

    logger.info(f"Starting RSH server on port {port}")
    await serve(app, config, shutdown_trigger=lambda: anyio.sleep_forever())  # type: ignore
