#!/usr/bin/env python3
"""
Exo P2P Bridge - Port Forwarder for Static Peer Discovery

This script creates FIXED port listeners that forward to Exo's DYNAMIC ports,
enabling static peer configuration without modifying Exo's core code.

How it works:
1. Monitors Exo logs to discover actual listening ports
2. Creates fixed port listeners (7000, 7001, 7002)
3. Forwards TCP connections to dynamic Exo ports
4. Updates in real-time when Exo restarts and ports change

Usage:
  python3 exo_p2p_bridge.py --fixed-port 7000 --log-file ~/.cache/exo/exo.log
"""

import asyncio
import re
import sys
import argparse
import signal
from pathlib import Path

class P2PBridge:
    def __init__(self, fixed_port: int, log_file: Path, check_interval: int = 5):
        self.fixed_port = fixed_port
        self.log_file = log_file
        self.check_interval = check_interval
        self.current_target_port = None
        self.server = None
        self.active_connections = []
        self.running = True

    async def monitor_log_for_port(self):
        """Monitor Exo log file to detect actual listening port"""
        print(f"[Bridge] Monitoring {self.log_file} for port changes...")

        # Pattern to match ONLY local listen addresses, not peer discovery
        # NewListenAddr { listener_id: ListenerId(1), address: /ip4/192.168.0.X/tcp/PORT }
        pattern = re.compile(r'NewListenAddr.*?/ip4/192\.168\.0\.\d+/tcp/(\d+)')

        last_size = 0

        # Scan entire log on first startup to find current port
        if self.log_file.exists():
            print(f"[Bridge] Scanning existing log for current port...")
            try:
                with open(self.log_file, 'r') as f:
                    content = f.read()
                    matches = pattern.findall(content)
                    if matches:
                        # Take the last match (most recent port)
                        current_port = int(matches[-1])
                        print(f"[Bridge] Found current Exo port in log: {current_port}")
                        await self.update_target_port(current_port)
                    else:
                        print(f"[Bridge] No port found in existing log, waiting for new announcements...")
            except Exception as e:
                print(f"[Bridge] Error scanning existing log: {e}")

            last_size = self.log_file.stat().st_size

        while self.running:
            try:
                if self.log_file.exists():
                    current_size = self.log_file.stat().st_size

                    # Only read new content
                    if current_size > last_size:
                        with open(self.log_file, 'r') as f:
                            f.seek(last_size)
                            new_content = f.read()

                            # Find all port matches in new content
                            matches = pattern.findall(new_content)
                            if matches:
                                # Take the last match (most recent port)
                                new_port = int(matches[-1])

                                if new_port != self.current_target_port:
                                    print(f"[Bridge] Detected new Exo port: {new_port}")
                                    await self.update_target_port(new_port)

                        last_size = current_size

            except Exception as e:
                print(f"[Bridge] Error monitoring log: {e}")

            await asyncio.sleep(self.check_interval)

    async def update_target_port(self, new_port: int):
        """Update target port and restart server if needed"""
        old_port = self.current_target_port
        self.current_target_port = new_port

        if old_port is None:
            # First time discovering port - start server
            print(f"[Bridge] Starting bridge: {self.fixed_port} -> {new_port}")
            await self.start_server()
        else:
            # Port changed - existing connections will gracefully fail and reconnect
            print(f"[Bridge] Port updated: {self.fixed_port} -> {new_port} (was {old_port})")

    async def start_server(self):
        """Start TCP server on fixed port"""
        try:
            self.server = await asyncio.start_server(
                self.handle_connection,
                '0.0.0.0',
                self.fixed_port
            )

            addr = self.server.sockets[0].getsockname()
            print(f"[Bridge] Listening on {addr[0]}:{addr[1]}")
            print(f"[Bridge] Forwarding to localhost:{self.current_target_port}")

        except Exception as e:
            print(f"[Bridge] Failed to start server: {e}")
            sys.exit(1)

    async def handle_connection(self, reader, writer):
        """Handle incoming connection and forward to Exo"""
        client_addr = writer.get_extra_info('peername')

        if self.current_target_port is None:
            print(f"[Bridge] Connection from {client_addr} - waiting for target port...")
            writer.close()
            await writer.wait_closed()
            return

        print(f"[Bridge] Connection from {client_addr} -> localhost:{self.current_target_port}")

        try:
            # Connect to actual Exo port
            target_reader, target_writer = await asyncio.open_connection(
                '127.0.0.1',
                self.current_target_port
            )

            # Bidirectional forwarding
            async def forward(src, dst, direction):
                try:
                    while True:
                        data = await src.read(8192)
                        if not data:
                            break
                        dst.write(data)
                        await dst.drain()
                except Exception as e:
                    print(f"[Bridge] {direction} error: {e}")
                finally:
                    try:
                        dst.close()
                        await dst.wait_closed()
                    except:
                        pass

            # Run both directions concurrently
            await asyncio.gather(
                forward(reader, target_writer, "client->exo"),
                forward(target_reader, writer, "exo->client")
            )

        except Exception as e:
            print(f"[Bridge] Forward error from {client_addr}: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass

    async def run(self):
        """Main run loop"""
        print(f"[Bridge] Exo P2P Bridge starting...")
        print(f"[Bridge] Fixed port: {self.fixed_port}")
        print(f"[Bridge] Log file: {self.log_file}")

        # Start monitoring in background
        monitor_task = asyncio.create_task(self.monitor_log_for_port())

        try:
            # Keep running until interrupted
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print(f"\n[Bridge] Shutting down...")
        finally:
            self.running = False
            monitor_task.cancel()

            if self.server:
                self.server.close()
                await self.server.wait_closed()

            print(f"[Bridge] Stopped")

async def main():
    parser = argparse.ArgumentParser(description='Exo P2P Bridge - Fixed port forwarder')
    parser.add_argument('--fixed-port', type=int, required=True,
                        help='Fixed port to listen on (e.g., 7000)')
    parser.add_argument('--log-file', type=str, required=True,
                        help='Path to Exo log file (e.g., ~/.cache/exo/exo.log)')
    parser.add_argument('--check-interval', type=int, default=5,
                        help='Log check interval in seconds (default: 5)')

    args = parser.parse_args()

    log_file = Path(args.log_file).expanduser()
    if not log_file.exists():
        print(f"[Bridge] Warning: Log file {log_file} does not exist yet")
        print(f"[Bridge] Will wait for it to be created...")

    bridge = P2PBridge(args.fixed_port, log_file, args.check_interval)

    # Handle graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(bridge)))

    await bridge.run()

async def shutdown(bridge):
    """Graceful shutdown handler"""
    bridge.running = False

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
