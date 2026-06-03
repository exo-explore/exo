"""exo-cli: management CLI for a running exo cluster.

Usage:
    exo-cli status                          Cluster overview
    exo-cli health                          Quick health check
    exo-cli nodes                           List all nodes
    exo-cli nodes <id>                      Single node detail
    exo-cli models                          Loaded models + downloads
    exo-cli models status <name>            Poll model readiness
    exo-cli models load [--wait] <name>     Load model (auto-placement)
    exo-cli models unload <name>            Unload by name
    exo-cli models swap [--wait] <old> <new>  Atomic model swap

Global flags:
    --host HOST     Cluster host (default: localhost)
    --port PORT     Cluster port (default: 52415)
    --json          Output raw JSON instead of formatted tables
"""

from __future__ import annotations

import argparse
import json
import sys

from exo.cli.client import ExoClient, ExoClientError
from exo.cli.format import (
    format_action_response,
    format_health,
    format_model_status,
    format_models,
    format_node_detail,
    format_nodes,
    format_overview,
)


def _output(data: dict, *, use_json: bool, formatter: object) -> None:
    if use_json:
        print(json.dumps(data, indent=2))
    else:
        print(formatter(data))  # type: ignore[operator]


def _progress_callback(status: dict) -> None:
    prog = status.get("progress") or status.get("status", "waiting...")
    print(f"\r  {prog}", end="", flush=True)


def cmd_status(client: ExoClient, args: argparse.Namespace) -> None:
    data = client.overview()
    _output(data, use_json=args.json, formatter=format_overview)


def cmd_health(client: ExoClient, args: argparse.Namespace) -> None:
    data = client.health()
    _output(data, use_json=args.json, formatter=format_health)
    if not data["healthy"]:
        sys.exit(1)


def cmd_nodes(client: ExoClient, args: argparse.Namespace) -> None:
    if args.node_id:
        data = client.node(args.node_id)
        _output(data, use_json=args.json, formatter=format_node_detail)
    else:
        data = client.nodes()
        _output(data, use_json=args.json, formatter=format_nodes)


def cmd_models(client: ExoClient, args: argparse.Namespace) -> None:
    action = getattr(args, "models_action", None)

    if action == "status":
        data = client.model_status(args.model_name)
        _output(data, use_json=args.json, formatter=format_model_status)

    elif action == "load":
        resp = client.load_model(
            args.model_name,
            min_nodes=args.min_nodes,
            sharding=args.sharding,
        )
        _output(resp, use_json=args.json, formatter=format_action_response)

        if args.wait:
            print("\nWaiting for model to be ready...")
            try:
                status = client.wait_for_model(
                    args.model_name,
                    poll_interval=args.poll_interval,
                    timeout=args.timeout,
                    on_progress=None if args.json else _progress_callback,
                )
                if not args.json:
                    print(f"\n✓ {args.model_name} is ready.")
                else:
                    print(json.dumps(status, indent=2))
            except ExoClientError as exc:
                print(f"\n✗ {exc.detail}", file=sys.stderr)
                sys.exit(1)

    elif action == "unload":
        resp = client.unload_model(args.model_name)
        _output(resp, use_json=args.json, formatter=format_action_response)

    elif action == "swap":
        resp = client.swap_model(
            args.unload_name,
            args.load_name,
            min_nodes=args.min_nodes,
            sharding=args.sharding,
        )
        _output(resp, use_json=args.json, formatter=format_action_response)

        if args.wait:
            print("\nWaiting for new model to be ready...")
            try:
                status = client.wait_for_model(
                    args.load_name,
                    poll_interval=args.poll_interval,
                    timeout=args.timeout,
                    on_progress=None if args.json else _progress_callback,
                )
                if not args.json:
                    print(f"\n✓ {args.load_name} is ready.")
                else:
                    print(json.dumps(status, indent=2))
            except ExoClientError as exc:
                print(f"\n✗ {exc.detail}", file=sys.stderr)
                sys.exit(1)

    else:
        # Default: list models
        data = client.models()
        _output(data, use_json=args.json, formatter=format_models)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exo-cli",
        description="Management CLI for a running exo cluster.",
    )
    parser.add_argument(
        "--host", default="localhost", help="Cluster host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=52415, help="Cluster port (default: 52415)"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output raw JSON"
    )

    sub = parser.add_subparsers(dest="command")

    # status
    sub.add_parser("status", help="Cluster overview")

    # health
    sub.add_parser("health", help="Quick health check (exits 1 if unhealthy)")

    # nodes
    nodes_p = sub.add_parser("nodes", help="List nodes or show node detail")
    nodes_p.add_argument("node_id", nargs="?", help="Node ID for detail view")

    # models
    models_p = sub.add_parser("models", help="Model management")
    models_sub = models_p.add_subparsers(dest="models_action")

    # models status
    ms = models_sub.add_parser("status", help="Check model readiness")
    ms.add_argument("model_name", help="Model name or ID")

    # models load
    ml = models_sub.add_parser("load", help="Load a model")
    ml.add_argument("model_name", help="Model name or HuggingFace ID")
    ml.add_argument("--wait", action="store_true", help="Block until model is ready")
    ml.add_argument("--min-nodes", type=int, default=1, help="Minimum nodes to spread across")
    ml.add_argument("--sharding", choices=["auto", "pipeline", "tensor"], default="auto")
    ml.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between status polls")
    ml.add_argument("--timeout", type=float, default=600.0, help="Max seconds to wait")

    # models unload
    mu = models_sub.add_parser("unload", help="Unload a model")
    mu.add_argument("model_name", help="Model name or ID")

    # models swap
    msw = models_sub.add_parser("swap", help="Swap one model for another")
    msw.add_argument("unload_name", help="Model to unload")
    msw.add_argument("load_name", help="Model to load")
    msw.add_argument("--wait", action="store_true", help="Block until new model is ready")
    msw.add_argument("--min-nodes", type=int, default=1, help="Minimum nodes for new model")
    msw.add_argument("--sharding", choices=["auto", "pipeline", "tensor"], default="auto")
    msw.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between status polls")
    msw.add_argument("--timeout", type=float, default=600.0, help="Max seconds to wait")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    client = ExoClient(host=args.host, port=args.port)

    try:
        handlers = {
            "status": cmd_status,
            "health": cmd_health,
            "nodes": cmd_nodes,
            "models": cmd_models,
        }
        handler = handlers.get(args.command)
        if handler:
            handler(client, args)
        else:
            parser.print_help()
    except ExoClientError as exc:
        print(f"Error: {exc.detail}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
