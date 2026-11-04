"""Human-friendly output formatting for exo-cli."""

from __future__ import annotations

from typing import Any


def _bar(pct: float, width: int = 20) -> str:
    filled = int(pct / 100 * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {pct:.1f}%"


def format_health(data: dict[str, Any]) -> str:
    status = "✓ healthy" if data["healthy"] else "✗ unhealthy"
    return (
        f"Cluster: {status}\n"
        f"Nodes:   {data['node_count']}\n"
        f"Master:  {data['master_node_id']}"
    )


def format_overview(data: dict[str, Any]) -> str:
    lines = [
        f"Cluster: {data['node_count']} nodes, "
        f"{data['loaded_model_count']} model(s) loaded, "
        f"{data['active_task_count']} active task(s)",
        f"Memory:  {data['used_ram_gb']:.1f} / {data['total_ram_gb']:.1f} GB "
        f"{_bar(data['ram_used_percent'])}",
        "",
    ]

    if data["nodes"]:
        lines.append("NODES")
        lines.append(f"{'Name':<20} {'Chip':<20} {'RAM':>12} {'GPU':>6} {'Status':<8}")
        lines.append("─" * 70)
        for n in data["nodes"]:
            ram = f"{n['ram_used_gb']:.0f}/{n['ram_total_gb']:.0f}GB"
            gpu = f"{n['gpu_usage_percent']:.0f}%" if n["gpu_usage_percent"] > 0 else "—"
            lines.append(
                f"{n['friendly_name']:<20} {n['chip']:<20} {ram:>12} {gpu:>6} {n['status']:<8}"
            )
        lines.append("")

    if data["models"]:
        lines.append("MODELS")
        lines.append(f"{'Model':<35} {'Nodes':<25} {'Size':>8} {'Status':<10}")
        lines.append("─" * 80)
        for m in data["models"]:
            nodes_str = ", ".join(m["node_names"])
            size = f"{m['storage_size_gb']:.0f}GB"
            lines.append(
                f"{m['model_name']:<35} {nodes_str:<25} {size:>8} {m['status']:<10}"
            )
        lines.append("")

    if data.get("downloads"):
        active = [d for d in data["downloads"] if d["status"] in ("pending", "downloading")]
        if active:
            lines.append("DOWNLOADS")
            for d in active:
                lines.append(
                    f"  {d['model_id']} → {d['node_name']}: "
                    f"{d['progress_percent']:.1f}% "
                    f"({d['downloaded_gb']:.1f}/{d['total_gb']:.1f}GB)"
                )
            lines.append("")

    return "\n".join(lines)


def format_nodes(data: dict[str, Any]) -> str:
    lines = [
        f"{data['node_count']} node(s)",
        "",
        f"{'Name':<20} {'Chip':<20} {'RAM Used':>12} {'Disk Free':>12} {'GPU':>6} {'Temp':>6} {'Power':>7} {'Status':<8}",
        "─" * 95,
    ]
    for n in data["nodes"]:
        ram = f"{n['ram_used_gb']:.0f}/{n['ram_total_gb']:.0f}GB"
        disk = f"{n['disk_available_gb']:.0f}GB"
        gpu = f"{n['gpu_usage_percent']:.0f}%" if n["gpu_usage_percent"] > 0 else "—"
        temp = f"{n['temperature_c']:.0f}°C" if n["temperature_c"] > 0 else "—"
        power = f"{n['power_watts']:.0f}W" if n["power_watts"] > 0 else "—"
        lines.append(
            f"{n['friendly_name']:<20} {n['chip']:<20} {ram:>12} {disk:>12} {gpu:>6} {temp:>6} {power:>7} {n['status']:<8}"
        )
    return "\n".join(lines)


def format_node_detail(data: dict[str, Any]) -> str:
    lines = [
        f"Node:        {data['friendly_name']} ({data['node_id'][:12]}...)",
        f"Chip:        {data['chip']}",
        f"OS:          {data['os_version']}",
        f"Status:      {data['status']}",
        "",
        f"RAM:         {data['ram_used_gb']:.1f} / {data['ram_total_gb']:.1f} GB "
        f"{_bar(data['ram_used_percent'])}",
        f"Disk:        {data['disk_available_gb']:.1f} / {data['disk_total_gb']:.1f} GB free",
        "",
        f"GPU:         {data['gpu_usage_percent']:.1f}%",
        f"Temperature: {data['temperature_c']:.1f}°C",
        f"Power:       {data['power_watts']:.1f}W",
        "",
        f"IPs:         {', '.join(data['ip_addresses']) or '—'}",
        f"Connections: {', '.join(data['connection_types']) or '—'}",
        f"Thunderbolt: {'yes' if data['has_thunderbolt'] else 'no'}",
        f"RDMA:        {'enabled' if data['rdma_enabled'] else 'disabled'}",
    ]
    if data["loaded_models"]:
        lines.append(f"\nLoaded:      {', '.join(data['loaded_models'])}")
    return "\n".join(lines)


def format_models(data: dict[str, Any]) -> str:
    lines: list[str] = []
    if data["loaded"]:
        lines.append(f"{'Model':<35} {'Nodes':<25} {'Size':>8} {'Status':<10}")
        lines.append("─" * 80)
        for m in data["loaded"]:
            nodes_str = ", ".join(m["node_names"])
            size = f"{m['storage_size_gb']:.0f}GB"
            ready = "✓" if m["ready"] else "✗"
            lines.append(
                f"{m['model_name']:<35} {nodes_str:<25} {size:>8} {ready} {m['status']:<8}"
            )
    else:
        lines.append("No models loaded.")

    if data.get("downloading"):
        lines.append("")
        lines.append("DOWNLOADING")
        for d in data["downloading"]:
            lines.append(
                f"  {d['model_id']} → {d['node_name']}: "
                f"{d['progress_percent']:.1f}%"
            )

    return "\n".join(lines)


def format_model_status(data: dict[str, Any]) -> str:
    if not data["found"]:
        return f"Model '{data['model_id']}' is not loaded."

    lines = [
        f"Model:    {data['model_id']}",
        f"Status:   {data['status']}",
        f"Ready:    {'yes' if data['ready'] else 'no'}",
    ]
    if data.get("progress"):
        lines.append(f"Progress: {data['progress']}")
    if data.get("nodes"):
        lines.append(f"Nodes:    {', '.join(data['nodes'])}")
    return "\n".join(lines)


def format_action_response(data: dict[str, Any]) -> str:
    return data.get("message", str(data))
