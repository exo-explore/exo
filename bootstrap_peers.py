#!/usr/bin/env python3
"""
Exo P2P Peer Bootstrap
Manually adds known peers to enable cross-platform discovery

This bypasses mDNS issues between macOS and Linux
"""

import requests
import json
import time
from typing import List, Dict

# Known cluster nodes
NODES = [
    {
        "name": "localhost (4x V100)",
        "ip": "192.168.0.160",
        "port": 34765,
        "api": "http://192.168.0.160:52415",
        "peer_id": "12D3KooWLC7a3t6givH4e7VtQ8WpmadfnpxvBSaVKxKa8Y3kCAuf"
    },
    {
        "name": ".106 (RTX 5070)",
        "ip": "192.168.0.106",
        "port": 38651,
        "api": "http://192.168.0.106:52415",
        "peer_id": "12D3KooWCw583XUFccb2RfCd55GpoSd8EFJ1NYwq7yNcjAj1swqv"
    },
    {
        "name": ".134 (M2 Mac)",
        "ip": "192.168.0.134",
        "port": 55987,
        "api": "http://192.168.0.134:52415",
        "peer_id": "12D3KooWB698CkfuX3CB9MZyySrirrHamvePF66Kz5xwbGZu4xsv"
    }
]

def check_node_online(node: Dict) -> bool:
    """Check if node is reachable"""
    try:
        resp = requests.get(f"{node['api']}/v1/models", timeout=2)
        return resp.status_code == 200
    except:
        return False

def get_multiaddr(node: Dict) -> str:
    """Generate libp2p multiaddr for node"""
    return f"/ip4/{node['ip']}/tcp/{node['port']}/p2p/{node['peer_id']}"

def bootstrap_peer(api_url: str, peer_multiaddr: str) -> bool:
    """
    Attempt to manually add peer via Exo API

    Note: This might not work if Exo doesn't expose a manual peer add endpoint.
    In that case, we need to modify the Rust code or use libp2p-daemon.
    """
    try:
        # Try to add peer (endpoint might not exist)
        resp = requests.post(
            f"{api_url}/api/peers/add",
            json={"multiaddr": peer_multiaddr},
            timeout=2
        )
        return resp.status_code == 200
    except:
        return False

def print_status():
    """Print cluster status"""
    print("\n" + "="*60)
    print("EXO CLUSTER STATUS")
    print("="*60)

    for node in NODES:
        status = "✅ ONLINE" if check_node_online(node) else "❌ OFFLINE"
        print(f"{node['name']:25} {status}")
        if status == "✅ ONLINE":
            print(f"  API: {node['api']}")
            print(f"  P2P: {get_multiaddr(node)}")
    print("="*60)

def print_bootstrap_instructions():
    """Print manual bootstrap instructions"""
    print("\n" + "="*60)
    print("MANUAL PEER BOOTSTRAP INSTRUCTIONS")
    print("="*60)
    print("\nSince Exo doesn't expose a peer add API endpoint,")
    print("we need to use libp2p-daemon or modify the code.")
    print("\nOption 1: Modify discovery.rs to add static peers")
    print("Option 2: Use libp2p-daemon to bridge connections")
    print("Option 3: Fix mDNS multicast on macOS")
    print("\nMultiaddrs to connect:")
    for node in NODES:
        print(f"\n{node['name']}:")
        print(f"  {get_multiaddr(node)}")
    print("="*60)

def create_static_peers_config():
    """Create a static peers configuration file"""
    config = {
        "static_peers": [
            {
                "name": node['name'],
                "multiaddr": get_multiaddr(node)
            }
            for node in NODES
        ]
    }

    with open("static_peers.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n✅ Created static_peers.json")
    print("This can be used to modify Exo to support static peer bootstrap")

def main():
    print("\n🔥 EXO P2P PEER BOOTSTRAP TOOL 🔥\n")

    # Check cluster status
    print_status()

    # Try to bootstrap (will likely fail without API endpoint)
    print("\n🔍 Attempting automatic peer bootstrap...")
    success = False
    for node in NODES:
        if not check_node_online(node):
            continue

        for peer in NODES:
            if peer['name'] == node['name']:
                continue

            peer_addr = get_multiaddr(peer)
            if bootstrap_peer(node['api'], peer_addr):
                print(f"  ✅ {node['name']} → {peer['name']}")
                success = True
            else:
                print(f"  ❌ {node['name']} → {peer['name']} (no API endpoint)")

    if not success:
        print("\n⚠️  Automatic bootstrap failed (expected)")
        print("Exo doesn't expose a peer add API endpoint.")

    # Generate static peers config
    create_static_peers_config()

    # Print manual instructions
    print_bootstrap_instructions()

    # Print mDNS debugging info
    print("\n" + "="*60)
    print("mDNS DEBUGGING")
    print("="*60)
    print("\nPossible issues:")
    print("1. macOS Firewall blocking mDNS multicast (224.0.0.251)")
    print("2. Different subnet (all nodes should be on 192.168.0.x)")
    print("3. mDNS query interval too long (1500 seconds!)")
    print("4. IPv6 disabled in Exo (might cause discovery issues)")
    print("\nTo test mDNS manually:")
    print("  macOS:  dns-sd -B _exo._tcp")
    print("  Linux:  avahi-browse -at")
    print("="*60)

if __name__ == "__main__":
    main()
