import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, parent_dir)

from exo.networking.grpc.grpc_discovery import get_network_interfaces, get_interface_priority

def simulate_broadcast_receive(peer_id, interface, priority, known_peers):
    print(f"\nReceived broadcast from peer {peer_id} on interface {interface} with priority {priority}")
    
    if peer_id in known_peers:
        current_interface, current_priority = known_peers[peer_id]
        if priority > current_priority:
            print(f"Higher priority detected. Disconnecting from {current_interface} and reconnecting on {interface}")
            known_peers[peer_id] = (interface, priority)
        else:
            print(f"Keeping existing connection on {current_interface} with priority {current_priority}")
    else:
        print(f"New peer discovered. Connecting on {interface}")
        known_peers[peer_id] = (interface, priority)

def run_tests():
    print("Testing network interface detection, prioritization, and reconnection logic")
    interfaces = get_network_interfaces()

    # Test 1: Check if interfaces are detected and sorted 
    assert len(interfaces) > 0, "No interfaces detected"
    assert all(interfaces[i][1] >= interfaces[i+1][1] for i in range(len(interfaces)-1)), "Interfaces are not correctly sorted by priority"

    print("\nPrioritized list of interfaces:")
    for i, (interface, priority) in enumerate(interfaces, 1):
        print(f"{i}. {interface} (Priority: {priority})")

    # Test 2: Simulate broadcast receives and reconnection 
    known_peers = {}

    # Simulate receiving broadcasts from the same peer on different interfaces
    peer_id = "test_peer_1"
    for interface, priority in interfaces:
        simulate_broadcast_receive(peer_id, interface, priority, known_peers)
        time.sleep(0.1)  # Add a small delay

    # Test 3: Check if the peer is connected to the highest priority interface
    assert peer_id in known_peers, "Peer was not added to known_peers"
    final_interface, final_priority = known_peers[peer_id]
    assert final_priority == max(priority for _, priority in interfaces), "Peer is not connected to the highest priority interface"

    print("\nFinal known peers:")
    for peer_id, (interface, priority) in known_peers.items():
        print(f"Peer {peer_id}: Connected on {interface} with priority {priority}")

    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    run_tests()
