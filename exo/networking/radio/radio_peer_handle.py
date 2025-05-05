"""
Simulated RadioPeerHandle using an in-memory message bus.
"""

from queue import Queue

# Shared in-memory message bus (simulates radio channel)
message_bus = {}

class RadioPeerHandle:
    def __init__(self, peer_id):
        self.peer_id = peer_id
        self.inbox = Queue()
        message_bus[peer_id] = self.inbox
        self.connected = True
        print(f"[{self.peer_id}] Initialized radio peer.")

    def send(self, target_id, data):
        """Simulate sending data to another peer."""
        if target_id not in message_bus:
            raise ValueError(f"Target peer '{target_id}' not found.")
        message_bus[target_id].put((self.peer_id, data))
        print(f"[{self.peer_id}] Sent to {target_id}: {data}")

    def receive(self):
        """Simulate receiving a message."""
        if self.inbox.empty():
            print(f"[{self.peer_id}] Inbox empty.")
            return None
        sender_id, data = self.inbox.get()
        print(f"[{self.peer_id}] Received from {sender_id}: {data}")
        return data
