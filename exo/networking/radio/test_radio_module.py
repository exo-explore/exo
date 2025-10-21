"""
Unit tests for the simulated radio networking module.
"""

import unittest
from radio_peer_handle import RadioPeerHandle
from radio_server import RadioServer

class TestRadioPeerHandle(unittest.TestCase):
    def test_peer_send_receive(self):
        peer1 = RadioPeerHandle("peer1")
        peer2 = RadioPeerHandle("peer2")
        peer1.send("peer2", "Hello, peer2!")
        received = peer2.receive()
        self.assertEqual(received, "Hello, peer2!")

    def test_empty_inbox(self):
        peer = RadioPeerHandle("peer3")
        result = peer.receive()
        self.assertIsNone(result)

class TestRadioServer(unittest.TestCase):
    def test_server_start_stop(self):
        server = RadioServer()
        server.start()
        self.assertTrue(server.running)
        server.stop()
        self.assertFalse(server.running)

if __name__ == "__main__":
    unittest.main()
