"""
Simulated RadioServer to manage peer logic.
"""

class RadioServer:
    def __init__(self, port=None):
        self.port = port
        self.running = False

    def start(self):
        self.running = True
        print("[RadioServer] Started simulated radio server.")

    def stop(self):
        self.running = False
        print("[RadioServer] Stopped radio server.")
