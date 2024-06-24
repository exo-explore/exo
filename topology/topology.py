class Topology:
    def __init__(self):
        self.nodes = {}  # Maps node IDs to a tuple of (host, port, stats)

    def update_node(self, node_id, stats):
        self.nodes[node_id] = stats

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def all_nodes(self):
        return self.nodes.items()
