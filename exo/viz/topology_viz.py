import math
from typing import Dict, List
from exo.helpers import exo_text
from exo.orchestration.node import Node
from exo.topology.topology import Topology
from exo.topology.partitioning_strategy import Partition
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.style import Style
from exo.topology.device_capabilities import DeviceCapabilities, UNKNOWN_DEVICE_CAPABILITIES

class TopologyViz:
    def __init__(self):
        self.console = Console()
        self.topology = Topology()
        self.partitions: List[Partition] = []
        self.panel = Panel(self._generate_layout(), title=f"Exo Cluster (0 nodes)", border_style="bright_yellow")
        self.live_panel = Live(self.panel, auto_refresh=False, console=self.console)
        self.live_panel.start()

    def update_visualization(self, topology: Topology, partitions: List[Partition]):
        self.topology = topology
        self.partitions = partitions
        self.refresh()

    def refresh(self):
        self.panel.renderable = self._generate_layout()
        # Update the panel title with the number of nodes and partitions
        node_count = len(self.topology.nodes)
        self.panel.title = f"Exo Cluster ({node_count} node{'s' if node_count != 1 else ''})"
        self.live_panel.update(self.panel, refresh=True)

    def _generate_layout(self) -> str:
        # Calculate visualization parameters
        num_partitions = len(self.partitions)
        radius = 12  # Reduced radius
        center_x, center_y = 45, 25  # Adjusted center_x to center the visualization

        # Generate visualization
        visualization = [[' ' for _ in range(90)] for _ in range(45)]  # Increased width to 90

        # Add exo_text at the top in bright yellow
        exo_lines = exo_text.split('\n')
        yellow_style = Style(color="bright_yellow")
        max_line_length = max(len(line) for line in exo_lines)
        for i, line in enumerate(exo_lines):
            centered_line = line.center(max_line_length)
            start_x = (90 - max_line_length) // 2  # Calculate starting x position to center the text
            colored_line = Text(centered_line, style=yellow_style)
            for j, char in enumerate(str(colored_line)):
                if 0 <= start_x + j < 90 and i < len(visualization):  # Ensure we don't exceed the width and height
                    visualization[i][start_x + j] = char

        for i, partition in enumerate(self.partitions):
            device_capabilities = self.topology.nodes.get(partition.node_id, UNKNOWN_DEVICE_CAPABILITIES)

            angle = 2 * math.pi * i / num_partitions
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))

            # Place node with different color for active node
            if partition.node_id == self.topology.active_node_id:
                visualization[y][x] = '🔴'  # Red circle for active node
            else:
                visualization[y][x] = '🔵'  # Blue circle for inactive nodes

            # Place node info (ID, start_layer, end_layer)
            node_info = [
                f"Model: {device_capabilities.model}",
                f"Mem: {device_capabilities.memory // 1024}GB",
                f"FLOPS: {device_capabilities.flops.fp16}T",
                f"Part: {partition.start:.2f}-{partition.end:.2f}"
            ]

            # Calculate info position based on angle
            info_distance = radius + 3  # Reduced distance
            info_x = int(center_x + info_distance * math.cos(angle))
            info_y = int(center_y + info_distance * math.sin(angle))

            # Adjust text position to avoid overwriting the node icon
            if info_x < x:  # Text is to the left of the node
                info_x = max(0, x - len(max(node_info, key=len)) - 1)
            elif info_x > x:  # Text is to the right of the node
                info_x = min(89 - len(max(node_info, key=len)), info_x)

            for j, line in enumerate(node_info):
                for k, char in enumerate(line):
                    if 0 <= info_y + j < 45 and 0 <= info_x + k < 90:  # Updated width check
                        # Ensure we're not overwriting the node icon
                        if info_y + j != y or info_x + k != x:
                            visualization[info_y + j][info_x + k] = char

            # Draw line to next node
            next_i = (i + 1) % num_partitions
            next_angle = 2 * math.pi * next_i / num_partitions
            next_x = int(center_x + radius * math.cos(next_angle))
            next_y = int(center_y + radius * math.sin(next_angle))

            # Simple line drawing
            steps = max(abs(next_x - x), abs(next_y - y))
            for step in range(1, steps):
                line_x = int(x + (next_x - x) * step / steps)
                line_y = int(y + (next_y - y) * step / steps)
                if 0 <= line_y < 45 and 0 <= line_x < 90:  # Updated width check
                    visualization[line_y][line_x] = '-'

        # Convert to string
        return '\n'.join(''.join(str(char) for char in row) for row in visualization)
