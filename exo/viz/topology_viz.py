import math
from typing import List
from exo.helpers import exo_text
from exo.topology.topology import Topology
from exo.topology.partitioning_strategy import Partition
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.style import Style
from rich.color import Color
from exo.topology.device_capabilities import UNKNOWN_DEVICE_CAPABILITIES


class TopologyViz:
  def __init__(self, chatgpt_api_endpoint: str = None, web_chat_url: str = None):
    self.chatgpt_api_endpoint = chatgpt_api_endpoint
    self.web_chat_url = web_chat_url
    self.topology = Topology()
    self.partitions: List[Partition] = []

    self.console = Console()
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
    radius_x = 30  # Increased horizontal radius
    radius_y = 12  # Decreased vertical radius
    center_x, center_y = 50, 28  # Centered horizontally and moved up slightly

    # Generate visualization
    visualization = [[" " for _ in range(100)] for _ in range(55)]  # Decreased height

    # Add exo_text at the top in bright yellow
    exo_lines = exo_text.split("\n")
    yellow_style = Style(color="bright_yellow")
    max_line_length = max(len(line) for line in exo_lines)
    for i, line in enumerate(exo_lines):
      centered_line = line.center(max_line_length)
      start_x = (100 - max_line_length) // 2 + 15  # Center the text plus empirical adjustment of 15
      colored_line = Text(centered_line, style=yellow_style)
      for j, char in enumerate(str(colored_line)):
        if 0 <= start_x + j < 100 and i < len(visualization):
          visualization[i][start_x + j] = char

    # Display chatgpt_api_endpoint and web_chat_url if set
    info_lines = []
    if self.web_chat_url:
      info_lines.append(f"Web Chat URL (tinychat): {self.web_chat_url}")
    if self.chatgpt_api_endpoint:
      info_lines.append(f"ChatGPT API endpoint: {self.chatgpt_api_endpoint}")

    info_start_y = len(exo_lines) + 1
    for i, line in enumerate(info_lines):
      start_x = (100 - len(line)) // 2 + 15  # Center the info lines plus empirical adjustment of 15
      for j, char in enumerate(line):
        if 0 <= start_x + j < 100 and info_start_y + i < 55:
          visualization[info_start_y + i][start_x + j] = char

    # Calculate total FLOPS and position on the bar
    total_flops = sum(
      self.topology.nodes.get(partition.node_id, UNKNOWN_DEVICE_CAPABILITIES).flops.fp16
      for partition in self.partitions
    )
    bar_pos = (math.tanh(total_flops / 20 - 2) + 1) / 2

    # Add GPU poor/rich bar
    bar_width = 30  # Increased bar width
    bar_start_x = (100 - bar_width) // 2  # Center the bar
    bar_y = info_start_y + len(info_lines) + 1  # Position the bar below the info section with two cells of space

    # Create a gradient bar using emojis
    gradient_bar = Text()
    emojis = ["🟥", "🟧", "🟨", "🟩"]  # Red, Orange, Yellow, Green
    for i in range(bar_width):
      emoji_index = min(int(i / (bar_width / len(emojis))), len(emojis) - 1)
      gradient_bar.append(emojis[emoji_index])

    # Add the gradient bar to the visualization
    visualization[bar_y][bar_start_x - 1] = "["
    visualization[bar_y][bar_start_x + bar_width] = "]"
    for i, segment in enumerate(str(gradient_bar)):
      visualization[bar_y][bar_start_x + i] = segment

    # Add labels
    visualization[bar_y - 1][bar_start_x - 10 : bar_start_x - 3] = "GPU poor"
    visualization[bar_y - 1][bar_start_x + bar_width * 2 + 2 : bar_start_x + bar_width * 2 + 11] = "GPU rich"

    # Add position indicator and FLOPS value
    pos_x = bar_start_x + int(bar_pos * bar_width)
    flops_str = f"{total_flops:.2f} TFLOPS"
    visualization[bar_y - 1][pos_x] = "▼"
    visualization[bar_y + 1][
      pos_x - len(flops_str) // 2 : pos_x + len(flops_str) // 2 + len(flops_str) % 2
    ] = flops_str
    visualization[bar_y + 2][pos_x] = "▲"

    for i, partition in enumerate(self.partitions):
      device_capabilities = self.topology.nodes.get(partition.node_id, UNKNOWN_DEVICE_CAPABILITIES)

      angle = 2 * math.pi * i / num_partitions
      x = int(center_x + radius_x * math.cos(angle))
      y = int(center_y + radius_y * math.sin(angle))

      # Place node with different color for active node
      if partition.node_id == self.topology.active_node_id:
        visualization[y][x] = "🔴"  # Red circle for active node
      else:
        visualization[y][x] = "🔵"  # Blue circle for inactive nodes

      # Place node info (model, memory, TFLOPS, partition) on three lines
      node_info = [
        f"{device_capabilities.model} {device_capabilities.memory // 1024}GB",
        f"{device_capabilities.flops.fp16}TFLOPS",
        f"[{partition.start:.2f}-{partition.end:.2f}]",
      ]

      # Calculate info position based on angle
      info_distance_x = radius_x + 6  # Increased horizontal distance
      info_distance_y = radius_y + 3  # Decreased vertical distance
      info_x = int(center_x + info_distance_x * math.cos(angle))
      info_y = int(center_y + info_distance_y * math.sin(angle))

      # Adjust text position to avoid overwriting the node icon and prevent cutoff
      if info_x < x:  # Text is to the left of the node
        info_x = max(0, x - len(max(node_info, key=len)) - 1)
      elif info_x > x:  # Text is to the right of the node
        info_x = min(99 - len(max(node_info, key=len)), info_x)

      # Adjust for top and bottom nodes
      if 5 * math.pi / 4 < angle < 7 * math.pi / 4:  # Node is near the top
        info_x += 4  # Shift text slightly to the right
      elif math.pi / 4 < angle < 3 * math.pi / 4:  # Node is near the bottom
        info_x += 3  # Shift text slightly to the right
        info_y -= 2  # Move text up by two cells

      for j, line in enumerate(node_info):
        for k, char in enumerate(line):
          if 0 <= info_y + j < 55 and 0 <= info_x + k < 100:  # Updated height check
            # Ensure we're not overwriting the node icon
            if info_y + j != y or info_x + k != x:
              visualization[info_y + j][info_x + k] = char

      # Draw line to next node
      next_i = (i + 1) % num_partitions
      next_angle = 2 * math.pi * next_i / num_partitions
      next_x = int(center_x + radius_x * math.cos(next_angle))
      next_y = int(center_y + radius_y * math.sin(next_angle))

      # Simple line drawing
      steps = max(abs(next_x - x), abs(next_y - y))
      for step in range(1, steps):
        line_x = int(x + (next_x - x) * step / steps)
        line_y = int(y + (next_y - y) * step / steps)
        if 0 <= line_y < 55 and 0 <= line_x < 100:  # Updated height check
          visualization[line_y][line_x] = "-"

    # Convert to string
    return "\n".join("".join(str(char) for char in row) for row in visualization)
