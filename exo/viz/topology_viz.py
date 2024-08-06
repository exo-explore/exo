import math
from typing import List, Optional, Tuple
from exo.helpers import exo_text, pretty_print_bytes, pretty_print_bytes_per_second
from exo.topology.topology import Topology
from exo.topology.partitioning_strategy import Partition
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.style import Style
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from exo.topology.device_capabilities import UNKNOWN_DEVICE_CAPABILITIES
from exo.inference.hf_helpers import HFRepoProgressEvent

class TopologyViz:
  def __init__(self, chatgpt_api_endpoint: str = None, web_chat_url: str = None):
    self.chatgpt_api_endpoint = chatgpt_api_endpoint
    self.web_chat_url = web_chat_url
    self.topology = Topology()
    self.partitions: List[Partition] = []
    self.download_progress = None

    self.console = Console()
    self.panel = Panel(self._generate_layout(), title="Exo Cluster (0 nodes)", border_style="bright_yellow")
    self.live_panel = Live(self.panel, auto_refresh=False, console=self.console)
    self.live_panel.start()

  def update_visualization(self, topology: Topology, partitions: List[Partition], download_progress: HFRepoProgressEvent = None):
    self.topology = topology
    self.partitions = partitions
    self.download_progress = download_progress
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

    # Draw download first so everything else is drawn on top
    # If a download is in progress, show the download info summary
    if self.download_progress and self.download_progress.status != "complete":
        download_summary = _generate_download_summary(self.download_progress)
        download_panel = Panel(
            download_summary,
            title="Download Progress",
            border_style="cyan",
            expand=False,
            width=96,  # Further reduced to ensure it fits within the visualization
            height=None  # Allow the panel to adjust its height based on content
        )
        console = Console(width=98, height=55)  # Reduced console width
        with console.capture() as capture:
            console.print(download_panel)
        download_lines = capture.get().split('\n')
        download_start_y = 15
        panel_width = len(max(download_lines, key=len))
        start_x = max(1, (100 - panel_width) // 2)  # Ensure start_x is at least 1 to avoid left border cut-off
        for i, line in enumerate(download_lines):
            for j, char in enumerate(line):
                if 1 <= start_x + j < 99 and download_start_y + i < 55:  # Ensure we don't write to the rightmost column
                    visualization[download_start_y + i][start_x + j] = char


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
    total_flops = sum(self.topology.nodes.get(partition.node_id, UNKNOWN_DEVICE_CAPABILITIES).flops.fp16 for partition in self.partitions)
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
    visualization[bar_y + 1][pos_x - len(flops_str) // 2 : pos_x + len(flops_str) // 2 + len(flops_str) % 2] = flops_str
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

def _generate_download_summary(download_progress) -> Table:
    summary = Table(show_header=False, box=None, padding=(0, 1))
    summary.add_column("Info", style="cyan", no_wrap=True)
    summary.add_column("Progress", style="cyan", no_wrap=True)
    summary.add_column("Percentage", style="cyan", no_wrap=True)

    title = f"Downloading model ({download_progress.completed_files}/{download_progress.total_files}):"
    summary.add_row(Text(title, style="bold"))
    progress_info = f"{pretty_print_bytes(download_progress.downloaded_bytes)} / {pretty_print_bytes(download_progress.total_bytes)} ({pretty_print_bytes_per_second(download_progress.overall_speed)})"
    summary.add_row(progress_info)

    eta_info = f"ETA: {download_progress.overall_eta}"
    summary.add_row(eta_info)

    summary.add_row("")  # Empty row for spacing

    for file_path, file_progress in download_progress.file_progress.items():
      if file_progress.status != "complete":
        progress = int(file_progress.downloaded / file_progress.total * 20)  # Increased bar width
        bar = f"[{'=' * progress}{' ' * (20 - progress)}]"
        percentage = f"{file_progress.downloaded / file_progress.total * 100:.0f}%"
        summary.add_row(Text(file_path[:20], style="cyan"), bar, percentage)  # Increased file path length

    return summary