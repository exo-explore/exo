import math
from collections import OrderedDict
from typing import List, Optional, Tuple, Dict
from exo.helpers import exo_text, pretty_print_bytes, pretty_print_bytes_per_second
from exo.topology.topology import Topology
from exo.topology.partitioning_strategy import Partition
from exo.download.download_progress import RepoProgressEvent
from exo.topology.device_capabilities import UNKNOWN_DEVICE_CAPABILITIES
from rich.console import Console, Group
from rich.text import Text
from rich.live import Live
from rich.style import Style
from rich.table import Table
from rich.layout import Layout
from rich.syntax import Syntax
from rich.panel import Panel
from rich.markdown import Markdown


class TopologyViz:
  def __init__(self, chatgpt_api_endpoints: List[str] = [], web_chat_urls: List[str] = []):
    self.chatgpt_api_endpoints = chatgpt_api_endpoints
    self.web_chat_urls = web_chat_urls
    self.topology = Topology()
    self.partitions: List[Partition] = []
    self.node_id = None
    self.node_download_progress: Dict[str, RepoProgressEvent] = {}
    self.requests: OrderedDict[str, Tuple[str, str]] = {}

    self.console = Console()
    self.layout = Layout()
    self.layout.split(Layout(name="main"), Layout(name="prompt_output", size=15), Layout(name="download", size=25))
    self.main_panel = Panel(self._generate_main_layout(), title="Exo Cluster (0 nodes)", border_style="bright_yellow")
    self.prompt_output_panel = Panel("", title="Prompt and Output", border_style="green")
    self.download_panel = Panel("", title="Download Progress", border_style="cyan")
    self.layout["main"].update(self.main_panel)
    self.layout["prompt_output"].update(self.prompt_output_panel)
    self.layout["download"].update(self.download_panel)

    # Initially hide the prompt_output panel
    self.layout["prompt_output"].visible = False
    self.live_panel = Live(self.layout, auto_refresh=False, console=self.console)
    self.live_panel.start()

  def update_visualization(self, topology: Topology, partitions: List[Partition], node_id: Optional[str] = None, node_download_progress: Dict[str, RepoProgressEvent] = {}):
    self.topology = topology
    self.partitions = partitions
    self.node_id = node_id
    if node_download_progress:
      self.node_download_progress = node_download_progress
    self.refresh()

  def update_prompt(self, request_id: str, prompt: Optional[str] = None):
    self.requests[request_id] = [prompt, self.requests.get(request_id, ["", ""])[1]]
    self.refresh()

  def update_prompt_output(self, request_id: str, output: Optional[str] = None):
    self.requests[request_id] = [self.requests.get(request_id, ["", ""])[0], output]
    self.refresh()

  def refresh(self):
    self.main_panel.renderable = self._generate_main_layout()
    # Update the panel title with the number of nodes and partitions
    node_count = len(self.topology.nodes)
    self.main_panel.title = f"Exo Cluster ({node_count} node{'s' if node_count != 1 else ''})"

    # Update and show/hide prompt and output panel
    if any(r[0] or r[1] for r in self.requests.values()):
      self.prompt_output_panel = self._generate_prompt_output_layout()
      self.layout["prompt_output"].update(self.prompt_output_panel)
      self.layout["prompt_output"].visible = True
    else:
      self.layout["prompt_output"].visible = False

    # Only show download_panel if there are in-progress downloads
    if any(progress.status == "in_progress" for progress in self.node_download_progress.values()):
      self.download_panel.renderable = self._generate_download_layout()
      self.layout["download"].visible = True
    else:
      self.layout["download"].visible = False

    self.live_panel.update(self.layout, refresh=True)

  def _generate_prompt_output_layout(self) -> Panel:
    content = []
    requests = list(self.requests.values())[-3:]  # Get the 3 most recent requests
    max_width = self.console.width - 6  # Full width minus padding and icon

    # Calculate available height for content
    panel_height = 15  # Fixed panel height
    available_lines = panel_height - 2  # Subtract 2 for panel borders
    lines_per_request = available_lines // len(requests) if requests else 0

    for (prompt, output) in reversed(requests):
      prompt_icon, output_icon = "💬️", "🤖"

      # Equal space allocation for prompt and output
      max_prompt_lines = lines_per_request // 2
      max_output_lines = lines_per_request - max_prompt_lines - 1  # -1 for spacing

      # Process prompt
      prompt_lines = []
      for line in prompt.split('\n'):
        words = line.split()
        current_line = []
        current_length = 0

        for word in words:
          if current_length + len(word) + 1 <= max_width:
            current_line.append(word)
            current_length += len(word) + 1
          else:
            if current_line:
              prompt_lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)

        if current_line:
          prompt_lines.append(' '.join(current_line))

      # Truncate prompt if needed
      if len(prompt_lines) > max_prompt_lines:
        prompt_lines = prompt_lines[:max_prompt_lines]
        if prompt_lines:
          last_line = prompt_lines[-1]
          if len(last_line) + 4 <= max_width:
            prompt_lines[-1] = last_line + " ..."
          else:
            prompt_lines[-1] = last_line[:max_width-4] + " ..."

      prompt_text = Text(f"{prompt_icon} ", style="bold bright_blue")
      prompt_text.append('\n'.join(prompt_lines), style="white")
      content.append(prompt_text)

      # Process output with similar word wrapping
      if output:  # Only process output if it exists
        output_lines = []
        for line in output.split('\n'):
          words = line.split()
          current_line = []
          current_length = 0

          for word in words:
            if current_length + len(word) + 1 <= max_width:
              current_line.append(word)
              current_length += len(word) + 1
            else:
              if current_line:
                output_lines.append(' '.join(current_line))
              current_line = [word]
              current_length = len(word)

          if current_line:
            output_lines.append(' '.join(current_line))

        # Truncate output if needed
        if len(output_lines) > max_output_lines:
          output_lines = output_lines[:max_output_lines]
          if output_lines:
            last_line = output_lines[-1]
            if len(last_line) + 4 <= max_width:
              output_lines[-1] = last_line + " ..."
            else:
              output_lines[-1] = last_line[:max_width-4] + " ..."

        output_text = Text(f"{output_icon} ", style="bold bright_magenta")
        output_text.append('\n'.join(output_lines), style="white")
        content.append(output_text)

      content.append(Text())  # Empty line between entries

    return Panel(
      Group(*content),
      title="",
      border_style="cyan",
      height=panel_height,
      expand=True
    )

  def _generate_main_layout(self) -> str:
    # Calculate visualization parameters
    num_partitions = len(self.partitions)
    radius_x = 30
    radius_y = 12
    center_x, center_y = 50, 24  # Increased center_y to add more space

    # Generate visualization
    visualization = [[" " for _ in range(100)] for _ in range(48)]  # Increased height to 48

    # Add exo_text at the top in bright yellow
    exo_lines = exo_text.split("\n")
    yellow_style = Style(color="bright_yellow")
    max_line_length = max(len(line) for line in exo_lines)
    for i, line in enumerate(exo_lines):
      centered_line = line.center(max_line_length)
      start_x = (100-max_line_length) // 2 + 15
      colored_line = Text(centered_line, style=yellow_style)
      for j, char in enumerate(str(colored_line)):
        if 0 <= start_x + j < 100 and i < len(visualization):
          visualization[i][start_x + j] = char

    # Display chatgpt_api_endpoints and web_chat_urls
    info_lines = []
    if len(self.web_chat_urls) > 0:
      info_lines.append(f"Web Chat URL (tinychat): {' '.join(self.web_chat_urls[:1])}")
    if len(self.chatgpt_api_endpoints) > 0:
      info_lines.append(f"ChatGPT API endpoint: {' '.join(self.chatgpt_api_endpoints[:1])}")

    info_start_y = len(exo_lines) + 1
    for i, line in enumerate(info_lines):
      start_x = (100 - len(line)) // 2 + 15
      for j, char in enumerate(line):
        if 0 <= start_x + j < 100 and info_start_y + i < 48:
          visualization[info_start_y + i][start_x + j] = char

    # Calculate total FLOPS and position on the bar
    total_flops = sum(self.topology.nodes.get(partition.node_id, UNKNOWN_DEVICE_CAPABILITIES).flops.fp16 for partition in self.partitions)
    bar_pos = (math.tanh(total_flops**(1/3)/2.5 - 2) + 1)

    # Add GPU poor/rich bar
    bar_width = 30
    bar_start_x = (100-bar_width) // 2
    bar_y = info_start_y + len(info_lines) + 1

    # Create a gradient bar using emojis
    gradient_bar = Text()
    emojis = ["🟥", "🟧", "🟨", "🟩"]
    for i in range(bar_width):
      emoji_index = min(int(i/(bar_width/len(emojis))), len(emojis) - 1)
      gradient_bar.append(emojis[emoji_index])

    # Add the gradient bar to the visualization
    visualization[bar_y][bar_start_x - 1] = "["
    visualization[bar_y][bar_start_x + bar_width] = "]"
    for i, segment in enumerate(str(gradient_bar)):
      visualization[bar_y][bar_start_x + i] = segment

    # Add labels
    visualization[bar_y - 1][bar_start_x - 10:bar_start_x - 3] = "GPU poor"
    visualization[bar_y - 1][bar_start_x + bar_width*2 + 2:bar_start_x + bar_width*2 + 11] = "GPU rich"

    # Add position indicator and FLOPS value
    pos_x = bar_start_x + int(bar_pos*bar_width)
    flops_str = f"{total_flops:.2f} TFLOPS"
    visualization[bar_y - 1][pos_x] = "▼"
    visualization[bar_y + 1][pos_x - len(flops_str) // 2:pos_x + len(flops_str) // 2 + len(flops_str) % 2] = flops_str
    visualization[bar_y + 2][pos_x] = "▲"

    # Add an extra empty line for spacing
    bar_y += 4

    for i, partition in enumerate(self.partitions):
      device_capabilities = self.topology.nodes.get(partition.node_id, UNKNOWN_DEVICE_CAPABILITIES)

      angle = 2*math.pi*i/num_partitions
      x = int(center_x + radius_x*math.cos(angle))
      y = int(center_y + radius_y*math.sin(angle))

      # Place node with different color for active node and this node
      if partition.node_id == self.topology.active_node_id:
        visualization[y][x] = "🔴"
      elif partition.node_id == self.node_id:
        visualization[y][x] = "🟢"
      else:
        visualization[y][x] = "🔵"

      # Place node info (model, memory, TFLOPS, partition) on three lines
      node_info = [
        f"{device_capabilities.model} {device_capabilities.memory // 1024}GB",
        f"{device_capabilities.flops.fp16}TFLOPS",
        f"[{partition.start:.2f}-{partition.end:.2f}]",
      ]

      # Calculate info position based on angle
      info_distance_x = radius_x + 6
      info_distance_y = radius_y + 3
      info_x = int(center_x + info_distance_x*math.cos(angle))
      info_y = int(center_y + info_distance_y*math.sin(angle))

      # Adjust text position to avoid overwriting the node icon and prevent cutoff
      if info_x < x:
        info_x = max(0, x - len(max(node_info, key=len)) - 1)
      elif info_x > x:
        info_x = min(99 - len(max(node_info, key=len)), info_x)

      # Adjust for top and bottom nodes
      if 5*math.pi/4 < angle < 7*math.pi/4:
        info_x += 4
      elif math.pi/4 < angle < 3*math.pi/4:
        info_x += 3
        info_y -= 2

      for j, line in enumerate(node_info):
        for k, char in enumerate(line):
          if 0 <= info_y + j < 48 and 0 <= info_x + k < 100:
            if info_y + j != y or info_x + k != x:
              visualization[info_y + j][info_x + k] = char

      # Draw line to next node and add connection description
      next_i = (i+1) % num_partitions
      next_angle = 2*math.pi*next_i/num_partitions
      next_x = int(center_x + radius_x*math.cos(next_angle))
      next_y = int(center_y + radius_y*math.sin(next_angle))

      # Get connection descriptions
      conn1 = self.topology.peer_graph.get(partition.node_id, set())
      conn2 = self.topology.peer_graph.get(self.partitions[next_i].node_id, set())
      description1 = next((c.description for c in conn1 if c.to_id == self.partitions[next_i].node_id), "")
      description2 = next((c.description for c in conn2 if c.to_id == partition.node_id), "")
      connection_description = f"{description1}/{description2}"

      # Simple line drawing
      steps = max(abs(next_x - x), abs(next_y - y))
      for step in range(1, steps):
        line_x = int(x + (next_x-x)*step/steps)
        line_y = int(y + (next_y-y)*step/steps)
        if 0 <= line_y < 48 and 0 <= line_x < 100:
          visualization[line_y][line_x] = "-"

      # Add connection description near the midpoint of the line
      mid_x = (x + next_x) // 2
      mid_y = (y + next_y) // 2
      # Center the description text around the midpoint
      desc_start_x = mid_x - len(connection_description) // 2
      for j, char in enumerate(connection_description):
        if 0 <= mid_y < 48 and 0 <= desc_start_x + j < 100:
          visualization[mid_y][desc_start_x + j] = char

    # Convert to string
    return "\n".join("".join(str(char) for char in row) for row in visualization)

  def _generate_download_layout(self) -> Table:
    summary = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    summary.add_column("Info", style="cyan", no_wrap=True, ratio=50)
    summary.add_column("Progress", style="cyan", no_wrap=True, ratio=40)
    summary.add_column("Percentage", style="cyan", no_wrap=True, ratio=10)

    # Current node download progress
    if self.node_id in self.node_download_progress:
      download_progress = self.node_download_progress[self.node_id]
      title = f"Downloading model {download_progress.repo_id}@{download_progress.repo_revision} ({download_progress.completed_files}/{download_progress.total_files}):"
      summary.add_row(Text(title, style="bold"))
      progress_info = f"{pretty_print_bytes(download_progress.downloaded_bytes)} / {pretty_print_bytes(download_progress.total_bytes)} ({pretty_print_bytes_per_second(download_progress.overall_speed)})"
      summary.add_row(progress_info)

      eta_info = f"{download_progress.overall_eta}"
      summary.add_row(eta_info)

      summary.add_row("")  # Empty row for spacing

      for file_path, file_progress in download_progress.file_progress.items():
        if file_progress.status != "complete":
          progress = int(file_progress.downloaded/file_progress.total*30)
          bar = f"[{'=' * progress}{' ' * (30 - progress)}]"
          percentage = f"{file_progress.downloaded / file_progress.total * 100:.0f}%"
          summary.add_row(Text(file_path[:30], style="cyan"), bar, percentage)

    summary.add_row("")  # Empty row for spacing

    # Other nodes download progress summary
    summary.add_row(Text("Other Nodes Download Progress:", style="bold"))
    for node_id, progress in self.node_download_progress.items():
      if node_id != self.node_id:
        device = self.topology.nodes.get(node_id)
        partition = next((p for p in self.partitions if p.node_id == node_id), None)
        partition_info = f"[{partition.start:.2f}-{partition.end:.2f}]" if partition else ""
        percentage = progress.downloaded_bytes/progress.total_bytes*100 if progress.total_bytes > 0 else 0
        speed = pretty_print_bytes_per_second(progress.overall_speed)
        device_info = f"{device.model if device else 'Unknown Device'} {device.memory // 1024 if device else '?'}GB {partition_info}"
        progress_info = f"{progress.repo_id}@{progress.repo_revision} ({speed})"
        progress_bar = f"[{'=' * int(percentage // 3.33)}{' ' * (30 - int(percentage // 3.33))}]"
        percentage_str = f"{percentage:.1f}%"
        eta_str = f"{progress.overall_eta}"
        summary.add_row(device_info, progress_info, percentage_str)
        summary.add_row("", progress_bar, eta_str)

    return summary
