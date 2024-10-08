import math
from collections import OrderedDict
from typing import List, Optional, Tuple, Dict
from exo.helpers import exo_text, pretty_print_bytes, pretty_print_bytes_per_second
from exo.topology.topology import Topology
from exo.topology.partitioning_strategy import Partition
from exo.download.hf.hf_helpers import RepoProgressEvent
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
    if request_id in self.requests:
      self.requests[request_id] = [prompt, self.requests[request_id][1]]
    else:
      self.requests[request_id] = [prompt, ""]
    self.refresh()

  def update_prompt_output(self, request_id: str, output: Optional[str] = None):
    if request_id in self.requests:
      self.requests[request_id] = [self.requests[request_id][0], output]
    else:
      self.requests[request_id] = ["", output]
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
    max_lines = 13  # Maximum number of lines for the entire panel content

    for (prompt, output) in reversed(requests):
      prompt_icon, output_icon = "💬️", "🤖"

      # Process prompt
      prompt_lines = prompt.split('\n')
      if len(prompt_lines) > max_lines // 2:
        prompt_lines = prompt_lines[:max_lines//2 - 1] + ['...']
      prompt_text = Text(f"{prompt_icon} ", style="bold bright_blue")
      prompt_text.append('\n'.join(line[:max_width] for line in prompt_lines), style="white")

      # Process output
      output_lines = output.split('\n')
      remaining_lines = max_lines - len(prompt_lines) - 2  # -2 for spacing
      if len(output_lines) > remaining_lines:
        output_lines = output_lines[:remaining_lines - 1] + ['...']
      output_text = Text(f"\n{output_icon} ", style="bold bright_magenta")
      output_text.append('\n'.join(line[:max_width] for line in output_lines), style="white")

      content.append(prompt_text)
      content.append(output_text)
      content.append(Text())  # Empty line between entries

    return Panel(
      Group(*content),
      title="",
      border_style="cyan",
      height=15,  # Increased height to accommodate multiple lines
      expand=True  # Allow the panel to expand to full width
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
          start_x = (100 - max_line_length) // 2 + 15
          for j, char in enumerate(centered_line):
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
      total_flops = sum(
          self.topology.nodes.get(partition.node_id, UNKNOWN_DEVICE_CAPABILITIES).flops.fp16
          for partition in self.partitions
      )
      bar_pos = (math.tanh(math.cbrt(total_flops)/2.5 - 2) + 1)

      # Calculate total memory available in GB
      total_memory_mb = sum(
          self.topology.nodes.get(partition.node_id, UNKNOWN_DEVICE_CAPABILITIES).memory_available
          for partition in self.partitions
      )
      total_memory_gb = total_memory_mb / 1024  # Convert MB to GB

      # Add GPU poor/rich bar
      bar_width = 30
      bar_start_x = (100 - bar_width) // 2
      bar_y = info_start_y + len(info_lines) + 4  # Adjusted for additional text lines

      # Create a gradient bar using emojis
      gradient_bar = Text()
      emojis = ["🟥", "🟧", "🟨", "🟩"]
      for i in range(bar_width):
          emoji_index = min(int(i / (bar_width / len(emojis))), len(emojis) - 1)
          gradient_bar.append(emojis[emoji_index])

      # Add the gradient bar to the visualization
      visualization[bar_y][bar_start_x - 1] = "["
      visualization[bar_y][bar_start_x + bar_width] = "]"
      for i, segment in enumerate(str(gradient_bar)):
          visualization[bar_y][bar_start_x + i] = segment

      # Add labels "GPU poor" and "GPU rich"
      gpu_poor_str = "GPU poor"
      gpu_poor_x = bar_start_x - len(gpu_poor_str) - 2
      gpu_poor_y = bar_y
      for i, char in enumerate(gpu_poor_str):
          visualization[gpu_poor_y][gpu_poor_x + i] = char

      gpu_rich_str = "GPU rich"
      gpu_rich_x = bar_start_x + bar_width + 2
      gpu_rich_y = bar_y
      for i, char in enumerate(gpu_rich_str):
          visualization[gpu_rich_y][gpu_rich_x + i] = char

      # Add position indicators
      pos_x = bar_start_x + int(bar_pos * bar_width)
      visualization[bar_y - 1][pos_x] = "▼"
      visualization[bar_y + 1][pos_x] = "▲"

      # Add total memory available at the top of the bar
      memory_str = f"{total_memory_gb:.2f} GB AVAILABLE"
      memory_str_x = (100 - len(memory_str)) // 2
      memory_str_y = bar_y - 3  # Position above the bar
      for i, char in enumerate(memory_str):
          visualization[memory_str_y][memory_str_x + i] = char

      # Add total FLOPS available at the bottom of the bar
      flops_str = f"{total_flops:.2f} TFLOPS AVAILABLE"
      flops_str_x = (100 - len(flops_str)) // 2
      flops_str_y = bar_y + 3  # Position below the bar
      for i, char in enumerate(flops_str):
          visualization[flops_str_y][flops_str_x + i] = char

      # Proceed with the rest of your visualization code...

      # Convert to string
      return "\n".join("".join(char for char in row) for row in visualization)


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
