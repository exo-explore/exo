import asyncio
import unittest
from datetime import timedelta
from exo.viz.topology_viz import TopologyViz
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.topology.partitioning_strategy import Partition
from exo.download.download_progress import RepoProgressEvent


def create_hf_repo_progress_event(
  completed_files: int = 5,
  total_files: int = 10,
  downloaded_bytes: int = 500000000,
  downloaded_bytes_this_session: int = 250000000,
  total_bytes: int = 1000000000,
  overall_speed: int = 5000000,
  overall_eta: timedelta = timedelta(seconds=100),
  file_progress: dict = None,
  status: str = "in_progress"
) -> RepoProgressEvent:
  if file_progress is None:
    file_progress = {
      "file1.bin":
        RepoFileProgressEvent(
          repo_id="repo_id",
          repo_revision="repo_revision",
          file_path="file1.bin",
          downloaded=100000000,
          downloaded_this_session=50000000,
          total=200000000,
          speed=1000000,
          eta=timedelta(seconds=100),
          status="in_progress"
        ), "file2.bin":
          RepoFileProgressEvent(
            repo_id="repo_id",
            repo_revision="repo_revision",
            file_path="file2.bin",
            downloaded=200000000,
            downloaded_this_session=100000000,
            total=200000000,
            speed=2000000,
            eta=timedelta(seconds=0),
            status="complete"
          )
    }

  return RepoProgressEvent(
    repo_id="repo_id",
    repo_revision="repo_revision",
    completed_files=completed_files,
    total_files=total_files,
    downloaded_bytes=downloaded_bytes,
    downloaded_bytes_this_session=downloaded_bytes_this_session,
    total_bytes=total_bytes,
    overall_speed=overall_speed,
    overall_eta=overall_eta,
    file_progress=file_progress,
    status=status
  )


class TestNodeViz(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    self.topology = Topology()
    self.topology.update_node(
      "node1",
      DeviceCapabilities(model="ModelA", chip="ChipA", memory=8*1024, flops=DeviceFlops(fp32=1.0, fp16=2.0, int8=4.0)),
    )
    self.topology.update_node(
      "node2",
      DeviceCapabilities(model="ModelB", chip="ChipB", memory=16*1024, flops=DeviceFlops(fp32=2.0, fp16=4.0, int8=8.0)),
    )
    self.topology.update_node(
      "node3",
      DeviceCapabilities(model="ModelC", chip="ChipC", memory=32*1024, flops=DeviceFlops(fp32=4.0, fp16=8.0, int8=16.0)),
    )
    self.topology.update_node(
      "node4",
      DeviceCapabilities(model="ModelD", chip="ChipD", memory=64*1024, flops=DeviceFlops(fp32=8.0, fp16=16.0, int8=32.0)),
    )

    self.top_viz = TopologyViz()
    await asyncio.sleep(2)  # Simulate running for a short time

  async def test_layout_generation(self):
    # self.top_viz._generate_layout()
    self.top_viz.refresh()
    import time

    time.sleep(2)
    self.top_viz.update_visualization(
      self.topology,
      [
        Partition("node1", 0, 0.2),
        Partition("node4", 0.2, 0.4),
        Partition("node2", 0.4, 0.8),
        Partition("node3", 0.8, 0.9),
      ],
      "node1",
      {
        "node1": create_hf_repo_progress_event(),
        "node2": create_hf_repo_progress_event(),
        "node3": create_hf_repo_progress_event(),
        "node4": create_hf_repo_progress_event(),
      },
    )
    time.sleep(2)
    self.topology.active_node_id = "node3"
    self.top_viz.update_visualization(
      self.topology,
      [
        Partition("node1", 0, 0.3),
        Partition("node5", 0.3, 0.5),
        Partition("node2", 0.5, 0.7),
        Partition("node4", 0.7, 0.9),
      ],
      "node5",
      {
        "node1": create_hf_repo_progress_event(),
        "node5": create_hf_repo_progress_event(),
      },
    )
    time.sleep(2)


if __name__ == "__main__":
  unittest.main()
