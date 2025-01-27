from typing import Dict, Callable, Coroutine, Any, Literal
from exo.inference.shard import Shard
from dataclasses import dataclass
from datetime import timedelta


@dataclass
class RepoFileProgressEvent:
  repo_id: str
  repo_revision: str
  file_path: str
  downloaded: int
  downloaded_this_session: int
  total: int
  speed: int
  eta: timedelta
  status: Literal["not_started", "in_progress", "complete"]
  start_time: float

  def to_dict(self):
    return {
      "repo_id": self.repo_id, "repo_revision": self.repo_revision, "file_path": self.file_path, "downloaded": self.downloaded, "downloaded_this_session": self.downloaded_this_session,
      "total": self.total, "speed": self.speed, "eta": self.eta.total_seconds(), "status": self.status, "start_time": self.start_time
    }

  @classmethod
  def from_dict(cls, data):
    if 'eta' in data: data['eta'] = timedelta(seconds=data['eta'])
    return cls(**data)


@dataclass
class RepoProgressEvent:
  shard: Shard
  repo_id: str
  repo_revision: str
  completed_files: int
  total_files: int
  downloaded_bytes: int
  downloaded_bytes_this_session: int
  total_bytes: int
  overall_speed: int
  overall_eta: timedelta
  file_progress: Dict[str, RepoFileProgressEvent]
  status: Literal["not_started", "in_progress", "complete"]

  def to_dict(self):
    return {
      "shard": self.shard.to_dict(), "repo_id": self.repo_id, "repo_revision": self.repo_revision, "completed_files": self.completed_files, "total_files": self.total_files, "downloaded_bytes": self.downloaded_bytes,
      "downloaded_bytes_this_session": self.downloaded_bytes_this_session, "total_bytes": self.total_bytes, "overall_speed": self.overall_speed, "overall_eta": self.overall_eta.total_seconds(),
      "file_progress": {k: v.to_dict()
                        for k, v in self.file_progress.items()}, "status": self.status
    }

  @classmethod
  def from_dict(cls, data):
    if 'overall_eta' in data: data['overall_eta'] = timedelta(seconds=data['overall_eta'])
    if 'file_progress' in data: data['file_progress'] = {k: RepoFileProgressEvent.from_dict(v) for k, v in data['file_progress'].items()}
    if 'shard' in data: data['shard'] = Shard.from_dict(data['shard'])

    return cls(**data)


RepoFileProgressCallback = Callable[[RepoFileProgressEvent], Coroutine[Any, Any, None]]
RepoProgressCallback = Callable[[RepoProgressEvent], Coroutine[Any, Any, None]]
