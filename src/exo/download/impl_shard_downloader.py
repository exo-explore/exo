"""The download settings main.py hands to the DownloadCoordinator.

Kept under its old name so main.py doesn't change in this PR.
"""

from exo.download.transfer import DownloadSettings


def exo_shard_downloader(
    max_parallel_downloads: int = 8, offline: bool = False
) -> DownloadSettings:
    del offline  # the coordinator is told directly
    return DownloadSettings(parallel_files=max_parallel_downloads)
