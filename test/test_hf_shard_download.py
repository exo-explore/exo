import pytest
import asyncio
from pathlib import Path
from unittest import mock
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.inference.shard import Shard
from exo import DEBUG

class MockPath:
  def __init__(self, exists=True):
    self.exists = lambda: exists
    self.iterdir = lambda: []

  def __truediv__(self, other):
    return self

mock_was_cancelled = False

async def mock_download_shard(self, shard):
  """Mock the _download_shard method to simulate a long download"""
  global mock_was_cancelled
  try:
    if DEBUG >= 2: print(f"Starting mock download for {shard}")
    await asyncio.sleep(0.5)  # Simulate download time
    if DEBUG >= 2: print(f"Mock download completed for {shard}")
    return Path("/mock/download/path")
  except asyncio.CancelledError:
    mock_was_cancelled = True
    if DEBUG >= 2: print(f"Mock download was cancelled for {shard}, completing anyway")
    # Continue despite cancellation
    await asyncio.sleep(0.5)
    if DEBUG >= 2: print(f"Mock download completed after cancellation for {shard}")
    return Path("/mock/download/path")

@pytest.mark.asyncio
async def test_download_protection():
  """Test that downloads are protected from cancellation"""
  global mock_was_cancelled
  mock_was_cancelled = False

  with mock.patch('exo.download.hf.hf_shard_download.get_repo_root', return_value=MockPath(exists=False)), \
       mock.patch.object(HFShardDownloader, '_download_shard', mock_download_shard):
    
    downloader = HFShardDownloader()
    shard = Shard(model_id="test-model", start_layer=0, end_layer=1, n_layers=1)

    # Create a future we'll use to control the test flow
    download_complete = asyncio.Future()

    async def do_download():
      try:
        result = await downloader.ensure_shard(shard)
        download_complete.set_result(result)
      except Exception as e:
        if not download_complete.done():
          download_complete.set_exception(e)

    # Start the download
    task = asyncio.create_task(do_download())
    
    # Give it a moment to start
    await asyncio.sleep(0.1)

    # Try to cancel the task
    if DEBUG >= 2: print("Attempting to cancel download task")
    task.cancel()

    # Wait for result with timeout
    try:
      result = await asyncio.wait_for(download_complete, timeout=2.0)
      assert isinstance(result, Path), "Should return a Path"
      assert shard not in downloader.active_downloads, "Download task should be cleaned up"
      assert shard in downloader.completed_downloads, "Download should be marked as completed"
      if DEBUG >= 2: print("Download completed successfully despite cancellation")
    except asyncio.TimeoutError:
      pytest.fail("Download did not complete in time")
    except Exception as e:
      pytest.fail(f"Download failed with error: {e}")

@pytest.mark.asyncio
async def test_multiple_downloads():
  """Test handling multiple downloads for the same shard"""
  with mock.patch('exo.download.hf.hf_shard_download.get_repo_root', return_value=MockPath(exists=False)), \
       mock.patch.object(HFShardDownloader, '_download_shard', mock_download_shard):
    
    downloader = HFShardDownloader()
    shard = Shard(model_id="test-model", start_layer=0, end_layer=1, n_layers=1)

    # Start both downloads with a small delay between them
    if DEBUG >= 2: print("Starting first download")
    download1 = asyncio.create_task(downloader.ensure_shard(shard))
    
    await asyncio.sleep(0.2)  # Give first download time to start
    
    if DEBUG >= 2: print("Starting second download")
    download2 = asyncio.create_task(downloader.ensure_shard(shard))
    
    # Wait for both downloads to complete
    if DEBUG >= 2: print("Waiting for downloads to complete")
    
    path1 = await download1
    if DEBUG >= 2: print(f"First download completed with path: {path1}")
    
    path2 = await download2
    if DEBUG >= 2: print(f"Second download completed with path: {path2}")
    
    # Verify results
    assert isinstance(path1, Path), "First download should return a Path"
    assert isinstance(path2, Path), "Second download should return a Path"
    assert path1 == path2, "Multiple downloads should return same path"
    assert shard not in downloader.active_downloads, "Download tasks should be cleaned up"
    assert shard in downloader.completed_downloads, "Download should be marked as completed"

@pytest.mark.asyncio
async def test_download_error_handling():
  """Test that errors during download are handled properly"""
  
  async def mock_download_error(self, shard):
    await asyncio.sleep(0.1)  # Simulate some work
    raise Exception("Download failed")
  
  with mock.patch('exo.download.hf.hf_shard_download.get_repo_root', return_value=MockPath(exists=False)), \
       mock.patch.object(HFShardDownloader, '_download_shard', mock_download_error):
    
    downloader = HFShardDownloader()
    shard = Shard(model_id="test-model", start_layer=0, end_layer=1, n_layers=1)
    
    with pytest.raises(Exception) as exc_info:
      await downloader.ensure_shard(shard)
    
    assert str(exc_info.value) == "Download failed"
    assert shard not in downloader.active_downloads, "Failed download should be cleaned up"

if __name__ == "__main__":
  pytest.main([__file__, "-v"])