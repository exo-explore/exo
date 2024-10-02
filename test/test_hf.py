import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import asyncio
from exo.download.hf.hf_helpers import get_weight_map

async def test_get_weight_map():
  repo_ids = [
    "mlx-community/quantized-gemma-2b",
    "mlx-community/Meta-Llama-3.1-8B-4bit",
    "mlx-community/Meta-Llama-3.1-70B-4bit",
    "mlx-community/Meta-Llama-3.1-405B-4bit",
  ]
  for repo_id in repo_ids:
    weight_map = await get_weight_map(repo_id)
    assert weight_map is not None, "Weight map should not be None"
    assert isinstance(weight_map, dict), "Weight map should be a dictionary"
    assert len(weight_map) > 0, "Weight map should not be empty"
    print(f"OK: {repo_id}")

if __name__ == "__main__":
  asyncio.run(test_get_weight_map())
