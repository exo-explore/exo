import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import torch
from exo.inference.shard import Shard
from exo.inference.pytorch.helpers import build_transformer

class TestBuildTransformer(unittest.TestCase):

    def test_build_transformer(self):
        # Call the build_transformer function
        model = build_transformer(
            "gpt2", 
            quantize=True, 
            device="cuda"
        )

        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
