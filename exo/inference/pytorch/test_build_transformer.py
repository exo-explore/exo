import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from exo.inference.shard import Shard
from exo.inference.pytorch.helpers import build_transformer

class TestBuildTransformer(unittest.TestCase):

    @patch('torch.load')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"weight_map": {"0": "pytorch_model.bin"}}')
    def test_build_transformer(self, mock_open, mock_from_pretrained, mock_torch_load):
        # Mocking model and weights
        mock_model = MagicMock(spec=AutoModelForCausalLM)
        mock_from_pretrained.return_value = mock_model

        mock_weights = {
            "model.embed_tokens.weight": torch.randn(1024, 768),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(768, 768),
            # Add other necessary mock weights here
        }
        mock_torch_load.return_value = mock_weights

        # Define the shard
        shard = Shard(model_id="mock_model", start_layer=0, end_layer=0, n_layers=1)

        # Call the build_transformer function
        model = build_transformer("mock_model", shard, model_size="8B", quantize=True, device="cpu")

        # Assertions to verify the function behavior
        mock_from_pretrained.assert_called_once_with(
            "mock_model",
            torch_dtype=torch.float32,
            device_map=None
        )

        mock_open.assert_called_once_with("mock_model/pytorch_model.bin.index.json")
        mock_torch_load.assert_called()

        mock_model.load_state_dict.assert_called()
        self.assertEqual(model, mock_model)

if __name__ == '__main__':
    unittest.main()
