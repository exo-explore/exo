import unittest
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import asyncio
import os
from transformers import AutoTokenizer
from exo.inference.shard import Shard
from exo.inference.pytorch.inference import PyTorchDynamicShardInferenceEngine

def setup(rank, world_size):
    """
    Set up the distributed environment.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """
    Clean up the distributed environment.
    """
    dist.destroy_process_group()

def run_engine(rank, world_size, shard, queue):
    """
    Run the inference engine in a distributed setting.
    """
    setup(rank, world_size)

    # Initialize the engine
    engine = PyTorchDynamicShardInferenceEngine(debug=True)
    
    # Run ensure_shard to set up the model
    asyncio.run(engine.ensure_shard(shard))
    
    # Prepare the prompt
    prompt = "Why is the sky blue?"
    
    # Run inference
    output_data, new_inference_state, is_eos = asyncio.run(
        engine.infer_prompt(
            request_id="test_request", shard=shard, prompt=prompt
        )
    )

    # Put results in the queue to be checked in the test
    queue.put((output_data, new_inference_state, is_eos))

    cleanup()

class TestPyTorchDynamicShardInferenceEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world_size = torch.cuda.device_count()
        
        # Create a shard
        cls.shard = Shard(
            model_id="llama3-8b-sfr",
            start_layer=0,
            end_layer=0,
            n_layers=12
        )

    def test_infer_prompt(self):
        """
        Test the inference on a text prompt in a distributed setting.
        """
        mp.set_start_method('spawn', force=True)
        queue = mp.Queue()

        processes = []
        for rank in range(self.world_size):
            p = mp.Process(target=run_engine, args=(rank, self.world_size, self.shard, queue))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        output_data, new_inference_state, is_eos = queue.get()

        # Assertions
        self.assertIsNotNone(output_data)
        self.assertIsNotNone(new_inference_state)
        self.assertFalse(is_eos)

    @classmethod
    def tearDownClass(cls):
        """
        Clean up after the test.
        """
        mp.set_start_method('fork', force=True)  # Reset the multiprocessing start method to default

if __name__ == '__main__':
    unittest.main()
