from exo.inference.inference_engine import InferenceEngine
from exo.download.shard_download import ShardDownloader
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json 
from exo.inference.dummy.lorem_ipsum import DummyTokenizer
import numpy as np 
from exo.inference.shard import Shard
from typing import Optional, Tuple

# Expanded Lorem Ipsum words list
words = """
lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor 
incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud 
exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute 
irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla 
pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui officia 
deserunt mollit anim id est laborum curabitur feugiat mauris fermentum praesent 
volutpat pellentesque libero aliquet gravida integer sagittis posuere morbi dictum 
donec tristique ultricies fringilla venenatis accumsan vestibulum vulputate integer 
nonummy augue ultrices lacinia convallis congue dictumst facilisis litora per aptent 
penatibus sociosqu placerat sociosqu himenaeos facilisi erat
""".split()


class DummyInferenceEngine(InferenceEngine):

    def __init__(self, shard_downloader: ShardDownloader):
        self.shard = None
        self.shard_downloader = shard_downloader
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def infer_prompt(self, 
                           request_id: str, 
                           shard: Shard, 
                           prompt: str, 
                           image_str: Optional[str] = None, 
                           inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        
        await self.ensure_shard(shard)
        
        start_pos = json.loads(inference_state or "{}").get("start_pos", 0)
        n_captured_toks = json.loads(inference_state or "{}").get("n_captured_toks", 0)
        
        toks = await asyncio.get_event_loop().run_in_executor(self.executor, self.tokenizer.encode, prompt)

        h = await asyncio.get_event_loop().run_in_executor(self.executor, lambda: self.tokenizer.model(toks, 1))
        print("infer_prompt")
        print(h)
        print(len(h))
        print("Captured tokens", n_captured_toks)

        if h.shape == (1,):
            start_pos += len(toks)
            start_pos += 1
            n_captured_toks = 0
            return np.array([[h.item()]]), json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}), h.item() == self.tokenizer.eos_token_id
        else:
            n_captured_toks = len(toks)
            return np.array(h), json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}), False


    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        await self.ensure_shard(shard)
        start_pos = json.loads(inference_state or "{}").get("start_pos", 0)
        n_captured_toks = json.loads(inference_state or "{}").get("n_captured_toks", 0)


        h = await asyncio.get_event_loop().run_in_executor(self.executor, lambda: self.tokenizer.model(input_data, 1))

        if h.shape == (1,):
            start_pos += n_captured_toks
            start_pos += 1
            n_captured_toks = 0
            return np.array([[h.item()]]), json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}), h.item() == self.tokenizer.eos_token_id
        else:
            return np.array(h), json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}), False


    async def ensure_shard(self, shard: Shard):
        """
        as a dummy, download the shard anyway to hold weights.
        """

        if self.shard == shard:
            return

        # ensure shard download as a dummy
        _ = await self.shard_downloader.ensure_shard(shard)

        if self.shard != shard:
            self.tokenizer = DummyTokenizer()
            self.model = self.tokenizer.model
            self.shard = shard
