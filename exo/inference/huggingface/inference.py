from exo.inference.inference_engine import InferenceEngine
import transformers
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
import numpy as np
import torch
import asyncio
from loguru import logger
import torch
from concurrent.futures import ThreadPoolExecutor

TEMPERATURE = 0.85
def download_model_from_model_id(model_id):
    pass

_executor = ThreadPoolExecutor(max_workers=1)
class HuggingfaceInferenceEngine(InferenceEngine):
    def __init__(self, shard: Shard):
        self.shard = None
        self.model_id = shard.model_id
        self.executor = _executor
        
        pass
    
    async def encode(self, shard, prompt):
        """Encodes prompt to tokens using the tokenizer"""
        await self.ensure_shard(shard)
        logger.info(f"Encoding prompt {prompt}")
        tokens = self.tokenizer(prompt, return_tensors="pt")
        logger.info(f"Prompt encoded to tokens shape: {tokens}")
        return tokens
    
    
    async def sample(self, x: np.ndarray, temp=TEMPERATURE, top_p: float = 0.0) -> np.ndarray:
        """Samples the next token from the logits"""
        def sample_wrapper():
            logits = torch.Tensor(x[:, -1, :])
            
            # Create processors list without None values
            processors_list = [
                transformers.TemperatureLogitsWarper(temp),
            ]
            if top_p > 0:
                processors_list.append(transformers.TopPLogitsWarper(top_p))
                
            processors = transformers.LogitsProcessorList(processors_list)
            
            # Process logits
            filtered_logits = processors(None, logits.detach().cpu().numpy())
            probs = torch.nn.functional.softmax(torch.Tensor(filtered_logits), dim=-1)
            return probs.multinomial(num_samples=1).cpu().numpy().astype(int)
        
        return await asyncio.get_running_loop().run_in_executor(self.executor, sample_wrapper)
    
    async def decode(self, shard, tokens):
        """Decodes tokens to text using the tokenizer"""
       
        # self.ensure_shard(shard)
        return self.tokenizer.decode(tokens)
    
    async def infer_tensor(self, request_id, shard, input_data, inference_state=None):
        logger.info(f"Running inference for request {request_id}, shard {shard}, input_data shape: {len(input_data)}")
        
        start_layer = shard.start_layer
        try:
            input_ids = input_data["input_ids"]
            attention_mask = input_data["attention_mask"]
        except:
            # Convert numpy array to tensor if needed
            input_ids = torch.tensor(input_data) if isinstance(input_data, np.ndarray) else input_data
            # Generate attention mask for all tokens
            attention_mask = torch.ones_like(input_ids)
        
        # Generate position IDs for RoPE
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Initial embedding
        if start_layer == 0:
            outputs = self.model.model.embed_tokens(input_ids)
        else:
            outputs = input_ids
        
        # Prepare attention mask
        batch_size = outputs.shape[0]
        causal_mask = self._prepare_causal_mask(
            batch_size,
            seq_length,
            dtype=outputs.dtype,
            device=outputs.device
        )
        
        # Process through layers
        for i, layer_idx in enumerate(range(start_layer, shard.end_layer + 1)):
            # logger.info(f"Running inference for layer {layer_idx}")
            residual = outputs
            
            outputs = self.model.model.layers[i](
                hidden_states=outputs,
                attention_mask=causal_mask,
                position_ids=position_ids
            )
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            outputs = outputs + residual
            
            # logger.info(f"Layer {layer_idx} output shape: {outputs.shape}")
        
        if shard.end_layer == shard.n_layers - 1 :
            logger.info("Final layer reached, applying final layer normalization and LM head")
            outputs = self.model.model.norm(outputs)
            outputs = self.model.lm_head(outputs)
            # all_tokens = torch.argmax(outputs, dim=-1)
            # logger.info(f" Returning all tokens can be sampled, Final output shape: {all_tokens.shape}")
            # outputs = all_tokens.detach().cpu().numpy()
            
        return outputs, inference_state

    def _prepare_causal_mask(self, batch_size, seq_length, dtype, device):
        # Create causal mask
        mask = torch.triu(torch.ones((seq_length, seq_length), device=device) * -float("inf"), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length, seq_length]
        mask = mask.expand(batch_size, 1, seq_length, seq_length)
        mask = mask.to(dtype=dtype)
        return mask
    
    async def load_checkpoint(self, shard, path):
        """ Loads the wieght into the model defined after enuring shard exits"""
        self.ensure_shard(shard)
        return 
    
    async def ensure_shard(self, shard: Shard):
        """make hure the ard exists otherwise downloads the model and the tokenizer and initialise the shard """
        if self.shard == shard:
            return
        
        # model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
        if self.shard != shard:
            logger.info(f"Downloading model {shard.model_id}")
            self.shard = shard
            self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            logger.info(f"Model {shard.model_id} downloaded")


    async def infer_prompt(self, request_id, shard, prompt, inference_state = None):
        """takes prompt and infers till the end layer of the shard"""
        tokens = await self.encode(shard, prompt) ## [input_ids, attention_mask, position_ids]
        # x = tokens.reshape(1, -1)
        output_data, inference_state = await self.infer_tensor(request_id, shard, tokens, inference_state)
        return output_data, inference_state