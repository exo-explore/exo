import numpy as np
import asyncio
import os
import time
import glob
import platform
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

from ..inference_engine import InferenceEngine
from ..shard import Shard
from ..memory_manager import get_memory_config_for_model, should_prioritize_gpu, get_gpu_optimization_config
from exo.download.shard_download import ShardDownloader
from exo.helpers import DEBUG

try:
    from llama_cpp import Llama
    from llama_cpp import llama_cpp
    LLAMA_CPP_AVAILABLE = True
    
    # Check GPU support compilation
    GPU_OFFLOAD_SUPPORT = False
    try:
        if hasattr(llama_cpp, 'llama_supports_gpu_offload'):
            GPU_OFFLOAD_SUPPORT = llama_cpp.llama_supports_gpu_offload()
        else:
            # Fallback: try to detect GPU functions
            GPU_OFFLOAD_SUPPORT = hasattr(llama_cpp, 'llama_get_device_count')
    except:
        GPU_OFFLOAD_SUPPORT = False
        
    if DEBUG >= 1:
        print(f"llama-cpp-python available with GPU offload support: {GPU_OFFLOAD_SUPPORT}")
        if not GPU_OFFLOAD_SUPPORT:
            print("WARNING:  llama-cpp-python was compiled without GPU support!")
            print("To enable GPU acceleration:")
            if platform.system().lower() == "windows":
                print("  Windows: Run fix_llamacpp_gpu.bat or fix_llamacpp_gpu.ps1")
                print("  Or manually: set CMAKE_ARGS=-DGGML_CUDA=on && pip install llama-cpp-python --force-reinstall --no-cache-dir")
            else:
                print("  Linux/macOS: Run ./fix_llamacpp_gpu.sh")
                print("  Or manually: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python --force-reinstall --no-cache-dir")
            
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    GPU_OFFLOAD_SUPPORT = False
    if DEBUG >= 1:
        print("llama-cpp-python not available. Install with: pip install llama-cpp-python")


class LlamaTokenizerWrapper:
    """Wrapper to provide tokenizer interface for llama.cpp model"""
    def __init__(self, model):
        self.model = model
        self._eos_token_id = None
        
    @property
    def eos_token_id(self):
        if self._eos_token_id is None:
            try:
                # First try the native llama.cpp method
                if hasattr(self.model, 'token_eos'):
                    self._eos_token_id = self.model.token_eos()
                else:
                    # Fallback to metadata approach
                    if hasattr(self.model, 'metadata'):
                        metadata = self.model.metadata
                        if 'tokenizer.ggml.eos_token_id' in metadata:
                            self._eos_token_id = int(metadata['tokenizer.ggml.eos_token_id'])
                        elif 'tokenizer.eos_token_id' in metadata:
                            self._eos_token_id = int(metadata['tokenizer.eos_token_id'])
                        else:
                            # Common fallback values for different model types
                            if 'qwen' in metadata.get('general.name', '').lower():
                                self._eos_token_id = 151645  # Qwen models typically use this
                            elif 'llama' in metadata.get('general.architecture', '').lower():
                                self._eos_token_id = 2  # Llama models typically use 2
                            else:
                                self._eos_token_id = 2  # Default fallback
                    else:
                        # If no metadata, use common default
                        self._eos_token_id = 2
                        
            except Exception as e:
                if DEBUG >= 1:
                    print(f"Could not determine EOS token ID, using default: {e}")
                self._eos_token_id = 2  # Safe default
                
        return self._eos_token_id
    
    def encode(self, text):
        """Encode text to tokens"""
        # Use the proper llama.cpp tokenize method
        if isinstance(text, str):
            text = text.encode('utf-8')
        return self.model.tokenize(text, add_bos=True, special=False)
    
    def decode(self, tokens):
        """Decode tokens to text"""
        # Use the proper llama.cpp detokenize method
        if isinstance(tokens, (list, tuple)):
            tokens = list(tokens)
        elif hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        
        result = self.model.detokenize(tokens, special=False)
        if isinstance(result, bytes):
            return result.decode('utf-8', errors='ignore')
        return result


class LlamaCppInferenceEngine(InferenceEngine):
    def __init__(self, shard_downloader: ShardDownloader):
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python is required but not installed. Install with: pip install llama-cpp-python")
        
        # Store GPU support status
        self.gpu_offload_available = GPU_OFFLOAD_SUPPORT
        
        self.shard = None
        self.shard_downloader = shard_downloader
        self.model = None
        self.tokenizer = None
        self.session = {}
        # Increase workers for better concurrent handling - use min(4, cpu_count) for optimal balance
        max_workers = min(4, (os.cpu_count() or 1))
        self._llama_thread = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="llama")
        self._shard_lock = asyncio.Lock()
        
        # Model cache for multiple concurrent requests
        self.model_cache = OrderedDict()
        self.inference_state_cache = OrderedDict()
        
        # Default parameters optimized for GGUF models and distributed inference
        self.default_params = {
            "n_ctx": 4096,  # Conservative context size for stability - will be adjusted based on VRAM
            "n_batch": 512,  # Conservative batch size to avoid memory issues
            "n_ubatch": 256,  # Smaller micro-batch for better memory control
            "n_threads": None,  # Auto-detect CPU threads
            "n_gpu_layers": -1,  # Offload all layers to GPU if available
            "use_mmap": True,  # Use memory mapping for better performance
            "use_mlock": False,  # Don't lock model in memory initially
            "f16_kv": True,  # Use half-precision for key/value cache
            "logits_all": False,  # Only compute logits for last token
            "vocab_only": False,  # Load full model
            "verbose": DEBUG >= 2,
            "offload_kqv": True,  # Try GPU KV cache first, fallback if needed
            # Optimizations for stable CUDA inference
            "low_vram": False,  # Will be set based on GPU detection
            "numa": False,  # NUMA support
            "rope_scaling_type": None,  # RoPE scaling type
            "rope_freq_base": 0.0,  # RoPE frequency base
            "rope_freq_scale": 1.0,  # RoPE frequency scale
            # Enable optimizations for distributed computing
            "split_mode": 1,  # Row-wise model splitting
            "main_gpu": 0,  # Primary GPU for tensor operations
            # Additional stability parameters
            "mul_mat_q": True,  # Use quantized matrix multiplication
            "cont_batching": False,  # Disable continuous batching for now
        }

    async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
        await self.ensure_shard(shard)
        tokens = await asyncio.get_running_loop().run_in_executor(
            self._llama_thread,
            lambda: self.tokenizer.encode(prompt)
        )
        return np.array(tokens, dtype=np.int32)

    async def decode(self, shard: Shard, tokens) -> str:
        await self.ensure_shard(shard)
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        return await asyncio.get_running_loop().run_in_executor(
            self._llama_thread,
            lambda: self.tokenizer.decode(tokens)
        )

    async def sample(self, x: np.ndarray, temp: float = 0.0, top_p: float = 1.0) -> np.ndarray:
        """Sample from the model output logits"""
        # llama.cpp handles sampling internally during generation
        # This method is called during the generation process
        if len(x.shape) == 3:
            # If we have batch dimension, take the last token
            logits = x[0, -1, :]
        elif len(x.shape) == 2:
            # Take the last token from sequence
            logits = x[-1, :]
        else:
            logits = x
        
        # Convert to float32 for sampling
        logits = logits.astype(np.float32)
        
        # Apply temperature
        if temp > 0:
            logits = logits / temp
        
        # Apply top-p filtering if needed
        if top_p < 1.0:
            sorted_indices = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_indices]
            cumulative_probs = np.cumsum(self._softmax(sorted_logits))
            
            # Find cutoff
            cutoff_idx = np.searchsorted(cumulative_probs, top_p)
            if cutoff_idx < len(sorted_indices):
                cutoff_logit = sorted_logits[cutoff_idx]
                logits[logits < cutoff_logit] = -np.inf
        
        # Sample from distribution
        if temp == 0:
            # Greedy sampling
            token = np.argmax(logits)
        else:
            # Probabilistic sampling
            probs = self._softmax(logits)
            token = np.random.choice(len(probs), p=probs)
        
        return np.array([token], dtype=np.int32)

    def _softmax(self, x):
        """Compute softmax values for array x"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _clean_llm_response(self, text: str, stop_sequences: list) -> str:
        """Clean LLM response from system tokens and artifacts"""
        import re
        
        if not text or not isinstance(text, str):
            return text
        
        cleaned = text
        
        # Remove stop sequences that leaked through
        for stop_seq in stop_sequences:
            if stop_seq in cleaned:
                cleaned = cleaned.split(stop_seq)[0]
                break
        
        # Remove system tokens that commonly leak through
        system_tokens = [
            '<|begin_of_text|>',
            '<|end_of_text|>',
            '<|start_header_id|>',
            '<|end_header_id|>', 
            '<|eot_id|>',
            '<|im_start|>',
            '<|im_end|>',
            '<|endoftext|>',
            '</s>',
            '<s>',
            '<|assistant|>',
            '<|user|>',
            '<|system|>',
            'system<|end_header_id|>',
            'user<|end_header_id|>',
            'assistant<|end_header_id|>',
            'Cutting Knowledge Date: December 2023',
        ]
        
        # Remove system tokens - case insensitive
        for token in system_tokens:
            cleaned = re.sub(re.escape(token), '', cleaned, flags=re.IGNORECASE)
        
        # Remove system message patterns
        system_patterns = [
            r'<\|begin_of_text\|>.*?<\|start_header_id\|>.*?<\|end_header_id\|>',
            r'Cutting Knowledge Date:.*?(?=\n|$)',
            r'Today Date:.*?(?=\n|$)',
            r'^.*?<\|end_header_id\|>\s*',  # Remove header remnants at start
            r'^\s*system\s*$',  # Remove standalone "system" 
            r'^\s*assistant\s*$',  # Remove standalone "assistant"
            r'^\s*user\s*$',  # Remove standalone "user"
        ]
        
        for pattern in system_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
        # Clean up whitespace and empty lines
        lines = cleaned.split('\n')
        non_empty_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.isspace():
                # Skip lines that are just system tokens or fragments
                if not any(token.lower() in stripped.lower() for token in ['cutting knowledge', 'today date:', '<|']):
                    non_empty_lines.append(line)
        
        # Rejoin and normalize
        cleaned = '\n'.join(non_empty_lines).strip()
        
        # Remove excessive newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
        await self.ensure_shard(shard)
        
        if DEBUG >= 2:
            print(f"LlamaCpp infer_tensor called with input shape: {input_data.shape}")
        
        # Convert input to tokens if needed
        if input_data.dtype == np.int32 or input_data.dtype == np.int64:
            tokens = input_data.flatten().tolist()
        else:
            # If we get logits, we need to sample
            return await self._process_logits(input_data, inference_state)
        
        # Get or create inference state for this request
        if inference_state is None:
            inference_state = self._get_inference_state(request_id)
        
        # Run inference
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                self._llama_thread,
                lambda: self._run_direct_generation(tokens, inference_state, request_id)
            )
            
            output_logits, new_state = result
            
            # Store updated state
            self._update_inference_state(request_id, new_state)
            
            return output_logits, new_state
            
        except Exception as e:
            if DEBUG >= 1:
                print(f"Error in LlamaCpp inference: {e}")
            raise

    def _run_direct_generation(self, tokens, inference_state, request_id):
        """Generate multiple tokens at once to avoid Unicode splitting issues"""
        try:
            # Get generation state
            gen_state = inference_state.get('generation_state', {})
            accumulated_tokens = gen_state.get('accumulated_tokens', [])
            generation_complete = gen_state.get('complete', False)
            prompt_processed = gen_state.get('prompt_processed', False)
            
            # Clear any stale state for new requests
            if not prompt_processed and accumulated_tokens:
                if DEBUG >= 1:
                    print(f"[{request_id}] Clearing stale tokens from previous request")
                accumulated_tokens = []
                generation_complete = False
            
            # If generation is complete, return EOS
            if generation_complete:
                if DEBUG >= 1:
                    print(f"[{request_id}] Generation already complete, returning EOS")
                vocab_size = self.model.n_vocab()
                logits = np.full(vocab_size, -100.0, dtype=np.float32)
                logits[self.tokenizer.eos_token_id] = 0.0
                return np.array([logits], dtype=np.float32), inference_state
            
            # If we don't have accumulated tokens yet, generate a complete response
            if not accumulated_tokens:
                # Convert input tokens to text
                prompt_text = self.model.detokenize(tokens)
                if isinstance(prompt_text, bytes):
                    prompt_text = prompt_text.decode('utf-8', errors='ignore')
                    if DEBUG >= 1:
                        print(f"[{request_id}] Generating complete response for prompt length: {len(prompt_text)}")
                
                # Safety check: ensure the prompt has meaningful content
                if len(prompt_text.strip()) < 10:
                    if DEBUG >= 1:
                        print(f"[{request_id}] WARNING: Very short prompt detected: '{prompt_text}'")
                    # Add a fallback prompt to encourage a proper response
                    prompt_text = prompt_text + "\n\nPlease provide a helpful and detailed response."
                
                # Generate complete response with proper stopping
                # Include comprehensive stop sequences to prevent system token leakage
                stop_sequences = [
                    "<|begin_of_text|>",
                    "<|start_header_id|>",
                    "<|end_header_id|>",
                    "<|eot_id|>",
                    "<|im_start|>",
                    "<|im_end|>", 
                    "<|endoftext|>",
                    "</s>", 
                    "<s>",
                    "\n\nHuman:", 
                    "\n\nUser:", 
                    "\n\nAssistant:",
                    "<|end|>",
                    "<|assistant|>",
                    "<|user|>",
                    "<|system|>",
                    "system<|end_header_id|>",
                    "user<|end_header_id|>", 
                    "assistant<|end_header_id|>",
                    "Cutting Knowledge Date:",
                    "\n\n---",  # Common document separator
                    "\n\n##",   # Markdown header separator
                ]
                
                if DEBUG >= 1:
                    print(f"[{request_id}] Generating with prompt: {prompt_text[:200]}...")
                
                # Calculate max tokens based on model's context and prompt length
                model_context = self.model.n_ctx()
                prompt_tokens = len(self.model.tokenize(prompt_text.encode('utf-8'), add_bos=False, special=False))
                max_response_tokens = max(1024, model_context - prompt_tokens - 100)  # Reserve 100 tokens buffer
                
                if DEBUG >= 1:
                    print(f"[{request_id}] Model context: {model_context}, prompt tokens: {prompt_tokens}, max response: {max_response_tokens}")
                
                response = self.model.create_completion(
                    prompt_text,
                    max_tokens=max_response_tokens,  # Use full available tokens
                    temperature=0.7,
                    top_p=0.9,
                    stream=False,
                    echo=False,
                    stop=stop_sequences,
                    repeat_penalty=1.1,
                    top_k=40,
                    frequency_penalty=0.0  # Reduce frequency penalty
                )
                
                generated_text = ""
                finish_reason = "unknown"
                if response and 'choices' in response and len(response['choices']) > 0:
                    choice = response['choices'][0]
                    generated_text = choice['text'].strip()
                    finish_reason = choice.get('finish_reason', 'unknown')
                
                if DEBUG >= 1:
                    print(f"[{request_id}] Generated text (finish_reason: {finish_reason}): '{generated_text[:200]}{'...' if len(generated_text) > 200 else ''}'")
                
                # Convert to tokens for streaming
                if generated_text:
                    # Clean up any leaked stop sequences and system tokens
                    original_text = generated_text
                    cleaned_text = self._clean_llm_response(generated_text, stop_sequences)
                    
                    if cleaned_text != original_text and DEBUG >= 1:
                        print(f"[{request_id}] Cleaned response from {len(original_text)} to {len(cleaned_text)} chars")
                    
                    generated_text = cleaned_text
                    
                    # Handle thinking content more carefully to preserve context
                    import re
                    
                    # Don't remove thinking content completely - it may contain important context
                    # Only clean up very specific issues that break responses
                    original_text = generated_text
                    
                    # Check for thinking tags but preserve the content
                    thinking_patterns = [
                        r'<think>(.*?)</think>',
                        r'<thinking>(.*?)</thinking>'
                    ]
                    
                    has_thinking = False
                    for pattern in thinking_patterns:
                        if re.search(pattern, generated_text, flags=re.DOTALL):
                            has_thinking = True
                            break
                    
                    # Only process thinking tags if the response seems complete
                    if has_thinking and len(generated_text.strip()) > 50:
                        for pattern in thinking_patterns:
                            matches = re.findall(pattern, generated_text, flags=re.DOTALL)
                            if matches:
                                thinking_content = matches[0].strip()
                                # Keep thinking content but format it better
                                response_content = re.sub(pattern, f'[Internal reasoning: {thinking_content[:100]}...]\n\n', generated_text, flags=re.DOTALL)
                                generated_text = response_content
                                if DEBUG >= 1:
                                    print(f"[{request_id}] Processed thinking content: '{thinking_content[:50]}...'")
                                break
                    
                    # Critical fix: Never accept very short responses that might be artifacts
                    if len(generated_text.strip()) < 10 and len(original_text.strip()) > len(generated_text.strip()):
                        if DEBUG >= 1:
                            print(f"[{request_id}] Response too short after processing, reverting to original")
                        generated_text = original_text
                    
                    # Additional safety: if the response is just "Sure" or similar, regenerate
                    if generated_text.strip().lower() in ['sure', 'sure.', 'ok', 'okay', 'yes']:
                        if DEBUG >= 1:
                            print(f"[{request_id}] Detected problematic short response: '{generated_text.strip()}', trying regeneration")
                        # Try to get a better response by sampling differently
                        better_response = self.model.create_completion(
                            prompt_text,
                            max_tokens=min(1024, max_response_tokens),
                            temperature=0.9,  # Higher temperature for more creativity
                            top_p=0.95,
                            stream=False,
                            echo=False,
                            stop=stop_sequences,
                            repeat_penalty=1.2,
                            top_k=50,
                            frequency_penalty=0.1
                        )
                        
                        if better_response and 'choices' in better_response and len(better_response['choices']) > 0:
                            better_text = better_response['choices'][0]['text'].strip()
                            if len(better_text) > len(generated_text.strip()) and better_text.lower() not in ['sure', 'sure.', 'ok', 'okay', 'yes']:
                                if DEBUG >= 1:
                                    print(f"[{request_id}] Using better regenerated response: '{better_text[:100]}...'")
                                generated_text = better_text
                    
                    # Tokenize the cleaned text
                    generated_tokens = self.model.tokenize(generated_text.encode('utf-8'), add_bos=False, special=False)
                    
                    # Handle EOS tokens more carefully
                    # Get the proper EOS token for this model
                    model_eos_token = self.tokenizer.eos_token_id
                    common_eos_tokens = [2, 128001, 151645, model_eos_token]
                    # Remove duplicates while preserving order
                    common_eos_tokens = list(dict.fromkeys(common_eos_tokens))
                    
                    has_eos = generated_tokens and any(generated_tokens[-1] == eos for eos in common_eos_tokens)
                    
                    # Only add EOS if we don't have one and the response is complete
                    if not has_eos and generated_tokens:
                        # Check if this looks like a complete response
                        response_text = generated_text.strip()
                        looks_complete = (
                            len(response_text) > 20 and  # Not too short
                            (response_text.endswith('.') or response_text.endswith('!') or 
                             response_text.endswith('?') or response_text.endswith('"') or
                             response_text.endswith("'") or '\n\n' in response_text[-20:])
                        )
                        
                        if looks_complete or finish_reason == "stop":
                            generated_tokens.append(model_eos_token)
                            if DEBUG >= 1:
                                print(f"[{request_id}] Added EOS token {model_eos_token} to complete response")
                    elif not generated_tokens:
                        # Fallback for empty generation
                        generated_tokens = [model_eos_token]
                        if DEBUG >= 1:
                            print(f"[{request_id}] No tokens generated, using EOS only")
                    
                    accumulated_tokens = generated_tokens
                    if DEBUG >= 1:
                        print(f"[{request_id}] Split into {len(accumulated_tokens)} tokens (including EOS)")
                else:
                    # No generation, mark complete
                    generation_complete = True
                    accumulated_tokens = [self.tokenizer.eos_token_id]
            
            # Return next token from accumulated tokens
            if accumulated_tokens:
                next_token = accumulated_tokens.pop(0)
                
                # Check if this is an EOS token or the last token
                # Handle multiple potential EOS token IDs
                common_eos_tokens = [2, 128001, 151645, self.tokenizer.eos_token_id]
                is_eos = any(next_token == eos for eos in common_eos_tokens)
                
                if is_eos or not accumulated_tokens:
                    generation_complete = True
                    if DEBUG >= 1:
                        if is_eos:
                            print(f"[{request_id}] EOS token {next_token} detected, marking generation complete")
                        else:
                            print(f"[{request_id}] Last token, marking generation complete")
                
                # Update state
                new_state = inference_state.copy()
                new_state['generation_state'] = {
                    'accumulated_tokens': accumulated_tokens,
                    'complete': generation_complete,
                    'prompt_processed': True  # Mark that we've processed the prompt
                }
                new_state['n_past'] = len(tokens) + 1
                
                # Create logits for this token
                vocab_size = self.model.n_vocab()
                logits = np.full(vocab_size, -100.0, dtype=np.float32)
                logits[next_token] = 0.0
                
                if DEBUG >= 1:
                    print(f"[{request_id}] Returning token {next_token}, {len(accumulated_tokens)} remaining")
                
                return np.array([logits], dtype=np.float32), new_state
            
            # Fallback to EOS
            if DEBUG >= 1:
                print(f"[{request_id}] Fallback to EOS token")
            
            vocab_size = self.model.n_vocab()
            logits = np.full(vocab_size, -100.0, dtype=np.float32)
            logits[self.tokenizer.eos_token_id] = 0.0
            
            new_state = inference_state.copy()
            new_state['generation_state'] = {'accumulated_tokens': [], 'complete': True, 'prompt_processed': True}
            
            return np.array([logits], dtype=np.float32), new_state
            
        except Exception as e:
            if DEBUG >= 1:
                print(f"Error in generation: {e}")
            
            # Safe fallback
            try:
                vocab_size = self.model.n_vocab()
                eos_id = self.tokenizer.eos_token_id
            except:
                vocab_size = 32000
                eos_id = 2
                
            logits = np.full(vocab_size, -100.0, dtype=np.float32)
            logits[eos_id] = 0.0
            
            new_state = inference_state.copy()
            new_state['generation_state'] = {'accumulated_tokens': [], 'complete': True, 'prompt_processed': True}
            
            return np.array([logits], dtype=np.float32), new_state

    async def _process_logits(self, logits: np.ndarray, inference_state: Optional[dict]) -> tuple[np.ndarray, Optional[dict]]:
        """Process logits and return them as-is for sampling"""
        return logits, inference_state

    def _get_inference_state(self, request_id: str) -> dict:
        """Get or create inference state for a request"""
        if request_id not in self.inference_state_cache:
            self.inference_state_cache[request_id] = {
                'n_past': 0,
                'reset': True,
                'generation_state': {
                    'accumulated_tokens': [],
                    'complete': False,
                    'prompt_processed': False  # Track if prompt has been processed
                }
            }
            # Limit cache size but be more conservative
            if len(self.inference_state_cache) > 20:  # Increased cache size
                # Remove oldest entries
                oldest_key = next(iter(self.inference_state_cache))
                del self.inference_state_cache[oldest_key]
        
        return self.inference_state_cache[request_id]

    def _update_inference_state(self, request_id: str, state: dict):
        """Update inference state for a request"""
        self.inference_state_cache[request_id] = state

    async def load_checkpoint(self, shard: Shard, path: str):
        """Load model checkpoint - not directly supported in llama.cpp"""
        if DEBUG >= 1:
            print("Checkpoint loading not supported for LlamaCpp engine")
        pass

    async def save_checkpoint(self, shard: Shard, path: str):
        """Save model checkpoint - not directly supported in llama.cpp"""
        if DEBUG >= 1:
            print("Checkpoint saving not supported for LlamaCpp engine")
        pass

    async def ensure_shard(self, shard: Shard):
        """Ensure the model shard is loaded"""
        async with self._shard_lock:
            if self.shard == shard and self.model is not None:
                return
            
            if DEBUG >= 1:
                print(f"Loading shard: {shard}")
            
            # Download the model if needed
            model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
            
            if DEBUG >= 1:
                print(f"Model path: {model_path}")
            
            # Find GGUF file in the model directory
            gguf_path = await self._find_gguf_file(model_path)
            if not gguf_path:
                raise FileNotFoundError(f"No GGUF file found in {model_path}")
            
            # Load the model
            await self._load_model(gguf_path, shard)
            
            self.shard = shard

    async def _find_gguf_file(self, model_path) -> Optional[str]:
        """Find the GGUF file in the model directory or use direct file path"""
        def find_gguf():
            # Convert Path object to string if needed
            model_path_str = str(model_path)
            
            # Check if this is a direct path to a GGUF file
            if model_path_str.endswith('.gguf'):
                # This might be a direct file path like "repo/file.gguf"
                # Check if it exists as-is first
                if os.path.isfile(model_path_str):
                    return model_path_str
                
                # If not, it might be a repo/file pattern for HuggingFace
                # The shard downloader should handle this and provide a local path
                # For now, we'll assume the downloader has placed it correctly
                return model_path_str
            
            # Original logic for directory-based discovery
            if os.path.isfile(model_path_str) and model_path_str.endswith('.gguf'):
                return model_path_str
            if os.path.isdir(model_path_str):
                # Look for specific GGUF files or just take the first one
                gguf_files = [f for f in os.listdir(model_path_str) if f.endswith('.gguf')]
                if gguf_files:
                    return os.path.join(model_path_str, gguf_files[0])
            
            return None
        
        return await asyncio.get_running_loop().run_in_executor(
            self._llama_thread,
            find_gguf
        )

    async def _load_model(self, gguf_path: str, shard: Shard):
        """Load the GGUF model with optimal quantization settings"""
        def load_model():
            # Windows-specific: Set LLAMA_CPP_LIB environment variable if not set
            if platform.system().lower() == "windows":
                self._setup_windows_gpu_environment()
            
            params = self.default_params.copy()
            
            # Adjust parameters based on shard or model requirements
            if hasattr(shard, 'n_ctx'):
                params['n_ctx'] = shard.n_ctx
            
            # Set appropriate chat format based on model
            chat_format = self._detect_chat_format(gguf_path, shard)
            if chat_format:
                params['chat_format'] = chat_format
            
            # Detect and optimize for different quantization levels
            self._optimize_for_quantization(gguf_path, params)
            
            # Get model file size for intelligent GPU layer allocation
            model_size = os.path.getsize(gguf_path)
            model_size_gb = model_size / (1024**3)
            
            # Optimized GPU memory allocation strategy
            try:
                # Check for GPU support and warn user if missing
                if not self.gpu_offload_available and DEBUG >= 1:
                    print("🚨 WARNING: GPU acceleration is not available!")
                    print("   This will significantly reduce inference speed.")
                    if platform.system().lower() == "windows":
                        print("   Run fix_llamacpp_gpu.bat or fix_llamacpp_gpu.ps1 to enable GPU support.")
                    else:
                        print("   Run ./fix_llamacpp_gpu.sh to enable GPU support.")
                
                # Always attempt GPU acceleration for better performance
                gpu_layers = self._detect_gpu_support(model_size_gb)
                if gpu_layers is not None:
                    params['n_gpu_layers'] = gpu_layers
                    
                    if DEBUG >= 1:
                        if gpu_layers > 0:
                            print(f"Setting n_gpu_layers to {gpu_layers} ({'all layers' if gpu_layers == -1 else f'{gpu_layers} layers'})")
                        else:
                            print("Setting n_gpu_layers to 0 (CPU-only mode)")
                
                # Try memory manager for additional optimizations
                if should_prioritize_gpu():
                    memory_config = get_memory_config_for_model(gguf_path)
                    gpu_config = get_gpu_optimization_config()
                    
                    if DEBUG >= 1:
                        print(f"Memory manager config: {memory_config}")
                        print(f"GPU optimization config: {gpu_config}")
                    
                    # Override GPU layers if memory manager provides better estimate
                    if memory_config.get('use_gpu', False):
                        memory_gpu_layers = memory_config.get('gpu_layers', gpu_layers)
                        if memory_gpu_layers != gpu_layers and memory_gpu_layers > 0:
                            if DEBUG >= 1:
                                print(f"Memory manager recommends {memory_gpu_layers} GPU layers vs our calculation of {gpu_layers}")
                            # Use the more conservative estimate
                            if memory_gpu_layers < gpu_layers or gpu_layers == -1:
                                params['n_gpu_layers'] = memory_gpu_layers
                        
                        # Apply GPU-specific optimizations
                        if gpu_config.get('metal_gpu', False):  # macOS Metal
                            params['use_mmap'] = True  # Use memory mapping for Metal
                            params['use_mlock'] = False  # Don't lock in system memory
                            params['low_vram'] = False  # Metal can handle larger models
                        elif gpu_config.get('cuda_gpu', False):  # CUDA
                            params['use_mmap'] = True
                            # Set low_vram based on our layer calculation
                            params['low_vram'] = (params['n_gpu_layers'] > 0 and params['n_gpu_layers'] < 20)
                            
                        # Apply memory fraction settings for batch size optimization
                        if 'gpu_memory_fraction' in gpu_config:
                            gpu_fraction = gpu_config['gpu_memory_fraction']
                            if gpu_fraction < 0.5:  # Limited GPU memory
                                params['n_batch'] = min(params.get('n_batch', 1024), 512)
                                params['n_ubatch'] = min(params.get('n_ubatch', 512), 256)
                            elif gpu_fraction > 0.8:  # Plenty of GPU memory
                                params['n_batch'] = min(params.get('n_batch', 1024), 2048)
                                params['n_ubatch'] = min(params.get('n_ubatch', 512), 1024)
                                
            except (ImportError, RuntimeError) as e:
                if DEBUG >= 1:
                    print(f"Memory manager not available, using direct GPU detection: {e}")
                # Use our improved GPU detection
                gpu_layers = self._detect_gpu_support(model_size_gb)
                if gpu_layers is not None:
                    params['n_gpu_layers'] = gpu_layers
            
            # Set optimal threading based on system
            if params['n_threads'] is None:
                params['n_threads'] = min(os.cpu_count() or 1, 16)
            
            # Configure memory optimization for quantized models
            self._configure_memory_optimization(gguf_path, params)
            
            if DEBUG >= 1:
                print(f"Loading GGUF model: {gguf_path}")
                print(f"Model parameters: {params}")
            
            model = Llama(model_path=gguf_path, **params)
            
            # Store GPU layers parameter for tracking
            model._n_gpu_layers = params.get('n_gpu_layers', 0)
            
            # Log model info after loading
            if DEBUG >= 1:
                self._log_model_info(model, gguf_path)
            
            # Validate GPU inference is working if GPU layers were specified
            if params.get('n_gpu_layers', 0) > 0:
                self._validate_gpu_inference(model, params.get('n_gpu_layers', 0))
            
            return model
        
        self.model = await asyncio.get_running_loop().run_in_executor(
            self._llama_thread,
            load_model
        )
        
        # Create tokenizer wrapper
        self.tokenizer = LlamaTokenizerWrapper(self.model)
        
        if DEBUG >= 1:
            print(f"GGUF model loaded successfully: {gguf_path}")
            print(f"Model tokenizer EOS token ID: {self.tokenizer.eos_token_id}")

    def _detect_chat_format(self, gguf_path: str, shard: Shard) -> Optional[str]:
        """Detect appropriate chat format based on model name/path"""
        filename = os.path.basename(gguf_path).lower()
        model_id = getattr(shard, 'model_id', '').lower()
        
        # Qwen models
        if 'qwen' in filename or 'qwen' in model_id:
            if DEBUG >= 1:
                print("Detected Qwen model, using chatml chat format")
            return "chatml"
        
        # Llama models
        elif 'llama' in filename or 'llama' in model_id:
            if DEBUG >= 1:
                print("Detected Llama model, using llama-2 chat format")
            return "llama-2"
        
        # Mistral models
        elif 'mistral' in filename or 'mistral' in model_id:
            if DEBUG >= 1:
                print("Detected Mistral model, using mistral-instruct chat format")
            return "mistral-instruct"
        
        # Default to chatml for unknown models
        if DEBUG >= 1:
            print("Unknown model type, using chatml chat format as default")
        return "chatml"

    def _optimize_for_quantization(self, gguf_path: str, params: dict):
        """Optimize parameters based on quantization type detected in filename"""
        filename = os.path.basename(gguf_path).lower()
        
        # Detect quantization type from filename
        if any(q in filename for q in ['q2_k', 'q3_k', 'q4_k']):
            # Low precision quantizations - optimize for memory
            params['f16_kv'] = True
            params['low_vram'] = True
            if DEBUG >= 1:
                print("Detected low-precision quantization, optimizing for memory")
        elif any(q in filename for q in ['q5_k', 'q6_k', 'q8_0']):
            # Higher precision quantizations - balance performance and memory
            params['f16_kv'] = True
            params['low_vram'] = False
            if DEBUG >= 1:
                print("Detected high-precision quantization, balancing performance and memory")
        elif 'f16' in filename or 'f32' in filename:
            # Full precision models
            params['f16_kv'] = True
            params['low_vram'] = False
            if DEBUG >= 1:
                print("Detected full-precision model")

    def _detect_gpu_support(self, model_size_gb: float = 0) -> Optional[int]:
        """Detect GPU support and return optimal number of layers to offload based on available VRAM"""
        
        # First check if llama-cpp-python was compiled with GPU support
        if not self.gpu_offload_available:
            if DEBUG >= 1:
                print("ERROR: llama-cpp-python compiled without GPU support - forcing CPU-only mode")
                print("To enable GPU acceleration:")
                if platform.system().lower() == "windows":
                    print("  Windows: Run fix_llamacpp_gpu.bat or fix_llamacpp_gpu.ps1")
                else:
                    print("  Linux/macOS: Run ./fix_llamacpp_gpu.sh")
            return 0
        
        try:
            # Try CUDA first - especially important on Windows
            torch_available = False
            try:
                import torch
                torch_available = True
                if DEBUG >= 1:
                    print(f"PyTorch available: {torch.__version__}")
            except ImportError:
                if DEBUG >= 1:
                    print("PyTorch not available for GPU detection")
                torch_available = False
            
            if torch_available and torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    # Get GPU memory and name
                    gpu_props = torch.cuda.get_device_properties(0)
                    total_memory = gpu_props.total_memory
                    gpu_name = gpu_props.name.upper()
                    memory_gb = total_memory // (1024**3)
                    
                    # Check for RTX 50 series support with PyTorch 2.6+
                    if "RTX 50" in gpu_name or "5070" in gpu_name or "5080" in gpu_name or "5090" in gpu_name:
                        torch_version = torch.__version__.split('.')
                        major, minor = int(torch_version[0]), int(torch_version[1])
                        if major >= 2 and minor >= 6:
                            if DEBUG >= 1:
                                print(f"✅ RTX 50 series detected with PyTorch {torch.__version__} (supports Blackwell)")
                        else:
                            if DEBUG >= 1:
                                print(f"⚠️ RTX 50 series detected but PyTorch {torch.__version__} may not fully support it")
                                print("   Consider upgrading: pip install torch>=2.6 --index-url https://download.pytorch.org/whl/cu124")
                    
                    # Calculate available VRAM with optimized Windows reservations for high-end GPUs
                    if platform.system().lower() == "windows":
                        # RTX 5070 Ti and other high-end GPUs need less conservative reservations
                        if memory_gb >= 16:  # RTX 5070 Ti, RTX 4090, etc.
                            reserved_gb = 1.5  # Fixed 1.5GB for high VRAM cards
                        elif memory_gb >= 12:  # RTX 4070 Ti, RTX 3080, etc.
                            reserved_gb = 1.2  # Fixed 1.2GB for mid-high VRAM cards
                        else:
                            reserved_gb = max(1.0, memory_gb * 0.08)  # 8% for smaller cards
                    else:
                        reserved_gb = 1.0  # Reserve 1GB on Linux/macOS
                    available_vram_gb = memory_gb - reserved_gb
                    
                    if DEBUG >= 1:
                        print(f"CUDA GPU detected: {gpu_name} with {memory_gb}GB total ({available_vram_gb:.1f}GB available)")
                        if platform.system().lower() == "windows":
                            print(f"Windows detected - reserving {reserved_gb:.1f}GB for system overhead")
                    
                    # Calculate optimal layer offloading based on model size and available VRAM
                    if model_size_gb > 0:
                        optimal_layers = self._calculate_optimal_gpu_layers(model_size_gb, available_vram_gb)
                        if DEBUG >= 1:
                            print(f"Model size: {model_size_gb:.1f}GB, Available VRAM: {available_vram_gb:.1f}GB")
                            print(f"Optimal GPU layers: {optimal_layers} ({'all layers' if optimal_layers == -1 else f'{optimal_layers} layers'})")
                        return optimal_layers
                    else:
                        # If model size unknown, use conservative approach based on VRAM
                        if available_vram_gb >= 8:
                            return -1  # All layers for high VRAM
                        elif available_vram_gb >= 4:
                            return 32  # Most layers for medium VRAM
                        elif available_vram_gb >= 2:
                            return 16  # Some layers for low VRAM
                        else:
                            return 8   # Few layers for very low VRAM
                        
        except Exception as e:
            if DEBUG >= 1:
                print(f"Error during CUDA detection: {e}")
        
        # Windows fallback: try direct GPU detection without PyTorch
        if platform.system().lower() == "windows":
            try:
                # Try nvidia-ml-py for Windows GPU detection
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_name_raw = pynvml.nvmlDeviceGetName(handle)
                    gpu_name = gpu_name_raw.decode('utf-8') if isinstance(gpu_name_raw, bytes) else str(gpu_name_raw)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_memory_gb = memory_info.total / (1024**3)
                    
                    # Windows-specific VRAM calculation - optimized for high-end GPUs
                    if total_memory_gb >= 16:  # RTX 5070 Ti, RTX 4090, etc.
                        reserved_gb = 1.5  # Fixed 1.5GB for high VRAM cards
                    elif total_memory_gb >= 12:  # RTX 4070 Ti, RTX 3080, etc.
                        reserved_gb = 1.2  # Fixed 1.2GB for mid-high VRAM cards
                    else:
                        reserved_gb = max(1.0, total_memory_gb * 0.1)  # 10% for smaller cards
                    available_vram_gb = total_memory_gb - reserved_gb
                    
                    if DEBUG >= 1:
                        print(f"Windows GPU detected via NVML: {gpu_name}")
                        print(f"Total VRAM: {total_memory_gb:.1f}GB, Available: {available_vram_gb:.1f}GB")
                        print(f"Reserved {reserved_gb:.1f}GB for Windows system overhead")
                    
                    if model_size_gb > 0:
                        optimal_layers = self._calculate_optimal_gpu_layers(model_size_gb, available_vram_gb)
                        return optimal_layers
                    else:
                        # Optimized Windows defaults for RTX 5070 Ti and high-end GPUs
                        if available_vram_gb >= 12:  # RTX 5070 Ti, RTX 4090, etc.
                            return -1  # All layers for high VRAM cards
                        elif available_vram_gb >= 8:   # RTX 4070 Ti, RTX 3080, etc.
                            return -1  # All layers for good VRAM cards
                        elif available_vram_gb >= 4:   # RTX 3070, RTX 4060 Ti, etc.
                            return 32  # Most layers for medium VRAM
                        else:
                            return 16  # Some layers for low VRAM
                            
            except ImportError:
                if DEBUG >= 1:
                    print("pynvml not available for Windows GPU detection")
            except Exception as e:
                if DEBUG >= 1:
                    print(f"Error in Windows GPU detection: {e}")
        
        try:
            # Try Metal (macOS) - detect unified memory
            if platform.system() == "Darwin":
                try:
                    import psutil
                    total_memory_gb = psutil.virtual_memory().total / (1024**3)
                    # On macOS with Metal, we can use more aggressive GPU offloading
                    # since it shares unified memory
                    if total_memory_gb >= 16:
                        if DEBUG >= 1:
                            print(f"macOS Metal detected with {total_memory_gb:.1f}GB unified memory - enabling full GPU acceleration")
                        return -1  # All layers for high memory
                    elif total_memory_gb >= 8:
                        if DEBUG >= 1:
                            print(f"macOS Metal detected with {total_memory_gb:.1f}GB unified memory - enabling most layers on GPU")
                        return 28  # Most layers
                    else:
                        if DEBUG >= 1:
                            print(f"macOS Metal detected with {total_memory_gb:.1f}GB unified memory - enabling some layers on GPU")
                        return 16  # Some layers
                except ImportError:
                    if DEBUG >= 1:
                        print("macOS Metal detected, enabling GPU acceleration")
                    return -1  # Default to all layers
        except Exception as e:
            if DEBUG >= 1:
                print(f"Error during Metal detection: {e}")
        
        # Fallback to CPU
        if DEBUG >= 1:
            print("No GPU acceleration detected, using CPU")
        return 0

    def _calculate_optimal_gpu_layers(self, model_size_gb: float, available_vram_gb: float) -> int:
        """Calculate optimal number of layers to offload to GPU based on model and VRAM size"""
        
        if DEBUG >= 1:
            print(f"Calculating GPU layers: Model={model_size_gb:.1f}GB, VRAM={available_vram_gb:.1f}GB")
        
        # Always try to use VRAM, even if model is larger - hybrid GPU/RAM approach
        if model_size_gb * 1.1 <= available_vram_gb:  # 10% headroom for KV cache
            if DEBUG >= 1:
                print("Model fits in VRAM with headroom - using all layers on GPU")
            return -1  # All layers
        
        # Model is larger than VRAM - calculate hybrid offloading
        if model_size_gb > available_vram_gb:
            # Use more aggressive VRAM utilization - fill VRAM as much as possible
            # For high VRAM cards (16GB+), use even more aggressive allocation
            if available_vram_gb >= 14:  # RTX 5070 Ti territory
                usable_vram = available_vram_gb * 0.95  # Use 95% of available VRAM
            else:
                usable_vram = available_vram_gb * 0.9   # Use 90% of available VRAM
            layer_ratio = usable_vram / model_size_gb
            
            # Estimate total layers based on model size
            if model_size_gb <= 4:    # Small models (3B-7B params)
                total_layers = 32
            elif model_size_gb <= 8:  # Medium models (7B-13B params) 
                total_layers = 40
            elif model_size_gb <= 16: # Large models (13B-30B params)
                total_layers = 60
            elif model_size_gb <= 32: # Very large models (30B-70B params)
                total_layers = 80
            else:                     # Extremely large models (70B+ params)
                total_layers = 120
            
            # Calculate layers that fit in VRAM - be aggressive
            optimal_layers = max(8, int(total_layers * layer_ratio))  # At least 8 layers on GPU
            optimal_layers = min(optimal_layers, total_layers - 4)    # Leave at least 4 layers for CPU
            
            if DEBUG >= 1:
                vram_usage_gb = model_size_gb * (optimal_layers / total_layers)
                print(f"Hybrid mode: {optimal_layers}/{total_layers} layers on GPU ({vram_usage_gb:.1f}GB)")
                print(f"Remaining {total_layers - optimal_layers} layers will use system RAM")
            
            return optimal_layers
        
        # Edge case: very small VRAM
        if available_vram_gb < 2:
            if DEBUG >= 1:
                print("Very low VRAM - using minimal GPU layers")
            return 4  # Use at least some GPU acceleration
        
        # Default: use all layers if we get here
        if DEBUG >= 1:
            print("Default case - using all layers on GPU")
        return -1

    def _configure_memory_optimization(self, gguf_path: str, params: dict):
        """Configure memory optimization based on model size and system resources"""
        try:
            # Get model file size
            model_size = os.path.getsize(gguf_path)
            model_size_gb = model_size / (1024**3)
            
            # Get system memory
            try:
                import psutil
                system_memory_gb = psutil.virtual_memory().total / (1024**3)
            except ImportError:
                if DEBUG >= 1:
                    print("psutil not available, using default memory settings")
                return
            
            if DEBUG >= 1:
                print(f"Model size: {model_size_gb:.1f}GB, System memory: {system_memory_gb:.1f}GB")
            
            # Adjust settings based on whether using GPU
            gpu_layers = params.get('n_gpu_layers', 0)
            using_gpu = gpu_layers > 0
            
            if using_gpu:
                # GPU-optimized settings
                params['use_mmap'] = True  # Always use mmap with GPU
                params['use_mlock'] = False  # Don't lock system memory when using GPU
                
                # Check if context size would exceed available VRAM and configure accordingly
                self._configure_context_memory(params, model_size_gb)
                
                # Adjust low_vram based on number of GPU layers
                if gpu_layers == -1:  # All layers on GPU
                    params['low_vram'] = False
                elif gpu_layers < 20:  # Partial GPU offload
                    params['low_vram'] = True
                else:  # Most layers on GPU
                    params['low_vram'] = False
                    
                if DEBUG >= 1:
                    print(f"GPU mode: {gpu_layers} layers, low_vram: {params['low_vram']}")
            else:
                # CPU-only settings - configure based on memory ratio
                memory_ratio = model_size_gb / system_memory_gb
                
                if memory_ratio > 0.7:
                    # Large model relative to system memory
                    params['use_mlock'] = False
                    params['low_vram'] = True
                    params['n_batch'] = min(params.get('n_batch', 1024), 256)  # Reduce batch size
                    if DEBUG >= 1:
                        print("CPU mode: Large model detected, optimizing for low memory usage")
                
                elif memory_ratio > 0.3:
                    # Medium model
                    params['use_mlock'] = False
                    params['low_vram'] = False
                    params['n_batch'] = min(params.get('n_batch', 1024), 512)
                    if DEBUG >= 1:
                        print("CPU mode: Medium model detected, using balanced settings")
                
                else:
                    # Small model relative to system memory
                    params['use_mlock'] = True  # Lock in memory for speed
                    params['low_vram'] = False
                    if DEBUG >= 1:
                        print("CPU mode: Small model detected, optimizing for speed")
                    
        except Exception as e:
            if DEBUG >= 1:
                print(f"Could not optimize memory settings: {e}")

    def _configure_context_memory(self, params: dict, model_size_gb: float):
        """Configure context memory allocation between GPU and system RAM"""
        try:
            import torch
            if not torch.cuda.is_available():
                if DEBUG >= 1:
                    print("CUDA not available, using CPU-only settings")
                params['offload_kqv'] = False
                params['n_ctx'] = min(params.get('n_ctx', 8192), 8192)  # Limit to 8k for CPU
                return
                
            # Get GPU memory info
            device = torch.cuda.current_device()
            gpu_props = torch.cuda.get_device_properties(device)
            total_gpu_memory = gpu_props.total_memory
            total_gpu_gb = total_gpu_memory / (1024**3)
            
            # Start with current context size (default is now 8k)
            n_ctx = params.get('n_ctx', 8192)
            
            # Estimate KV cache size based on context length
            # KV cache size formula: 2 * n_layers * n_ctx * n_embd * bytes_per_element
            # For Qwen3-4B: 36 layers, 2560 embedding dimension, f16 (2 bytes)
            estimated_layers = 36
            estimated_embd = 2560
            bytes_per_element = 2  # f16
            kv_cache_size_bytes = 2 * estimated_layers * n_ctx * estimated_embd * bytes_per_element
            kv_cache_size_gb = kv_cache_size_bytes / (1024**3)
            
            # Conservative estimate of available GPU memory after model loading
            reserved_gb = 2.0 if total_gpu_gb <= 12 else 3.0  # More conservative for smaller GPUs
            available_gpu_gb = total_gpu_gb - model_size_gb - reserved_gb
            
            if DEBUG >= 1:
                print(f"Context memory analysis:")
                print(f"  Context size: {n_ctx}")
                print(f"  Estimated KV cache: {kv_cache_size_gb:.1f}GB")
                print(f"  Model size: {model_size_gb:.1f}GB") 
                print(f"  Total GPU memory: {total_gpu_gb:.1f}GB")
                print(f"  Available for KV cache: {available_gpu_gb:.1f}GB")
            
            # Try to find the largest context that fits
            if kv_cache_size_gb <= available_gpu_gb:
                # Current context fits in VRAM - can we go larger?
                params['offload_kqv'] = True
                
                # Try to increase context if there's room
                for test_ctx in [16384, 32768, 65536]:
                    if test_ctx <= n_ctx:
                        continue  # Already at or above this size
                    
                    test_kv_size = 2 * estimated_layers * test_ctx * estimated_embd * bytes_per_element / (1024**3)
                    if test_kv_size <= available_gpu_gb:
                        params['n_ctx'] = test_ctx
                        if DEBUG >= 1:
                            print(f"  Increasing context to {test_ctx} (fits in VRAM)")
                        break
                
                if DEBUG >= 1:
                    print(f"  Using GPU for KV cache with context {params['n_ctx']}")
                    
            else:
                # KV cache too large - use system RAM and reduce context
                params['offload_kqv'] = False
                
                if DEBUG >= 1:
                    print(f"  KV cache ({kv_cache_size_gb:.1f}GB) > available VRAM ({available_gpu_gb:.1f}GB)")
                    print(f"  Using system RAM for KV cache")
                
                # Find the largest context that doesn't use excessive system RAM
                max_ram_gb = 16  # Don't use more than 16GB of system RAM for KV cache
                
                for test_ctx in [4096, 8192, 16384, 24576]:
                    test_kv_size = 2 * estimated_layers * test_ctx * estimated_embd * bytes_per_element / (1024**3)
                    if test_kv_size <= max_ram_gb:
                        params['n_ctx'] = test_ctx
                    else:
                        break
                
                if DEBUG >= 1:
                    final_kv_size = 2 * estimated_layers * params['n_ctx'] * estimated_embd * bytes_per_element / (1024**3)
                    print(f"  Reduced context to {params['n_ctx']} (KV cache: {final_kv_size:.1f}GB in RAM)")
                    
        except Exception as e:
            if DEBUG >= 1:
                print(f"Could not configure context memory: {e}")
                print("  Using safe fallback settings")
            # Safe fallback
            params['offload_kqv'] = False
            params['n_ctx'] = 4096

    def _log_model_info(self, model, gguf_path: str):
        """Log information about the loaded model including GPU utilization"""
        try:
            # Basic model info
            print(f"Model loaded from: {gguf_path}")
            
            # Try to get model metadata if available
            if hasattr(model, 'metadata'):
                metadata = model.metadata
                if 'general.name' in metadata:
                    print(f"Model name: {metadata['general.name']}")
                if 'general.architecture' in metadata:
                    print(f"Architecture: {metadata['general.architecture']}")
                if 'general.parameter_count' in metadata:
                    params = metadata['general.parameter_count']
                    print(f"Parameters: {params / 1e9:.1f}B")
            
            # Context and quantization info
            print(f"Context length: {model.n_ctx()}")
            print(f"Vocabulary size: {model.n_vocab()}")
            
            # GPU utilization info
            self._log_gpu_utilization(model)
            
        except Exception as e:
            if DEBUG >= 2:
                print(f"Could not log model info: {e}")

    def _log_gpu_utilization(self, model):
        """Log GPU utilization and memory usage"""
        try:
            # Try to get GPU layer count from llama.cpp model
            if hasattr(model, '_model') and hasattr(model._model, 'n_gpu_layers'):
                gpu_layers = model._model.n_gpu_layers
            else:
                # Fallback: check the parameters we used
                gpu_layers = getattr(model, '_n_gpu_layers', 'unknown')
            
            print(f"GPU layers: {gpu_layers} ({'all layers' if gpu_layers == -1 else f'{gpu_layers} layers' if isinstance(gpu_layers, int) else gpu_layers})")
            
            # Try to get actual GPU memory usage
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    
                    print(f"GPU Memory - Allocated: {gpu_memory_allocated:.1f}GB, Reserved: {gpu_memory_reserved:.1f}GB, Total: {gpu_memory_total:.1f}GB")
                    print(f"GPU Memory Usage: {(gpu_memory_reserved/gpu_memory_total)*100:.1f}% of total VRAM")
                    
                    if gpu_memory_reserved > 0:
                        print("SUCCESS: Model successfully loaded to GPU!")
                    elif gpu_layers > 0:
                        print("WARNING:  GPU layers specified but no VRAM usage detected")
                    else:
                        print("INFO: Model running on CPU only")
            except ImportError:
                if DEBUG >= 1:
                    print("PyTorch not available for GPU memory reporting")
            except Exception as e:
                if DEBUG >= 2:
                    print(f"Could not get GPU memory info: {e}")
                    
        except Exception as e:
            if DEBUG >= 2:
                print(f"Could not log GPU utilization: {e}")

    def _validate_gpu_inference(self, model, gpu_layers: int):
        """Validate that GPU inference is actually working"""
        try:
            if DEBUG >= 1:
                print(f"Validating GPU inference with {gpu_layers} layers...")
                
            # Additional Windows validation
            if platform.system().lower() == "windows":
                self._windows_gpu_diagnostic()
                
        except Exception as e:
            if DEBUG >= 1:
                print(f"Error in GPU validation setup: {e}")
                
        try:
            
            # Try a simple inference to ensure GPU is being used
            test_prompt = "Test"
            test_tokens = model.tokenize(test_prompt.encode('utf-8'), add_bos=True, special=False)
            
            # Check GPU memory before inference
            gpu_memory_before = 0
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_before = torch.cuda.memory_allocated(0)
            except:
                pass
            
            # Run a minimal completion
            try:
                result = model.create_completion(
                    test_prompt,
                    max_tokens=1,
                    temperature=0.0,
                    stream=False,
                    echo=False
                )
                
                # Check GPU memory after inference
                gpu_memory_after = 0
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory_after = torch.cuda.memory_allocated(0)
                except:
                    pass
                
                if result and 'choices' in result and len(result['choices']) > 0:
                    if DEBUG >= 1:
                        memory_change = (gpu_memory_after - gpu_memory_before) / (1024**2)  # MB
                        print(f"SUCCESS: GPU inference validation successful!")
                        if memory_change > 0:
                            print(f"GPU memory increased by {memory_change:.1f}MB during inference")
                        print(f"Test output: '{result['choices'][0]['text']}'")
                else:
                    if DEBUG >= 1:
                        print("WARNING: GPU inference validation: No output generated")
                        
            except Exception as e:
                if DEBUG >= 1:
                    print(f"WARNING: GPU inference validation failed: {e}")
                    print("This may indicate GPU layers are not being used properly")
                    
        except Exception as e:
            if DEBUG >= 1:
                print(f"Could not validate GPU inference: {e}")

    def _setup_windows_gpu_environment(self):
        """Setup Windows-specific GPU environment variables for proper offloading"""
        try:
            # Set LLAMA_CPP_LIB environment variable if not already set
            if not os.environ.get('LLAMA_CPP_LIB'):
                try:
                    import importlib.util
                    from pathlib import Path
                    
                    # Try CUDA version first
                    spec = None
                    try:
                        spec = importlib.util.find_spec("llama_cpp_cuda")
                    except ImportError:
                        pass
                    
                    # Fallback to regular llama_cpp
                    if not spec:
                        spec = importlib.util.find_spec("llama_cpp")
                    
                    if spec and spec.origin:
                        lib_path = Path(spec.origin).parent / "llama.dll"
                        if lib_path.exists():
                            os.environ['LLAMA_CPP_LIB'] = str(lib_path)
                            if DEBUG >= 1:
                                print(f"[Windows] Set LLAMA_CPP_LIB={lib_path}")
                        else:
                            if DEBUG >= 1:
                                print(f"[Windows] WARNING: llama.dll not found at {lib_path}")
                                
                except Exception as e:
                    if DEBUG >= 1:
                        print(f"[Windows] Could not set LLAMA_CPP_LIB: {e}")
            
            # Ensure CUDA environment variables are set
            if not os.environ.get('CUDA_PATH') and not os.environ.get('CUDA_HOME'):
                # Try to find CUDA installation
                possible_paths = [
                    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
                    r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA"
                ]
                
                for base_path in possible_paths:
                    if os.path.exists(base_path):
                        try:
                            versions = [d for d in os.listdir(base_path) if d.startswith('v')]
                            if versions:
                                latest_version = sorted(versions)[-1]
                                cuda_path = os.path.join(base_path, latest_version)
                                os.environ['CUDA_PATH'] = cuda_path
                                os.environ['CUDA_HOME'] = cuda_path
                                if DEBUG >= 1:
                                    print(f"[Windows] Set CUDA_PATH={cuda_path}")
                                break
                        except Exception:
                            continue
            
        except Exception as e:
            if DEBUG >= 1:
                print(f"[Windows] Error setting up GPU environment: {e}")

    def _windows_gpu_diagnostic(self):
        """Run Windows-specific GPU diagnostics with RTX 5070 Ti support"""
        try:
            if DEBUG >= 1:
                print("Running Windows GPU diagnostics...")
                
            # Check NVIDIA driver and CUDA
            try:
                import pynvml
                pynvml.nvmlInit()
                
                driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
                
                print(f"NVIDIA Driver: {driver_version}")
                print(f"CUDA Driver: {cuda_version // 1000}.{(cuda_version % 1000) // 10}")
                
                # Check driver requirements for latest GPUs
                try:
                    major_version = int(driver_version.split('.')[0])
                    if major_version >= 566:
                        print("Driver supports RTX 5070 Ti (Blackwell architecture)")
                    elif major_version >= 560:
                        print("Driver may support RTX 5070 Ti but newer version recommended")
                    else:
                        print("WARNING: Driver may be too old for RTX 5070 Ti (need 566.03+)")
                except:
                    print("Could not parse driver version for compatibility check")
                
                # Check GPU utilization
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    gpu_name_raw = pynvml.nvmlDeviceGetName(handle)
                    gpu_name = gpu_name_raw.decode('utf-8') if isinstance(gpu_name_raw, bytes) else str(gpu_name_raw)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    used_mb = memory_info.used / (1024**2)
                    total_mb = memory_info.total / (1024**2)
                    free_mb = memory_info.free / (1024**2)
                    
                    print(f"GPU {i}: {gpu_name}")
                    print(f"  Memory: {used_mb:.0f}MB used, {free_mb:.0f}MB free, {total_mb:.0f}MB total")
                    
                    # RTX 5070 Ti specific checks
                    if "5070" in gpu_name:
                        print("  RTX 5070 Ti detected!")
                        if total_mb >= 15000:  # ~16GB
                            print("  16GB GDDR7 memory confirmed")
                            # Check compute capability
                            try:
                                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                                print(f"  Compute capability: {major}.{minor}")
                                if major >= 9:
                                    print("  Blackwell architecture (9.0) confirmed")
                                else:
                                    print(f"  WARNING: Unexpected compute capability for RTX 5070 Ti")
                            except:
                                print("  Could not get compute capability")
                        else:
                            print("  WARNING: Unexpected VRAM amount for RTX 5070 Ti")
                    
                    # Check if GPU is being used
                    if used_mb > 100:  # More than 100MB indicates some usage
                        print(f"  Status: GPU memory in use - likely working!")
                    else:
                        print(f"  Status: Minimal GPU memory usage - may not be offloading")
                        if "5070" in gpu_name:
                            print("  RTX 5070 Ti ISSUE: GPU not being utilized!")
                            print("  Recommendation: Run fix_windows_rtx_5070_ti.ps1")
                        
            except ImportError:
                print("pynvml not available for detailed GPU diagnostics")
            except Exception as e:
                print(f"GPU diagnostic error: {e}")
                
            # Check llamacpp CUDA functions
            try:
                from llama_cpp import llama_cpp
                if hasattr(llama_cpp, 'llama_get_device_count'):
                    device_count = llama_cpp.llama_get_device_count()
                    print(f"llamacpp CUDA device count: {device_count}")
                else:
                    print("llamacpp CUDA functions not available")
            except Exception as e:
                print(f"llamacpp CUDA check error: {e}")
                
        except Exception as e:
            if DEBUG >= 1:
                print(f"Windows GPU diagnostic failed: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        if self._llama_thread:
            self._llama_thread.shutdown(wait=True)
        
        # Clear caches
        self.model_cache.clear()
        self.inference_state_cache.clear()
        
        # Reset model
        if self.model:
            try:
                await asyncio.get_running_loop().run_in_executor(
                    self._llama_thread,
                    lambda: setattr(self, 'model', None)
                )
            except:
                pass
