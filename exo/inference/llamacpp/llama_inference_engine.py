import numpy as np
import asyncio
import os
import time
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

from ..inference_engine import InferenceEngine
from ..shard import Shard
from exo.download.shard_download import ShardDownloader
from exo.helpers import DEBUG

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
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
        
        self.shard = None
        self.shard_downloader = shard_downloader
        self.model = None
        self.tokenizer = None
        self.session = {}
        self._llama_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llama")
        self._shard_lock = asyncio.Lock()
        
        # Model cache for multiple concurrent requests
        self.model_cache = OrderedDict()
        self.inference_state_cache = OrderedDict()
        
        # Default parameters optimized for GGUF models
        self.default_params = {
            "n_ctx": 4096,  # Context length
            "n_batch": 512,  # Batch size for prompt processing
            "n_threads": None,  # Auto-detect CPU threads
            "n_gpu_layers": -1,  # Offload all layers to GPU if available (-1 = auto)
            "use_mmap": True,  # Use memory mapping for better performance
            "use_mlock": False,  # Lock model in memory
            "f16_kv": True,  # Use half-precision for key/value cache
            "logits_all": False,  # Only compute logits for last token (streaming completion handles this)
            "vocab_only": False,  # Load full model
            "verbose": DEBUG >= 2,
            # Quantization-related parameters
            "low_vram": False,  # Reduce VRAM usage
            "numa": False,  # NUMA support
            "rope_scaling_type": None,  # RoPE scaling type
            "rope_freq_base": 0.0,  # RoPE frequency base
            "rope_freq_scale": 1.0,  # RoPE frequency scale
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
                
                # Generate complete response with proper stopping
                response = self.model.create_completion(
                    prompt_text,
                    max_tokens=50,  # Generate a reasonable chunk
                    temperature=0.7,
                    top_p=0.9,
                    stream=False,
                    echo=False,
                    stop=["<|im_end|>", "</s>", "\n\n", "Human:", "User:"],
                    repeat_penalty=1.1
                )
                
                generated_text = ""
                if response and 'choices' in response and len(response['choices']) > 0:
                    generated_text = response['choices'][0]['text'].strip()
                
                if DEBUG >= 1:
                    print(f"[{request_id}] Generated complete text: '{generated_text}'")
                
                # Convert to tokens for streaming
                if generated_text:
                    generated_tokens = self.model.tokenize(generated_text.encode('utf-8'), add_bos=False, special=False)
                    accumulated_tokens = generated_tokens
                    if DEBUG >= 1:
                        print(f"[{request_id}] Split into {len(accumulated_tokens)} tokens")
                else:
                    # No generation, mark complete
                    generation_complete = True
                    accumulated_tokens = [self.tokenizer.eos_token_id]
            
            # Return next token from accumulated tokens
            if accumulated_tokens:
                next_token = accumulated_tokens.pop(0)
                
                # Check if this is the last token
                if not accumulated_tokens:
                    generation_complete = True
                    if DEBUG >= 1:
                        print(f"[{request_id}] Last token, marking generation complete")
                
                # Update state
                new_state = inference_state.copy()
                new_state['generation_state'] = {
                    'accumulated_tokens': accumulated_tokens,
                    'complete': generation_complete
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
            new_state['generation_state'] = {'accumulated_tokens': [], 'complete': True}
            
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
            new_state['generation_state'] = {'accumulated_tokens': [], 'complete': True}
            
            return np.array([logits], dtype=np.float32), new_state

    async def _process_logits(self, logits: np.ndarray, inference_state: Optional[dict]) -> tuple[np.ndarray, Optional[dict]]:
        """Process logits and return them as-is for sampling"""
        return logits, inference_state

    def _get_inference_state(self, request_id: str) -> dict:
        """Get or create inference state for a request"""
        if request_id not in self.inference_state_cache:
            self.inference_state_cache[request_id] = {
                'n_past': 0,
                'reset': True
            }
            # Limit cache size
            if len(self.inference_state_cache) > 10:
                self.inference_state_cache.popitem(last=False)
        
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
                # This might be a direct file path like "repo/file.gguf"                # Check if it exists as-is first
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
            
            # Use GPU memory manager for optimal memory allocation
            try:
                from exo.inference.memory_manager import get_memory_config_for_model, should_prioritize_gpu, get_gpu_optimization_config
                
                if should_prioritize_gpu():
                    # Get memory configuration recommendations
                    memory_config = get_memory_config_for_model(gguf_path)
                    gpu_config = get_gpu_optimization_config()
                    
                    if DEBUG >= 1:
                        print(f"Memory manager config: {memory_config}")
                        print(f"GPU optimization config: {gpu_config}")
                    
                    # Apply memory manager recommendations
                    if memory_config.get('use_gpu', False):
                        # Prioritize GPU memory allocation
                        params['n_gpu_layers'] = memory_config.get('gpu_layers', -1)  # -1 = all layers
                        
                        # Apply GPU-specific optimizations
                        if gpu_config.get('metal_gpu', False):  # macOS Metal
                            params['use_mmap'] = True  # Use memory mapping for Metal
                            params['use_mlock'] = False  # Don't lock in system memory
                        elif gpu_config.get('cuda_gpu', False):  # CUDA
                            params['use_mmap'] = True
                            params['low_vram'] = not memory_config.get('sufficient_gpu_memory', True)
                            
                        # Apply memory fraction settings
                        if 'gpu_memory_fraction' in gpu_config:
                            # Note: llama.cpp doesn't have direct GPU memory fraction control
                            # but we can adjust batch size and context based on available memory
                            gpu_fraction = gpu_config['gpu_memory_fraction']
                            if gpu_fraction < 0.5:  # Limited GPU memory
                                params['n_batch'] = min(params.get('n_batch', 512), 256)
                                params['n_ctx'] = min(params.get('n_ctx', 4096), 2048)
                    else:
                        # Fallback to system memory
                        params['n_gpu_layers'] = memory_config.get('cpu_fallback_layers', 0)
                        if DEBUG >= 1:
                            print("GPU memory insufficient, using system memory fallback")
                else:
                    # Memory manager recommends CPU-only
                    params['n_gpu_layers'] = 0
                    if DEBUG >= 1:
                        print("Memory manager recommends CPU-only inference")
                        
            except (ImportError, RuntimeError) as e:
                if DEBUG >= 1:
                    print(f"Memory manager not available, using legacy GPU detection: {e}")
                # Fallback to original GPU detection logic
                gpu_layers = self._detect_gpu_support()
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
            
            # Log model info after loading
            if DEBUG >= 1:
                self._log_model_info(model, gguf_path)
            
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
            # Higher precision quantizations - balance performance and memory            params['f16_kv'] = True
            params['low_vram'] = False
            if DEBUG >= 1:
                print("Detected high-precision quantization, balancing performance and memory")
        
        elif 'f16' in filename or 'f32' in filename:
            # Full precision models
            params['f16_kv'] = True
            params['low_vram'] = False
            if DEBUG >= 1:
                print("Detected full-precision model")

    def _detect_gpu_support(self) -> Optional[int]:
        """Detect GPU support and return optimal number of layers to offload"""
        try:
            # Try CUDA first
            try:
                import torch
            except ImportError:
                torch = None
            
            if torch and torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    # Get GPU memory and name
                    gpu_props = torch.cuda.get_device_properties(0)
                    total_memory = gpu_props.total_memory
                    gpu_name = gpu_props.name.upper()
                    memory_gb = total_memory // (1024**3)
                    
                    if DEBUG >= 1:
                        print(f"CUDA GPU detected: {gpu_name} with {memory_gb}GB memory")
                    
                    # Optimize layer offloading for RTX 50 series (Blackwell architecture)
                    if "RTX 50" in gpu_name or "RTX 5090" in gpu_name or "RTX 5080" in gpu_name or "RTX 5070" in gpu_name:
                        if DEBUG >= 1:
                            print("RTX 50 series detected - enabling full GPU acceleration with optimizations")
                        return -1  # All layers - RTX 50 series has excellent AI performance
                    
                    # RTX 40 series optimization
                    elif "RTX 40" in gpu_name or "RTX 4090" in gpu_name or "RTX 4080" in gpu_name or "RTX 4070" in gpu_name:
                        if DEBUG >= 1:
                            print("RTX 40 series detected - enabling full GPU acceleration")
                        return -1  # All layers
                      # Memory-based fallback for other GPUs
                    elif memory_gb >= 12:  # High memory GPUs
                        if DEBUG >= 1:
                            print(f"High memory GPU detected ({memory_gb}GB), offloading all layers")
                        return -1  # All layers
                    
                    elif memory_gb >= 8:  # Medium memory GPUs
                        if DEBUG >= 1:
                            print(f"Medium memory GPU detected ({memory_gb}GB), offloading 32 layers")
                        return 32  # Most layers
                    
                    elif memory_gb >= 6:  # Lower memory GPUs
                        if DEBUG >= 1:
                            print(f"Lower memory GPU detected ({memory_gb}GB), offloading 20 layers")
                        return 20  # Partial offload
                    else:  # Very low memory GPUs
                        if DEBUG >= 1:
                            print(f"Low memory GPU detected ({memory_gb}GB), offloading 10 layers")
                        return 10  # Minimal offload
                        
        except Exception as e:
            if DEBUG >= 1:
                print(f"Error during CUDA detection: {e}")
        
        try:
            # Try Metal (macOS)
            import platform
            if platform.system() == "Darwin":
                if DEBUG >= 1:
                    print("macOS detected, enabling Metal GPU acceleration")
                return -1  # All layers for Metal
        except:
            pass
          # Fallback to CPU
        if DEBUG >= 1:
            print("No GPU acceleration detected, using CPU")
        return 0

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
            
            # Configure based on memory ratio
            memory_ratio = model_size_gb / system_memory_gb
            
            if memory_ratio > 0.7:
                # Large model relative to system memory
                params['use_mlock'] = False
                params['low_vram'] = True
                params['n_batch'] = 256  # Reduce batch size
                if DEBUG >= 1:
                    print("Large model detected, optimizing for low memory usage")
            
            elif memory_ratio > 0.3:
                # Medium model
                params['use_mlock'] = False
                params['low_vram'] = False
                params['n_batch'] = 512
                if DEBUG >= 1:
                    print("Medium model detected, using balanced settings")
            
            else:
                # Small model relative to system memory
                params['use_mlock'] = True  # Lock in memory for speed
                params['low_vram'] = False
                params['n_batch'] = 1024  # Larger batch for speed
                if DEBUG >= 1:
                    print("Small model detected, optimizing for speed")
                    
        except Exception as e:
            if DEBUG >= 1:
                print(f"Could not optimize memory settings: {e}")

    def _log_model_info(self, model, gguf_path: str):
        """Log information about the loaded model"""
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
            
        except Exception as e:
            if DEBUG >= 2:
                print(f"Could not log model info: {e}")

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