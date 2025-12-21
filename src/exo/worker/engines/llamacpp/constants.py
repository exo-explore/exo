"""
Constants for the llama.cpp engine.
"""

# Maximum tokens to generate if not specified
MAX_TOKENS = 4096

# Default context size for models (reduced for Android memory constraints)
DEFAULT_CONTEXT_SIZE = 2048

# Number of threads to use (0 = auto-detect)
DEFAULT_N_THREADS = 0

# GPU layers to offload (0 = CPU only, -1 = all layers)
DEFAULT_N_GPU_LAYERS = 0

# Default batch size for prompt processing
DEFAULT_N_BATCH = 512

# Temperature for sampling
DEFAULT_TEMPERATURE = 0.7

# Top-p (nucleus) sampling
DEFAULT_TOP_P = 0.9

