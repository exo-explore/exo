import ggml

# TODO: adjust size to fit whatever GGUF model being used
# Q8 LLama 3.1 8b
mem_size = int(1e9) * 8

PARAMS = ggml.ggml.ggml_init_params(mem_size, mem_buffer=None)
