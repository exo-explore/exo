import ggml

# TODO: adjust size to fit
mem_size = int(1e9)
PARAMS = ggml.ggml.ggml_init_params(mem_size, mem_buffer=None)
