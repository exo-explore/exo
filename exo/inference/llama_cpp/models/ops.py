import ggml
import numpy as np

class Linear:
    def __init__(self, ctx: ggml.gguf_context_p):
        self.ctx = ctx
