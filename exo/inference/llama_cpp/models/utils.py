import gguf
import ggml
import numpy as np

def load_weights(file_data, weight_name) -> np.ndarray:
    _, tensorinfo = gguf.load_gguf(file_data)
    numpy_tensor = gguf.load_gguf_tensor(file_data, tensorinfo, weight_name)
    return numpy_tensor
