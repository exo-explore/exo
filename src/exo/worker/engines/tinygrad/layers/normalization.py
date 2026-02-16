from tinygrad import Tensor

def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    variance = (x * x).mean(axis=-1, keepdim=True)
    x_normed = x * (variance + eps).rsqrt()

    return x_normed * weight
