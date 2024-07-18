import math
from typing import Optional, Union, Tuple
from tinygrad.tensor import Tensor
from tinygrad.helpers import prod
from tinygrad.nn import optim, state, datasets  # noqa: F401

class BatchNorm:
  """
  Applies Batch Normalization over a 2D or 3D input.

  - Described: https://paperswithcode.com/method/batch-normalization
  - Paper: https://arxiv.org/abs/1502.03167v3

  See: `Tensor.batchnorm`

  ```python exec="true" session="tensor"
  from tinygrad import Tensor, dtypes, nn
  import numpy as np
  np.set_printoptions(precision=4)
  ```

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.BatchNorm(3)
  t = Tensor.rand(2, 3, 4, 4)
  print(t.mean().item(), t.std().item())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = norm(t)
  print(t.mean().item(), t.std().item())
  ```
  """
  def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum

    if affine: self.weight, self.bias = Tensor.ones(sz), Tensor.zeros(sz)
    else: self.weight, self.bias = None, None

    self.running_mean, self.running_var = Tensor.zeros(sz, requires_grad=False), Tensor.ones(sz, requires_grad=False)
    self.num_batches_tracked = Tensor.zeros(1, requires_grad=False)

  def __call__(self, x:Tensor):
    shape_mask = [1, -1, *([1]*(x.ndim-2))]
    if Tensor.training:
      # This requires two full memory accesses to x
      # https://github.com/pytorch/pytorch/blob/c618dc13d2aa23625cb0d7ada694137532a4fa33/aten/src/ATen/native/cuda/Normalization.cuh
      # There's "online" algorithms that fix this, like https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
      batch_mean = x.mean(axis=(reduce_axes:=tuple(x for x in range(x.ndim) if x != 1)))
      y = (x - batch_mean.detach().reshape(shape=shape_mask))  # d(var)/d(mean) = 0
      batch_var = (y*y).mean(axis=reduce_axes)
      batch_invstd = batch_var.add(self.eps).pow(-0.5)

      # NOTE: wow, this is done all throughout training in most PyTorch models
      if self.track_running_stats:
        self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach())
        self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * prod(y.shape)/(prod(y.shape)-y.shape[1]) * batch_var.detach())
        self.num_batches_tracked += 1
    else:
      batch_mean = self.running_mean
      # NOTE: this can be precomputed for static inference. we expand it here so it fuses
      batch_invstd = self.running_var.reshape(shape=shape_mask).expand(x.shape).add(self.eps).rsqrt()
    return x.batchnorm(self.weight, self.bias, batch_mean, batch_invstd)
BatchNorm2d = BatchNorm3d = BatchNorm

def Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
  """
  Applies a 1D convolution over an input signal composed of several input planes.

  See: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d

  ```python exec="true" source="above" session="tensor" result="python"
  conv = nn.Conv1d(1, 1, 3)
  t = Tensor.rand(1, 1, 4)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = conv(t)
  print(t.numpy())
  ```
  """
  return Conv2d(in_channels, out_channels, (kernel_size,), stride, padding, dilation, groups, bias)

class Conv2d:
  """
  Applies a 2D convolution over an input signal composed of several input planes.

  See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d

  ```python exec="true" source="above" session="tensor" result="python"
  conv = nn.Conv2d(1, 1, 3)
  t = Tensor.rand(1, 1, 4, 4)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = conv(t)
  print(t.numpy())
  ```
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
    scale = 1 / math.sqrt(in_channels * prod(self.kernel_size))
    self.weight = Tensor.uniform(out_channels, in_channels//groups, *self.kernel_size, low=-scale, high=scale)
    self.bias = Tensor.uniform(out_channels, low=-scale, high=scale) if bias else None

  def __call__(self, x:Tensor):
    return x.conv2d(self.weight, self.bias, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)

def ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True):
  """
  Applies a 1D transposed convolution operator over an input signal composed of several input planes.

  See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d

  ```python exec="true" source="above" session="tensor" result="python"
  conv = nn.ConvTranspose1d(1, 1, 3)
  t = Tensor.rand(1, 1, 4)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = conv(t)
  print(t.numpy())
  ```
  """
  return ConvTranspose2d(in_channels, out_channels, (kernel_size,), stride, padding, output_padding, dilation, groups, bias)

class ConvTranspose2d(Conv2d):
  """
  Applies a 2D transposed convolution operator over an input image.

  See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d

  ```python exec="true" source="above" session="tensor" result="python"
  conv = nn.ConvTranspose2d(1, 1, 3)
  t = Tensor.rand(1, 1, 4, 4)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = conv(t)
  print(t.numpy())
  ```
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True):
    super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    scale = 1 / math.sqrt(in_channels * prod(self.kernel_size))
    self.weight = Tensor.uniform(in_channels, out_channels//groups, *self.kernel_size, low=-scale, high=scale)
    self.output_padding = output_padding

  def __call__(self, x:Tensor):
    return x.conv_transpose2d(self.weight, self.bias, padding=self.padding, output_padding=self.output_padding, stride=self.stride,
                              dilation=self.dilation, groups=self.groups)

class Linear:
  """
  Applies a linear transformation to the incoming data.

  See: https://pytorch.org/docs/stable/generated/torch.nn.Linear

  ```python exec="true" source="above" session="tensor" result="python"
  lin = nn.Linear(3, 4)
  t = Tensor.rand(2, 3)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = lin(t)
  print(t.numpy())
  ```
  """
  def __init__(self, in_features, out_features, bias=True):
    bound = 1 / math.sqrt(in_features)
    self.weight = Tensor.uniform(out_features, in_features, low=-bound, high=bound)
    self.bias = Tensor.uniform(out_features, low=-bound, high=bound) if bias else None

  def __call__(self, x:Tensor):
    return x.linear(self.weight.transpose(), self.bias)

class GroupNorm:
  """
  Applies Group Normalization over a mini-batch of inputs.

  - Described: https://paperswithcode.com/method/group-normalization
  - Paper: https://arxiv.org/abs/1803.08494v3

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.GroupNorm(2, 12)
  t = Tensor.rand(2, 12, 4, 4) * 2 + 1
  print(t.mean().item(), t.std().item())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = norm(t)
  print(t.mean().item(), t.std().item())
  ```
  """
  def __init__(self, num_groups:int, num_channels:int, eps:float=1e-5, affine:bool=True):
    self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
    self.weight: Optional[Tensor] = Tensor.ones(num_channels) if affine else None
    self.bias: Optional[Tensor] = Tensor.zeros(num_channels) if affine else None

  def __call__(self, x:Tensor):
    # reshape for layernorm to work as group norm
    # subtract mean and divide stddev
    x = x.reshape(x.shape[0], self.num_groups, -1).layernorm(eps=self.eps).reshape(x.shape)

    if self.weight is None or self.bias is None: return x
    # elementwise_affine on channels
    return x * self.weight.reshape(1, -1, *[1] * (len(x.shape)-2)) + self.bias.reshape(1, -1, *[1] * (len(x.shape)-2))

class InstanceNorm:
  """
  Applies Instance Normalization over a mini-batch of inputs.

  - Described: https://paperswithcode.com/method/instance-normalization
  - Paper: https://arxiv.org/abs/1607.08022v3

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.InstanceNorm(3)
  t = Tensor.rand(2, 3, 4, 4) * 2 + 1
  print(t.mean().item(), t.std().item())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = norm(t)
  print(t.mean().item(), t.std().item())
  ```
  """
  def __init__(self, num_features:int, eps:float=1e-5, affine:bool=True):
    self.num_features, self.eps = num_features, eps
    self.weight: Optional[Tensor] = Tensor.ones(num_features) if affine else None
    self.bias: Optional[Tensor] = Tensor.zeros(num_features) if affine else None

  def __call__(self, x:Tensor):
    x = x.reshape(x.shape[0], self.num_features, -1).layernorm(eps=self.eps).reshape(x.shape)
    if self.weight is None or self.bias is None: return x
    return x * self.weight.reshape(1, -1, *[1] * (len(x.shape)-2)) + self.bias.reshape(1, -1, *[1] * (len(x.shape)-2))

class LayerNorm:
  """
  Applies Layer Normalization over a mini-batch of inputs.

  - Described: https://paperswithcode.com/method/layer-normalization
  - Paper: https://arxiv.org/abs/1607.06450v1

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.LayerNorm(3)
  t = Tensor.rand(2, 5, 3) * 2 + 1
  print(t.mean().item(), t.std().item())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = norm(t)
  print(t.mean().item(), t.std().item())
  ```
  """
  def __init__(self, normalized_shape:Union[int, Tuple[int, ...]], eps:float=1e-5, elementwise_affine:bool=True):
    self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
    self.weight, self.bias = (Tensor.ones(*self.normalized_shape), Tensor.zeros(*self.normalized_shape)) if elementwise_affine else (None, None)

  def __call__(self, x:Tensor):
    assert self.normalized_shape == x.shape[-len(self.normalized_shape):], f"last dimensions of {x.shape} must match {self.normalized_shape}"
    x = x.layernorm(eps=self.eps, axis=self.axis)
    if not self.elementwise_affine: return x
    return x * self.weight + self.bias

class LayerNorm2d(LayerNorm):
  """
  Applies Layer Normalization over a mini-batch of 2D inputs.

  See: `LayerNorm`

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.LayerNorm2d(3)
  t = Tensor.rand(2, 3, 4, 4) * 2 + 1
  print(t.mean().item(), t.std().item())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = norm(t)
  print(t.mean().item(), t.std().item())
  ```
  """
  def __call__(self, x): return super().__call__(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class RMSNorm:
  """
  Applies Root Mean Square Normalization to input.

  - Described: https://paperswithcode.com/method/rmsnorm
  - Paper: https://arxiv.org/abs/1910.07467

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.RMSNorm(4)
  t = Tensor.arange(12, dtype=dtypes.float).reshape(3, 4)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  print(norm(t).numpy())
  ```
  """
  def __init__(self, dim, eps=1e-6): self.eps, self.weight = eps, Tensor.ones(dim)

  def _norm(self, x:Tensor): return x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()

  def __call__(self, x:Tensor) -> Tensor: return self._norm(x.float()).cast(x.dtype) * self.weight

class Embedding:
  """
  A simple lookup table that stores embeddings of a fixed dictionary and size.

  See: https://pytorch.org/docs/stable/generated/torch.nn.Embedding

  ```python exec="true" source="above" session="tensor" result="python"
  emb = nn.Embedding(10, 3)
  print(emb(Tensor([1, 2, 3, 1])).numpy())
  ```
  """
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_sz, self.embed_sz, self.weight = vocab_size, embed_size, Tensor.glorot_uniform(vocab_size, embed_size)

  def __call__(self, idx:Tensor) -> Tensor:
    if idx.numel() == 0: return Tensor.empty(idx.shape+(self.embed_sz,), device=self.weight.device)
    arange_shp, weight_shp, big_shp = (1, 1, self.vocab_sz, 1), (1, 1, self.vocab_sz, self.embed_sz), idx.shape+(self.vocab_sz, self.embed_sz,)
    if not hasattr(self, 'arange'): self.arange = Tensor.arange(self.vocab_sz, requires_grad=False, device=self.weight.device).reshape(arange_shp)
    arange, idx, vals = self.arange.expand(big_shp), idx.reshape(idx.shape+(1, 1,)).expand(big_shp), self.weight.reshape(weight_shp).expand(big_shp)
    return (arange == idx).mul(vals).sum(2)
