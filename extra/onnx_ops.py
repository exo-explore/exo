import functools, io, math
from typing import Union, Tuple, Optional, List, Any
from tinygrad.tensor import Tensor, _broadcast_shape
from tinygrad.dtype import ImageDType, dtypes
from tinygrad.helpers import prod, flatten
from extra.onnx import DTYPE_MAP, to_python_const
import numpy as np

tensor_methods = {"Neg", "Reciprocal", "Pow", "Sqrt", "Sign", "Abs", "Exp", "Log", "Mish", "Sin", "Cos", "Tan", "Relu", "Sigmoid", "MatMul",
                  "Floor", "Ceil", "Softplus", "HardSwish", "Where", "Mul", "Sinh", "Cosh", "Tanh", "Softsign", "Asinh", "Acosh", "Atanh",
                  "Elu", "Celu", "Xor", "Round"}

# **************** Free Ops ****************

def Identity(x: Tensor): return x
# TODO: fix buffer_parse
def Add(x: Tensor, other: Tensor, broadcast=None, axis=None): return x + other if x.dtype == dtypes.float or isinstance(x.dtype, ImageDType) else (x + other).cast(x.dtype)
def Sub(x: Union[Tensor, Any], other: Tensor): return x - other # some test has input as int
def Less(x:Tensor,y:Tensor): return x < y
def LessOrEqual(x:Tensor,y:Tensor): return x <= y
def Greater(x:Tensor,y:Tensor): return x > y
def GreaterOrEqual(x:Tensor,y:Tensor): return x >= y
def Equal(x:Tensor,y:Tensor): return x == y
def Max(*data_0): return functools.reduce(Tensor.maximum, data_0)
def Min(*data_0): return functools.reduce(Tensor.minimum, data_0)
def Sum(*data_0): return functools.reduce(Tensor.add, data_0)
def Mean(*data_0): return Sum(*data_0) / len(data_0)
# NOTE: does not support saturate
def Cast(x: Tensor, to: int, saturate=1): return x.cast(DTYPE_MAP[to])
def CastLike(x: Tensor, target_type: Tensor, saturate=1): return x.cast(target_type.dtype)

# **************** Simple Ops ****************

# https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_div.py
def Div(x: Tensor, other: Tensor): return (x/other).cast(x.dtype)

def Constant(value:Optional[Tensor]=None, value_float=None, value_floats=None, value_int=None, value_ints=None, value_string=None, value_strings=None):
  if value is not None: return value
  if value_float is not None: return Tensor(value_float, dtype=dtypes.float32, requires_grad=False)
  if value_floats is not None: return Tensor(list(value_floats), dtype=dtypes.float32, requires_grad=False)
  if value_int is not None: return Tensor(value_int, dtype=dtypes.int64, requires_grad=False)
  if value_ints is not None: return Tensor(list(value_ints), dtype=dtypes.int64, requires_grad=False)
  if value_string is not None or value_strings is not None: raise NotImplementedError('value_string or value_strings not implemented for Constant op')

def HardSigmoid(x: Tensor, alpha=0.2, beta=0.5): return (alpha*x + beta).clip(0, 1)
def Gelu(x:Tensor, approximate=None): return x.gelu() if approximate == "tanh" else 0.5 * x * (1 + Erf(x/math.sqrt(2)))
def Selu(X: Tensor, alpha=1.67326319217681884765625, gamma=1.05070102214813232421875): return gamma * (X.relu() - (-alpha*X.exp()+alpha).relu())
def PRelu(X:Tensor, slope:Tensor):
  slope = slope[0] if slope.shape[-1] != X.shape[-1] else slope # HACK OnnxBackendPyTorchConvertedModelTest HAS WEIRD SLOPE WHERE IT'S [0.25, 0.25, 0.25] FOR ANY X.SHAPE
  return (X > 0).where(X, X * slope)
def LeakyRelu(X: Tensor, alpha=0.01): return X.leakyrelu(alpha)
def ThresholdedRelu(X: Tensor, alpha=1.0): return (X > alpha).where(X, 0)
def Softmax_1(x: Tensor, axis=1): return x.softmax(axis)
def Softmax_13(x: Tensor, axis=-1): return x.softmax(axis)
Softmax = {1: Softmax_1, 13: Softmax_13}   # Softmax default axis changed
def LogSoftmax(x: Tensor, axis=-1): return x.log_softmax(axis)
def Clip(x: Tensor, min=None, max=None): return x.clip(float('-inf') if min is None else min, float('inf') if max is None else max).cast(x.dtype)

# NOTE ReduceProd would require a new llop
def _axes(axes, noop_with_empty_axes):
  if axes is not None and not (isinstance(axes, Tensor) and axes.shape == (0,)): return to_python_const(axes)
  return [] if noop_with_empty_axes else None
def ReduceMax(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return data.max(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceMin(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return data.min(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceSum(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return data.sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceMean(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return data.mean(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceSumSquare(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return ReduceSum(data.square(), axes, keepdims, noop_with_empty_axes)
def ReduceL1(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return ReduceSum(data.abs(), axes, keepdims, noop_with_empty_axes)
def ReduceL2(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return ReduceSumSquare(data, axes, keepdims, noop_with_empty_axes).sqrt()
def ReduceLogSum(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return ReduceSum(data, axes, keepdims, noop_with_empty_axes).log()
def ReduceLogSumExp(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return ReduceSum(data.exp(), axes, keepdims, noop_with_empty_axes).log()

def GlobalAveragePool(X: Tensor): return X.mean(axis=tuple(range(2, X.ndim)), keepdim=True)
def GlobalMaxPool(X: Tensor): return X.max(axis=tuple(range(2, X.ndim)), keepdim=True)
def OptionalHasElement(x: Optional[Tensor]=None): return Tensor(x is not None and x.numel() > 0)
def OptionalGetElement(x: Optional[Tensor]=None): return x if x is not None else Tensor([])

def Tile(x: Tensor, repeats): return x.repeat(to_python_const(repeats))
def Range(start: Tensor, limit, delta): return Tensor.arange(start=to_python_const(start), stop=to_python_const(limit), step=to_python_const(delta))
def Shape(data: Tensor, end=None, start=0): return Tensor(data.shape[start:end], dtype=dtypes.int64)
def Size(data: Tensor): return prod(data if isinstance(data, list) else data.shape)
def Flatten(x: Tensor, axis=1): return x.reshape(prod(x.shape[0:axis]), -1)
def Reshape(data: Tensor, shape: Tensor, allowzero=0):
  return data.reshape([int(x) if x != 0 else (0 if allowzero else data.shape[i]) for i,x in enumerate(to_python_const(shape))])
def Expand(x: Tensor, shape:Tensor): return x.expand(_broadcast_shape(x.shape, tuple(to_python_const(shape))))
def Shrink(x: Tensor, bias=0.0, lambd=0.5): return (x < -lambd)*(x+bias) + (x > lambd)*(x-bias)
def And(x:Tensor, y:Tensor): return (x==y).where(x, False)
def Or(x:Tensor, y:Tensor): return (x==y).where(x, True)
def Not(x:Tensor): return x.logical_not()

def Asin(x): return Atan(x / (1 - x * x).sqrt())
def Acos(x: Tensor):
  negate = (x < 0)
  x = x.abs()
  ret = ((((-0.0187293 * x) + 0.0742610)*x - 0.2121144) * x + 1.5707288) * (1.0 - x).sqrt()
  ret = ret - 2 * negate * ret
  return negate * math.pi + ret
def Atan(y: Tensor):
  t1 = y.abs()
  t3 = (1 > t1).where(t1, t1.reciprocal())
  t4 = t3 * t3
  t0 = ((((-0.013480470 * t4 + 0.057477314) * t4 - 0.121239071) * t4 + 0.195635925) * t4 - 0.332994597) * t4 + 0.999995630
  t3 = t0 * t3
  t3 = (t1 > 1).where(1.570796327 - t3, t3)
  return y.sign() * t3

def Trilu(x: Tensor, k: Union[Tensor, int]=0, upper=1):
  k = to_python_const(k) if isinstance(k, Tensor) else 0 # onnx passes k as a tensor int64 with one element, default is 0
  return x.triu(k) if upper else x.tril(k)

def Squeeze(data: Tensor, axes):
  if isinstance(axes, Tensor): axes = to_python_const(axes)
  axes = [data._resolve_dim(x) for x in axes]
  return data.reshape([s for i,s in enumerate(data.shape) if i not in axes])
def Unsqueeze(data: Tensor, axes):
  axes = sorted([x + data.ndim if x < 0 else x for x in to_python_const(axes)])
  new_shape = list(data.shape)
  for axis in axes: new_shape.insert(axis, 1)
  return data.reshape(new_shape)

def Binarizer(x, threshold=0.0): return (x > threshold).float()

def ArgMax(x: Tensor, axis=0, keepdims=1, select_last_index=0):
  if select_last_index: return ((x.shape[axis]-1) - x.flip(axis).argmax(axis, keepdim=keepdims)).cast(dtypes.int64)
  return x.argmax(axis, keepdim=keepdims).cast(dtypes.int64)
def ArgMin(x, axis=0, keepdims=1, select_last_index=0): return ArgMax(-x, axis=axis, keepdims=keepdims, select_last_index=select_last_index)

def Concat(*xs: List[Tensor], axis): return Tensor.cat(*xs, dim=axis)
def Transpose(x: Tensor, perm=None): return x.permute(order=list(range(x.ndim)[::-1]) if perm is None else perm)

def ConstantOfShape(x, value:Tensor=None):
  if value is None: value = 0.0
  shape = to_python_const(x)
  return Tensor.ones(*shape, dtype=value.dtype) * (value if shape[0]!=0 else 1)

# **************** Complex Ops ****************

def Gemm(A: Tensor, B: Tensor, C: Tensor=None, alpha=1.0, beta=1.0, transA=0, transB=0, broadcast=0):
  ret = alpha * (A.transpose(transA) @ B.transpose(transB))
  if C is not None: ret = ret + beta * (C if broadcast == 0 else C.reshape([-1 if i <  len(C.shape) else 1 for i in range(ret.ndim)][::-1]))
  return ret

def Einsum(*Inputs: List[Tensor], equation): return Tensor.einsum(equation, Inputs)

def CumSum(X:Tensor, axis:Tensor, exclusive=0, reverse=0):
  axis = to_python_const(axis)
  if axis < 0: axis += X.ndim
  if reverse: X = X.flip(axis)
  if exclusive:
    pad_arg, shrink_arg = [None] * X.ndim, [None] * X.ndim
    pad_arg[axis] = (1, 0)
    shrink_arg[axis] = (0, X.shape[axis])
    X = X.pad(tuple(pad_arg)).shrink(tuple(shrink_arg))
  if reverse: return X.cumsum(axis).flip(axis)
  return X.cumsum(axis)

# TODO: this is copied from tinygrad/nn/__init__.py
# spatial is from opset 7 and has since been removed
def BatchNormalization(X: Tensor, scale, B, input_mean, input_var, epsilon=1e-05, momentum=0.9, training_mode=0, spatial=1, is_test=0):
  if training_mode:
    x_detached = X.detach()
    current_mean = x_detached.mean(axis=(0,2,3))
    y = (x_detached - current_mean.reshape(shape=[1, -1, 1, 1]))
    current_var = (y*y).mean(axis=(0,2,3))
    current_invstd = current_var.add(epsilon).rsqrt()

    running_mean = input_mean * momentum + current_mean * (1 - momentum)
    running_var = input_var * momentum + current_var * (1 - momentum)

    return X.batchnorm(scale, B, current_mean, current_invstd), running_mean, running_var
  invstd = (input_var + epsilon).rsqrt()
  return X.batchnorm(scale, B, input_mean, invstd)

def InstanceNormalization(x: Tensor, scale: Tensor, bias: Tensor, epsilon=1e-05):
  axis = tuple(range(2, x.ndim))
  mean = x.mean(axis=axis, keepdim=True)
  invstd = x.sub(mean).pow(2).mean(axis=axis, keepdim=True).add(epsilon).rsqrt()
  return x.sub(mean).mul(scale.reshape(shape=[-1, 1, 1])).mul(invstd).add(bias.reshape(shape=[-1, 1, 1]))

def LayerNormalization(x: Tensor, scale, bias, axis=-1, epsilon=1e-05, stash_type=1):
  assert stash_type == 1, "only float32 is supported"
  axis = tuple(i for i in range(axis if axis >= 0 else x.ndim + axis, x.ndim))
  mean = x.mean(axis=axis, keepdim=True)
  return x.layernorm(axis, epsilon).mul(scale).add(bias), mean, (x.sub(mean)).pow(2).mean(axis=axis, keepdim=True).add(epsilon).rsqrt()

def GroupNormalization(x: Tensor, scale: Tensor, bias: Tensor, num_groups, epsilon=1e-05):
  return x.reshape(x.shape[0], num_groups, -1).layernorm(axis=-1, eps=epsilon).mul(scale.unsqueeze(-1)).add(bias.unsqueeze(-1)).reshape(x.shape)

# onnx: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
# numpy.pad: ((x1_begin, x1_end), (x2_begin, x2_end), ...)
def _format_padding(onnx_pads, ndims=None, axes=None):
  if ndims and len(onnx_pads)//2 != ndims:  onnx_pads = onnx_pads * ndims # for OnnxBackendPyTorchConvertedModelTest the len(onnx_pads) == 2
  if ndims is None: ndims = len(onnx_pads) // 2
  if axes is None: axes = list(range(ndims))
  num_axes = len(axes)
  np_pads = [(0,0)] * ndims
  for i in range(num_axes):
    np_pads[axes[i]] = (onnx_pads[i], onnx_pads[i + num_axes])
  return np_pads

def _padded(X: Tensor, pads=None, auto_pad="NOTSET", axes=None, constant_value=0., strides=None, kernel_shape=None, dilations=None, ceil_mode=0):
  if auto_pad != "NOTSET": pads = _auto_pad(X, auto_pad, strides, kernel_shape, dilations)
  elif ceil_mode:
    if strides is not None: strides = [strides]*len(kernel_shape) if isinstance(strides, int) else strides if strides else [1]*len(kernel_shape)
    if dilations is not None: dilations = [1]*len(kernel_shape) if dilations == 1 else dilations
    out_spatial_shape = [math.ceil((sh - dil * (ker-1)-1)/st + 1) if ceil_mode else math.floor((sh - dil * (ker-1)-1)/st + 1) for sh, st, ker, dil in zip(X.shape[-len(kernel_shape):], strides, kernel_shape, dilations)]
    pad_shape = [(osh-1)*st+((ks-1)*dil+1)-ish for osh, st, ks, dil, ish in zip(out_spatial_shape, strides, kernel_shape, dilations, X.shape[-len(kernel_shape):])]
    pad_shape = [[sh//2, sh-sh//2] for sh in pad_shape]
    # ceil_mode case follows NOTE in https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
    # so if any kernels start in right padded region, we decrease right pads to omit that kernel. Only omitting 1 kernel now.
    pad_shape = [[start,end-rpad] if (rpad := ks + st%(st-(((start+xs)%st)))) <= end else [start,end]
                 for (start,end), ks, st, xs in zip(pad_shape, kernel_shape, strides, X.shape[-len(kernel_shape):])]
    pad_shape = flatten(pad_shape)
    pads = pad_shape[::2] + pad_shape[1::2]
  if pads is None: return X
  pads = _format_padding(pads, ndims=len(X.shape), axes=axes)
  return X.pad(tuple(pads), value=constant_value)

def _auto_pad(X: Tensor, auto_pad, strides, kernel_shape, dilations):
  strides = [strides]*len(kernel_shape) if isinstance(strides, int) else strides if strides else [1]*len(kernel_shape)
  dilations = [1]*len(kernel_shape) if dilations == 1 else dilations
  if auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
    pad_shape = [(math.ceil(sh/st)-1)*st+((ks-1)*di+1)-sh for sh, st, ks, di in zip(X.shape[-len(kernel_shape):], strides, kernel_shape, dilations)]
    pad_shape = flatten([[sh//2, sh-sh//2] for sh in pad_shape])
    return pad_shape[::2] + pad_shape[1::2] if auto_pad == "SAME_UPPER" else pad_shape[1::2] + pad_shape[::2]
  raise NotImplementedError(f"auto_pad={auto_pad} not implemented")

def Pad(x: Tensor, pads: Union[Tensor, Tuple[int, ...]], constant_value: Tensor=None, axes: Tensor=None, mode="constant", value: float=0.):
  constant_value = value if constant_value is None else float(to_python_const(constant_value))
  seq_pads = list(pads) if isinstance(pads, tuple) else to_python_const(pads)
  seq_pads = [math.ceil(i) for i in seq_pads]
  seq_axes = to_python_const(axes) if axes is not None else None
  base_shape = x.shape
  pads = _format_padding(seq_pads, ndims=len(x.shape), axes=seq_axes)
  if mode == "wrap":
    repeat_args = [math.ceil(dim[0]/sh) + math.ceil(dim[1]/sh) + 1 for dim, sh in zip(pads, base_shape)]
    new_shape = [s*r for s,r in zip(base_shape, repeat_args)]
    shrink_args = [(sh-dim[0]%sh if dim[0]%sh != 0 else 0, nsh-(sh-dim[1]%sh if dim[1]%sh != 0 else 0)) for dim, sh, nsh in zip(pads, base_shape, new_shape)]
    return x.repeat(tuple(repeat_args)).shrink(tuple(shrink_args))
  if mode == "reflect":
    for i,s in enumerate(x.shape):
      if pads[i] != (0,0):
        xL = x.flip(i).shrink(tuple((s-pads[i][0]-1, s_-1) if i_ == i else None for i_,s_ in enumerate(x.shape)))
        xR = x.flip(i).shrink(tuple((1, pads[i][1]+1) if i_ == i else None for i_ in range(x.ndim)))
        x = xL.cat(x, xR, dim=i)
    return x
  if mode == "edge":
    for i,s in enumerate(x.shape):
      if pads[i] != (0,0):
        xL = x.shrink(tuple((0,1) if i_ == i else None for i_ in range(x.ndim))).expand([pads[i][0] if i_ == i else None for i_ in range(x.ndim)])
        xR = x.shrink(tuple((s_-1, s_) if i_ == i else None for i_,s_ in enumerate(x.shape))).expand([pads[i][1] if i_ == i else None for i_ in range(x.ndim)])
        x = xL.cat(x, xR, dim=i)
    return x
  if mode == "constant":
    return _padded(x, seq_pads, axes=seq_axes, constant_value=constant_value)

def AveragePool(X: Tensor, kernel_shape, auto_pad="NOTSET", ceil_mode=0, count_include_pad=0, dilations=1, pads=None, strides=1):
  pixel_axes = tuple(range(2, X.ndim))
  ret = _padded(X, pads, auto_pad, axes=pixel_axes, strides=strides, kernel_shape=kernel_shape, dilations=dilations, ceil_mode=ceil_mode)
  ret = ret.avg_pool2d(kernel_shape, stride=strides, dilation=dilations)
  if count_include_pad: return ret
  div = _padded(Tensor.ones(X.shape), pads, auto_pad, axes=pixel_axes, strides=strides, kernel_shape=kernel_shape, dilations=dilations, ceil_mode=ceil_mode).avg_pool2d(kernel_shape, stride=strides, dilation=dilations)
  return ret / div

def MaxPool(X: Tensor, kernel_shape, auto_pad="NOTSET", ceil_mode=0, dilations=1, pads=None, storage_order=0, strides=1):
  pixel_axes = tuple(range(2, X.ndim))
  ret = _padded(X, pads, auto_pad, constant_value=-math.inf, axes=pixel_axes, strides=strides, kernel_shape=kernel_shape, dilations=dilations, ceil_mode=ceil_mode)
  ret = ret.max_pool2d(kernel_shape, stride=strides, dilation=dilations).cast(X.dtype)
  ret_len, X_len = ret.numel(), X.numel()
  indices = ((ret.flatten().unsqueeze(1).expand(ret_len, X_len) == X.flatten().unsqueeze(0).expand(ret_len, X_len)) * \
             Tensor.arange(X_len, dtype=dtypes.int64).unsqueeze(0).expand(ret_len, X_len)).sum(1).reshape(ret.shape)
  if storage_order: indices = indices.transpose(-2, -1)
  return ret, indices

def MaxUnpool(xT: Tensor, xI: Tensor, outshape: Optional[Tensor]=None, kernel_shape=None, pads=None, strides=None):
  out_sh = [(ks//2)*2 + st * inps for inps, st, ks in zip(xI.shape, strides, kernel_shape)]
  outlength = prod(out_sh)
  xI = xI.flatten().unsqueeze(1).expand(None, outlength)
  arange = Tensor.arange(outlength, requires_grad=False).reshape(1, outlength).expand(xI.shape)
  xT = xT.flatten().unsqueeze(1).expand(None, outlength)
  ret = ((xI == arange) * xT).sum(0).reshape([1, 1] + out_sh)
  if outshape is not None and (outshape := to_python_const(outshape)) != ret.shape:
    diff = [outshape[2] - ret.shape[2], outshape[3] - ret.shape[3]]
    pad_args = [diff[0]//2, diff[1]//2, diff[0]-diff[0]//2, diff[1]-diff[1]//2]
    ret = ret.pad2d((pad_args[1], pad_args[3], pad_args[0], pad_args[2]))
  return ret

def Conv(X: Tensor, W: Tensor, B:Optional[Tensor]=None, auto_pad="NOTSET", dilations=1, group=1, kernel_shape=None, pads=None, strides=1):
  if auto_pad != "NOTSET":
    padding = _auto_pad(X, auto_pad, strides, kernel_shape, dilations)
  else:
    # reorder padding
    padding = [p for ps in zip(pads[:len(pads)//2][::-1], pads[len(pads)//2:][::-1]) for p in ps] if pads is not None else 0
  return X.conv2d(W, B, stride=strides, groups=group, dilation=dilations, padding=padding)

def ConvTranspose(X: Tensor, W: Tensor, B:Optional[Tensor]=None, auto_pad="NOTSET", dilations=1, group=1, kernel_shape=None, pads=None, output_shape=None, output_padding=0, strides=1):
  if kernel_shape is None: kernel_shape = W.shape[2:]
  if isinstance(strides, int): strides = [strides]*(W.ndim-2)
  if isinstance(dilations, int): dilations = [dilations]*(W.ndim-2)
  if isinstance(output_padding, int): output_padding = [output_padding]*(W.ndim-2)
  out_sh = [st*(xs-1) + (ks-1)*di+1 if n < 2 else st*(xs-1) + (ks-1)*di+1 - pads[n-2] - pads[n-1] for n, (st, xs, ks, di) in enumerate(zip(strides, X.shape[2:], kernel_shape, dilations))] if output_shape is not None or auto_pad != "NOTSET" else []
  if pads is None:
    if output_shape is None: output_shape = [xs*st for xs, st in zip(X.shape[2:], strides)]
    if auto_pad == "NOTSET": pads = [0,0] * (X.ndim - 2)
    else:
      total_padding = [st*(ish-1) + pad + ((ks-1)*dil+1)-osh for st, ish, pad, ks, dil, osh in zip(strides, X.shape[2:], output_padding, kernel_shape, dilations, output_shape)]
      pad_shape = flatten([[sh//2, sh-sh//2] for sh in total_padding])
      pads = pad_shape[::2] + pad_shape[1::2] if auto_pad == "SAME_UPPER" else pad_shape[1::2] + pad_shape[::2]
  else:
    if output_shape is None: output_shape = [st*(xs-1) + (ks-1)*di+1 if n < 2 else st*(xs-1) + (ks-1)*di+1 - pads[n-2] - pads[n-1] for n, (st, xs, ks, di) in enumerate(zip(strides, X.shape[2:], kernel_shape, dilations))]
  if out_sh: output_padding = [os - rs for os, rs in zip(output_shape, out_sh)]
  return X.conv_transpose2d(W, B, stride=strides, groups=group, dilation=dilations, padding=pads if pads is not None else 0, output_padding=output_padding)

def DepthToSpace(X:Tensor, blocksize:int, mode:str="DCR"):
  b, c, h, w = X.shape
  if mode == "DCR":
    return X.reshape(b, blocksize, blocksize, c // (blocksize**2), h, w).permute(0, 3, 4, 1, 5, 2).reshape(b, c // (blocksize**2), h * blocksize, w * blocksize)
  elif mode == "CRD":
    return X.reshape(b, c // (blocksize ** 2), blocksize, blocksize, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // (blocksize ** 2), h * blocksize, w * blocksize)

def SpaceToDepth(X:Tensor, blocksize:int):
  b, c, h, w = X.shape
  return X.reshape(b, c, h // blocksize, blocksize, w // blocksize, blocksize).permute(0, 3, 5, 1, 2, 4).reshape(b, c * (blocksize**2), h // blocksize, w // blocksize)

# Reimplemented here because you need legacy RNG for passing ONNX tests.
def Dropout(data: Tensor, ratio=0.5, training_mode=False, seed=None):
  if isinstance(ratio, Tensor) and not ratio.shape: ratio = to_python_const(ratio) # ratio and tensor is passed in as Tensor with shape: ()
  if isinstance(training_mode, Tensor) and not training_mode.shape: training_mode = to_python_const(training_mode)
  if not training_mode: return data, Tensor.ones(data.shape, dtype=dtypes.bool)  # if mask is requested as output it will contain all True's.
  rng = np.random.RandomState(seed)
  if isinstance(ratio, Tensor): ratio = ratio.item()
  mask = Tensor(rng.random(data.shape) >= ratio, requires_grad=False, device=data.device)
  return data * mask * (1/(1.0 - ratio)), mask

def LRN(x: Tensor, size, alpha=1e-4, beta=0.75, bias=1.0):
  bs, c, iy, ix = x.shape
  return x / x.mul(x).reshape(bs,1,c,iy*ix).pad2d((0,0,(size-1)//2, size//2)).avg_pool2d((size, 1), 1).reshape(bs,c,iy,ix).mul(alpha).add(bias).pow(beta)

def MeanVarianceNormalization(x: Tensor, axis=(0, 2, 3)):
  mean = x.mean(axis, keepdim=True)
  std = x.std(axis, keepdim=True, correction=0)
  return (x - mean) / (std + 1e-9)

def NegativeLogLikelihoodLoss(x: Tensor, target: Tensor, weight=None, ignore_index=None, reduction="mean"):
  N, C, i_shape = x.shape[0], x.shape[1], x.shape
  t_shape = target.shape
  if len(x.shape) != 3:
    x = x.reshape((N, C, -1))
    target = target.reshape((N, -1))
  if weight is not None:
    mask = target.unsqueeze(-1) == Tensor.arange(C).repeat((N, 1, 1))
    weight = (mask * weight).sum(axis=-1)
  if ignore_index is not None:
    cond = target == ignore_index
    weight = cond.where(0, weight) if weight is not None else cond.where(0, 1)
  mask = target[:, None, :] == Tensor.arange(C).reshape([1, C] + [1]*(x.ndim -2))
  loss = -(mask * x).sum(axis=1) * (1 if weight is None else weight)
  if reduction == "mean": return loss.mean() if weight is None else loss.sum() / weight.sum()
  if reduction == "sum": return loss.sum()
  return loss.reshape(t_shape) if len(i_shape) != 3 else loss

def SoftmaxCrossEntropyLoss(scores: Tensor, labels: Tensor, weights=None, ignore_index=None, reduction="mean"):
  _N, C, *s_dimensions = scores.shape
  if ignore_index is not None: labels = (labels == ignore_index).where(C+1, labels)
  mask = labels.unsqueeze(1) == Tensor.arange(C).reshape(1, C, *[1]*len(s_dimensions))
  y = scores.log_softmax(axis=1)
  loss = (mask * -y).sum(1)
  if weights is not None:
    weights = weights[labels, ...]
    loss = loss * weights
  if reduction == "mean": loss = loss.sum() / ((loss != 0).sum() if weights is None else weights.sum())
  elif reduction == "sum": loss = loss.sum()
  return loss, y

def ArrayFeatureExtractor(x: Tensor, indices: Tensor): return x[..., indices]

def Gather(x: Tensor, indices: Tensor, axis=0):
  if indices.numel() < 9: # NOTE lessor kernels for smaller indices but kernel number increases depending on size of indices
    x_sh = list(x.shape)
    ret_shape = x_sh[:axis] + list(indices.shape) + x_sh[axis+1:]
    if indices.ndim > 1: indices = indices.flatten()
    indices = [to_python_const(indices)] if indices.shape == () else [x_sh[axis]+x if x<0 else x for x in to_python_const(indices)]
    args = [[(0,x) if j != axis else (i,i+1) for j, x in enumerate(x_sh)] for i in indices]
    return x.shrink(arg=tuple(args[0])).cat(*[x.shrink(arg=tuple(arg)) for arg in args[1:]], dim=axis).reshape(ret_shape)
  # NOTE faster gather, fixed number of kernels, but exceeds limited kernels for openpilot
  return x[tuple([slice(None) if i != axis else indices for i in range(x.ndim)])]

def GatherElements(x: Tensor, indices: Tensor, axis):
  indices = (indices < 0).where(x.shape[axis], 0) + indices
  return x.gather(axis, indices)

# TODO clean this up, it's taking the longest in CI
def Resize(X:Tensor, roi=None, scales=None, sizes=None, antialias=0, axes=None, coordinate_transformation_mode='half_pixel',
           cubic_coeff_a=-0.75, exclude_outside=0, extrapolation_value=0.0, keep_aspect_ratio_policy='stretch',
           mode='nearest', nearest_mode='round_prefer_floor'):
  def _nearest_gather(X: Tensor, x_out, y_out): return X[:,:,y_out,:][:,:,:,x_out]
  def _nearest_mode(x_resized: Tensor, nearest_mode: str, x_len):
    if nearest_mode == "round_prefer_floor": ret = (x_resized - 0.5).ceil()
    elif nearest_mode == "round_prefer_ceil": ret = (x_resized + 0.5).floor()
    elif nearest_mode == "floor": ret = x_resized.floor()
    elif nearest_mode == "ceil": ret = x_resized.ceil()
    return ret.cast(dtypes.int32).clip(0, x_len-1)
  def _coordinate_transformation(x_out, y_out, output_shape, scales_, roi=None):
    if coordinate_transformation_mode == "half_pixel":
      x_out = (x_out + 0.5) / scales_[-1] - 0.5
      y_out = (y_out + 0.5) / scales_[-2] - 0.5
    elif coordinate_transformation_mode == "align_corners":
      x_out = x_out * (X.shape[-1] - 1) / (output_shape[-1] - 1)
      y_out = y_out * (X.shape[-2] - 1) / (output_shape[-2] - 1)
    elif coordinate_transformation_mode == "asymmetric":
      x_out = x_out / scales_[-1]
      y_out = y_out / scales_[-2]
    elif coordinate_transformation_mode == "half_pixel_symmetric":
      x_out = X.shape[-1] / 2 * (1 - int(output_shape[-1]) / output_shape[-1]) + (x_out + 0.5) / scales_[-1] - 0.5
      y_out = X.shape[-2] / 2 * (1 - int(output_shape[-2]) / output_shape[-2]) + (y_out + 0.5) / scales_[-2] - 0.5
    elif coordinate_transformation_mode == "pytorch_half_pixel":
      x_out = (x_out + 0.5) / scales_[-1] - 0.5 if output_shape[-1] > 1 else Tensor([0])
      y_out = (y_out + 0.5) / scales_[-2] - 0.5 if output_shape[-2] > 1 else Tensor([0])
    elif coordinate_transformation_mode == "tf_crop_and_resize":
      x_out = roi[-1][0] * (X.shape[-1] - 1) + x_out * ((roi[-1][1] - roi[-1][0]) * (X.shape[-1] - 1) / (output_shape[-1] - 1)) if output_shape[-1] > 1 else Tensor([0.5 * (roi[-1][0] + roi[-1][1]) * (X.shape[-1] - 1)])
      y_out = roi[-2][0] * (X.shape[-2] - 1) + y_out * ((roi[-2][1] - roi[-2][0]) * (X.shape[-2] - 1) / (output_shape[-2] - 1)) if output_shape[-2] > 1 else Tensor([0.5 * (roi[-2][0] + roi[-2][1]) * (X.shape[-2] - 1)])
    return x_out.clip(0, X.shape[-1]-1), y_out.clip(0, X.shape[-2]-1)
  if roi is not None:
    roi = to_python_const(roi)
    roi = [(st,ed) for st, ed in zip(roi[:len(roi)//2], roi[len(roi)//2:])]
    roi_ = [(1,1)] * 4
    if axes is not None:
      for a,r in zip(axes, roi):
        roi_[a] = r
      roi = roi_
  if scales is not None:
    scales = to_python_const(scales)
    if axes is not None:
      scales_ = [1]*X.ndim
      for a,s in zip(axes, scales):
        scales_[a] = s
      scales = scales_
  elif sizes is not None:
    sizes = to_python_const(sizes)
    scales = []
    if axes is not None:
      sizes_ = [1]*X.ndim
      for a,s in zip(axes, sizes):
        sizes_[a] = s
        scales.append(s/X.shape[a])
      sizes = sizes_
    else: scales = [si/xs for xs, si in zip(X.shape, sizes)]
    if keep_aspect_ratio_policy == "not_larger":
      scale = min(scales)
      sizes = list(X.shape[:-2]) + [math.ceil(sh*scale) for sh in X.shape[-2:]]
    elif keep_aspect_ratio_policy == "not_smaller":
      scale = max(scales)
      sizes = list(X.shape[:-2]) + [math.ceil(sh*scale) for sh in X.shape[-2:]]
  output_shape = sizes if sizes else [math.floor(x*s) for x,s in zip(X.shape, scales)]
  output_shape_ = sizes if sizes else [x*s for x,s in zip(X.shape, scales)]
  scales_ = [os/xs for xs, os in zip(X.shape, output_shape)]
  x_out = Tensor.arange(output_shape[-1], dtype=dtypes.default_float)
  y_out = Tensor.arange(output_shape[-2], dtype=dtypes.default_float)
  if mode == "nearest":
    x_out, y_out = _coordinate_transformation(x_out, y_out, output_shape, scales_, roi)
    x_out = _nearest_mode(x_out, nearest_mode, X.shape[-1])
    y_out = _nearest_mode(y_out, nearest_mode, X.shape[-1])
    return _nearest_gather(X, x_out, y_out)
  if mode == "linear":
    x_out, y_out = _coordinate_transformation(x_out, y_out, output_shape_, scales, roi)
    ret = []
    for y in to_python_const(y_out):
      for x in to_python_const(x_out):
        x_floor, y_floor = int(x), int(y)
        y_shrink = (y_floor, math.ceil(y)+1)
        x_shrink = (x_floor, math.ceil(x)+1)
        corners = to_python_const(X.shrink((None, None, y_shrink, x_shrink)))[0][0]

        wx, wy = math.ceil(x) - x, math.ceil(y) - y
        if x == x_floor and y == y_floor:
          weighted = corners[0][0]
        elif x == x_floor:
          weighted = corners[0][0] * wy + corners[1][0] * (1-wy)
        elif y == y_floor:
          weighted = corners[0][0] * wx + corners[0][1] * (1-wx)
        else:
          weighted = (corners[0][0] * wx + corners[0][1] * (1-wx)) * wy + \
                     (corners[1][0] * (wx) + corners[1][1] * (1-wx)) * (1-wy)
        ret.append(weighted)
    return Tensor(ret).reshape(output_shape)
  if mode == "cubic":
    raise NotImplementedError("cubic interpolation is not implemented")

def CenterCropPad(t: Tensor, shape: Tensor, axes=None):
  if not axes: axes = list(range(t.ndim))
  shrink_arg = [None] * t.ndim
  pad_arg = [None] * t.ndim
  shape = to_python_const(shape)
  for s, x in zip(shape, axes):
    tx = t.shape[x]
    if s < tx: shrink_arg[x] = (tx//2 - (s+1)//2, tx//2 + s//2)
    elif s > tx: pad_arg[x] = ((s-tx)//2, (s-tx+1)//2)
  return t.shrink(tuple(shrink_arg)).pad(tuple(pad_arg))

def OneHot(indices: Tensor, depth: Tensor, values: Tensor, axis=-1):
  depth = int(to_python_const(depth))
  indices, rank = (indices < 0).where(indices+depth, indices), indices.ndim
  if axis < 0: axis += rank + 1
  ls, rs = indices.shape[0:axis], indices.shape[axis: rank]
  cond = indices[:,None] == Tensor.arange(depth).reshape((1,) * len(ls) + (depth,) + (1,) * len(rs))
  return cond.where(values[1], values[0])

def Erf(x: Tensor):
  t = 1.0 / (1.0 + 0.3275911 * x.abs())
  term1 = 0.254829592 * t
  term2 = -0.284496736 * t ** 2
  term3 = 1.421413741 * t ** 3
  term4 = -1.453152027 * t ** 4
  term5 = 1.061405429 * t ** 5
  y = (term1 + term2 + term3 + term4 + term5)
  z = 1.0 - y * (-x * x).exp()
  return (x > 0).where(z, -z)

def Compress(inp: Tensor, condition: Tensor, axis=None):
  if axis is None:
    inp = inp.flatten()
    axis = 0

  if axis < 0: axis += inp.ndim

  con_np = to_python_const(condition)
  con = Tensor(np.arange(condition.shape[0])[con_np]) # no boolean indexing in Tensor
  return inp[tuple(con if i == axis else slice(None) for i in range(inp.ndim))]

def EyeLike(x: Tensor, dtype=None, k=0):
  if dtype is None: dtype = x.dtype
  else: dtype = DTYPE_MAP[int(dtype)]
  dim = min(x.shape)
  if x.shape[0] == x.shape[1]:
    return Tensor.eye(dim, dtype=dtype)
  padarg = tuple(None if d == dim else (k, d-dim-k) for d in x.shape)
  return Tensor.eye(dim, dtype=dtype).pad(padarg)

def Upsample(X, scales, mode): return Resize(X=X, scales=scales, mode=mode)

def IsInf(x: Tensor, detect_negative=1, detect_positive=1):
  return (x == float("inf")) * bool(detect_positive) + (x == float("-inf")) * bool(detect_negative)

def DequantizeLinear(x: Tensor, x_scale: Tensor, x_zero_point: Union[Tensor, int] = 0, axis=1, block_size=0):
  def numpy_repeat(t: Tensor, axis, repeats, out_shape):
    t = t.reshape(tuple(-1 if i == axis-1 else 1 if i == axis else sh for i,sh in enumerate(t.shape)))
    return t.repeat([repeats if i == axis else 1 for i in range(t.ndim)]).reshape(out_shape)
  if axis < 0: axis += x.ndim
  if block_size:
    x_zer, x_sc = numpy_repeat(x_zero_point, axis, block_size, x.shape), numpy_repeat(x_scale, axis, block_size, x.shape)
  else:
    x_sc = x_scale.reshape(*[1]*axis, *x_scale.shape, *[1]*(x.ndim - axis - x_scale.ndim))
    x_zer = x_zero_point.reshape(*[1]*axis, *x_scale.shape, *[1]*(x.ndim - axis - x_scale.ndim)) if isinstance(x_zero_point, Tensor) else x_zero_point
  return ((x.float() - x_zer) * x_sc).cast(x_scale.dtype)

def IsNaN(x: Tensor): return x != x

# copied from https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_image_decoder.py
# without importing PIL we'll have to manually decode a bunch of image formats like PNG, JPEG, WebP, etc
def ImageDecoder(encoded_stream: Tensor, pixel_format="RGB"):
  try:
    import PIL.Image
  except ImportError as e:
    raise ImportError("Pillow must be installed to use the reference implementation of the ImageDecoder operator") from e
  img = PIL.Image.open(io.BytesIO(to_python_const(encoded_stream, tobytes=True)))
  if pixel_format == "BGR":
    return Tensor(np.array(img))[:, :, ::-1]
  if pixel_format == "RGB":
    return Tensor(np.array(img))
  if pixel_format == "Grayscale":
    img = img.convert("L")
    decoded = Tensor(np.array(img))
    return decoded.unsqueeze(-1) # (H, W) to (H, W, 1)
  raise ValueError(f"pixel_format={pixel_format!r} is not supported.")

def AffineGrid(theta: Tensor, size: Tensor, align_corners=0):
  _, _, *data_sz = to_python_const(size)
  size_zeros, original_grid = Tensor.zeros(data_sz), Tensor.ones(data_sz)
  stackable = [original_grid]
  for dim, dim_sz in enumerate(data_sz):
    a = Tensor.arange(-1, 1.0001, 2/(dim_sz-1)) if align_corners == 1 else Tensor.arange(-1+1/dim_sz, 1, 2/dim_sz)
    if dim == 0: stackable = [a.reshape(dim_sz, *[1]*(len(data_sz)-1)) + size_zeros, *stackable]
    elif dim == 1: stackable = [a.reshape(1, dim_sz, *[1]*(len(data_sz)-2)) + size_zeros, *stackable]
    else: stackable = [a.reshape(1, dim_sz) + size_zeros, *stackable]
  original_grid = Tensor.stack(*stackable, dim=len(data_sz))
  if original_grid.ndim == 3:
    N, dim_2d, dim_homo = theta.shape
    assert dim_2d == 2 and dim_homo == 3
    H, W, dim_homo = original_grid.shape
    assert dim_homo == 3
    original_grid = original_grid.reshape(H*W, dim_homo).transpose()
    return theta.matmul(original_grid).permute(0,2,1).reshape(N, H, W, dim_2d)
  assert original_grid.ndim == 4
  N, dim_3d, dim_homo = theta.shape
  assert dim_3d == 3 and dim_homo == 4
  D, H, W, dim_homo = original_grid.shape
  assert dim_homo == 4
  original_grid = original_grid.reshape(D*H*W, dim_homo).transpose()
  return theta.matmul(original_grid).permute(0,2,1).reshape(N, D, H, W, dim_3d)

# **************** com.microsoft Ops ****************

def SkipLayerNormalization(x:Tensor, skip:Tensor, gamma, beta:Optional[Tensor]=None, bias:Optional[Tensor]=None, epsilon=None):
  if epsilon is None: epsilon=1e-12
  x = x + skip + bias
  return x.layernorm(eps=epsilon) * gamma + beta, None, None, x

def FastGelu(x:Tensor, bias:Optional[Tensor]=None):
  # this is tanh approamixated
  return (x + bias).gelu()

def EmbedLayerNormalization(input_ids: Tensor, segment_ids:Optional[Tensor]=None, word_embedding:Tensor=None, position_embedding:Tensor=None, segment_embedding:Optional[Tensor]=None, gamma=None, beta=None, mask:Optional[Tensor]=None, position_ids:Optional[Tensor]=None, epsilon=None, mask_index_type=None):
  # https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.EmbedLayerNormalization
  assert (segment_ids is None) is (segment_embedding is None)
  assert (mask is None) is (mask_index_type is None)
  assert mask is None, "functionality not supported yet"  # TODO
  input_shape = input_ids.shape
  seq_length = input_shape[1]
  compute_seg_emb = (segment_embedding is not None and segment_ids is not None)
  vocab_size, max_position_embeddings, type_vocab_size = word_embedding.shape[0], position_embedding.shape[0], (segment_embedding.shape[0] if compute_seg_emb else None)

  def embedding(x:Tensor, vocab_size, weight:Tensor) -> Tensor:  # TODO from nn.Embedding. Could probably upstream this to Tensor
    vocab_counter = Tensor.arange(vocab_size, dtype=x.dtype, requires_grad=False).reshape(1, 1, vocab_size).expand(*x.shape, vocab_size)
    return (vocab_counter == x.unsqueeze(2).expand(*x.shape, vocab_size)) @ weight

  # bert embedding layer
  if epsilon is None: epsilon = 1e-12
  if position_ids is None: position_ids = Tensor.arange(seq_length, requires_grad=False).unsqueeze(0).expand(*input_shape)
  wrd_embedding_res = embedding(input_ids, vocab_size, word_embedding)
  pos_embedding_res = embedding(position_ids, max_position_embeddings, position_embedding)
  seg_embedding_res = embedding(segment_ids, type_vocab_size, segment_embedding) if compute_seg_emb else None

  embedding_sum = wrd_embedding_res + pos_embedding_res
  if seg_embedding_res is not None: embedding_sum = embedding_sum + seg_embedding_res
  out = embedding_sum.layernorm(eps=epsilon) * gamma + beta
  return out, None, embedding_sum

def Attention(x:Tensor, weights, bias:Optional[Tensor]=None, mask_index:Optional[Tensor]=None, past:Optional[Tensor]=None, relative_position_bias:Optional[Tensor]=None, past_sequence_length:Optional[Tensor]=None, do_rotary=None, mask_filter_value=None, num_heads=None, past_present_share_buffer=None, qkv_hidden_sizes=None, scale=None, unidirectional=None):
  # https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Attention
  assert num_heads is not None  # required
  assert (qkv_hidden_sizes is None and past is not None) or (qkv_hidden_sizes is not None)
  assert relative_position_bias==do_rotary==past_sequence_length==mask_filter_value==past_present_share_buffer==scale==None, "functionality not supported yet"  # TODO strange params
  hidden_size, v_hidden_size = qkv_hidden_sizes[1:] if qkv_hidden_sizes is not None else 2*(weights.shape[1] // 3,)

  if unidirectional:  # gpt-style
    assert hidden_size == v_hidden_size
    xqkv = x.linear(weights, bias)
    xq, xk, xv = [xqkv._slice([None, None, (i*hidden_size, (i+1)*hidden_size)]) for i in range(3)]
  else:  # bert-style
    wq, wk, wv = weights[:,:hidden_size], weights[:,hidden_size:hidden_size+v_hidden_size], weights[:,hidden_size+v_hidden_size:]
    bq, bk, bv = (bias[:hidden_size], bias[hidden_size:hidden_size+v_hidden_size], bias[hidden_size+v_hidden_size]) if bias is not None else None
    xq, xk, xv = [x.linear(w, b) for w, b in zip((wq, wk, wv), (bq, bk, bv))]
  xq, xk, xv = [x.reshape(x.shape[0], x.shape[1], num_heads, -1).transpose(1, 2) for x in (xq, xk, xv)]

  if past is not None:
    xk, xv = Tensor.cat(past[0], xk, dim=-2), Tensor.cat(past[1], xv, dim=-2)
    present = Tensor.cat(xk.unsqueeze(0), xv.unsqueeze(0))

  def attn(query, key, value, attn_mask):
    query_length, key_length = query.shape[-2], key.shape[-2]
    cdim = max(query_length, key_length) + 1
    attn_weights = query @ key.transpose(-1, -2) / math.sqrt(value.shape[-1])
    # This is where Tensor.scaled_dot_product_attention differs:
    causal_mask = Tensor.ones((cdim, cdim), requires_grad=False, dtype=dtypes.bool).tril(0)[key_length - query_length : key_length, :key_length]
    masked = Tensor.where(causal_mask, attn_weights, -math.inf)
    if attn_mask is not None: masked = masked + attn_mask
    return masked.softmax(-1) @ value

  bsz, _, seq_len, _ = xq.shape
  out = attn(xq, xk, xv, mask_index).transpose(1, 2).reshape(bsz, seq_len, -1)
  return out, present

# **************** ai.onnx.preview.training Ops ****************

# TODO not entirely sure these optimizers are correct
def Adagrad(R, T, *inputs, decay_factor=0.0, epsilon=0.0, norm_coefficient=0.0):
  groups = len(inputs) // 3
  grouped_inputs = [inputs[i::groups] for i in range(groups)]
  r = to_python_const(R / (1 + T * decay_factor))
  ret = []
  for X, G, H in grouped_inputs:
    X.grad = norm_coefficient * X + G
    X.grad.requires_grad, H.requires_grad = False, False # TODO manually turning off requires_grad, see TODO under (domain == "ai.onnx.preview.training") in onnx.py
    H.assign(H.detach() + X.grad * X.grad).realize()
    H_adaptive = H.sqrt() + epsilon
    X.assign(X.detach() - r * X.grad / H_adaptive)
    ret.extend([X, H])
  ret = ret[::2] + ret[1::2]
  return tuple(ret)

def Momentum(R, T, *inputs, alpha, beta, mode, norm_coefficient):
  groups = len(inputs) // 3
  grouped_inputs = [inputs[i::groups] for i in range(groups)]
  T, R.requires_grad = to_python_const(T), False
  beta_adjusted = beta if T > 0 else 1
  ret = []
  for X, G, V in grouped_inputs:
    X.grad = (norm_coefficient * X + G).realize()
    X.grad.requires_grad, V.requires_grad = False, False
    V.assign(alpha * V + beta_adjusted * X.grad).realize()
    if mode == "standard": X.assign(X.detach() - R * V).realize()
    elif mode == "nesterov": X.assign(X.detach() - R * (X.grad + alpha + V)).realize()
    ret.extend([X, V])
  ret = ret[::2] + ret[1::2]
  return tuple(ret)

# copied from tinygrad/nn/optim.py: LAMB with some edits
def Adam(R, T, *inputs, alpha=0.9, beta=0.999, epsilon=0.0, norm_coefficient=0.0, norm_coefficient_post=0.0):
  groups = len(inputs) // 4
  grouped_inputs = [inputs[i::groups] for i in range(groups)]
  T, R.requires_grad = to_python_const(T), False
  ret = []
  for X, G, V, H in grouped_inputs:
    X.grad = (norm_coefficient * X + G).realize()
    V.requires_grad, H.requires_grad, X.grad.requires_grad = False, False, False
    V.assign(alpha * V + (1.0 - alpha) * X.grad).realize()
    H.assign(beta * H + (1.0 - beta) * (X.grad * X.grad)).realize()
    up = (V / (1.0 - alpha**T)) / ((H / (1.0 - beta**T)).sqrt() + epsilon) if T > 0 else V / (H.sqrt() + epsilon)
    X.assign(X.detach() - R * up).realize()
    X = (1 - norm_coefficient_post) * X
    ret.extend([X, V, H])
  ret = ret[::3] + ret[1::3] + ret[2::3]
  return tuple(ret)
