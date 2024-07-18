import numpy as np
from tinygrad import Device, dtypes, Tensor

# TODO: will be better when tinygrad does math in the target dtype, can remove the floor and use a mul
def bit_extract(x, s, e) -> Tensor:
  # extract the top bits we don't want
  top_bits = (x / (1<<(s+1))).floor() * (1<<(s+1))
  x = (x - top_bits) / (1<<e)
  return x.contiguous()

def u16_to_f16(x):
  sign = bit_extract(x, 15, 15).float()
  exponent = bit_extract(x, 14, 10).float()
  fraction = bit_extract(x, 9, 0).float()
  return sign.where(-1, 1) * exponent.where((exponent - 15).exp2() * (1 + fraction / 0x400), 6.103515625e-5 * (fraction / 0x400))

def u32_to_f16(oo):
  oo1 = (oo/0x10000).floor().contiguous()
  # TODO: this is wrong and unextractable until we do this math in u32
  oo2 = (oo-(oo1*0x10000)).floor().contiguous()
  f1 = u16_to_f16(oo1)
  f2 = u16_to_f16(oo2)
  return Tensor.cat(f2.reshape(-1, 1), f1.reshape(-1, 1), dim=1).flatten()

if __name__ == "__main__":
  # random float16
  Tensor.manual_seed(2)
  a = Tensor.randn(100, dtype=dtypes.float16)

  # this converts it to u32 on disk
  oo = a.to("disk:/tmp/f16").cast(dtypes.uint32)[:50].to(Device.DEFAULT).realize()

  # convert to 2xf16 using tinygrad math ops
  f16 = u32_to_f16(oo)

  ref = a.numpy()
  out = f16.numpy().astype(np.float16)
  print(ref-out)

  np.testing.assert_allclose(ref, out)
