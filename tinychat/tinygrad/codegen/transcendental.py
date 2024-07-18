import math, functools
from typing import Tuple, List
from tinygrad.dtype import dtypes, DType
from tinygrad.codegen.uops import UOp

TRANSCENDENTAL_SUPPORTED_DTYPES = {dtypes.float16, dtypes.float32, dtypes.float64}

def _lazy_map_numbers(x:UOp, inf:UOp, _inf:UOp, nan:UOp, ratio:UOp):
  """replace inf -> inf, -inf -> _inf, nan -> nan, otherwise -> ratio"""
  return x.ne(math.inf).where(x.ne(x).where(nan, x.ne(-math.inf).where(ratio, _inf)), inf)
# *** helper functions for double/quad precision arithmetics ***
def dfadd2_f2_f2_f2(xx:UOp, xy:UOp, yx:UOp, yy:UOp) -> Tuple[UOp, UOp]: return xx + yx, xy + yy
def dfmul2_f2_f2_f2(xx:UOp, xy:UOp, yx:UOp, yy:UOp) -> Tuple[UOp, UOp]: return xx * yx, xx * yy + xy * yx
def dfdiv2_f2_f2_f2(nx:UOp, ny:UOp, dx:UOp, dy:UOp) -> Tuple[UOp, UOp]:
  t = dx.recip()
  qx = nx * t
  qy = (ny - qx * dy) * t
  return qx, qy
# *** helper functions for bit manipulation ***
def significand_bits(d:DType) -> int: return {dtypes.float64: 52, dtypes.float32: 23, dtypes.float16: 10}[d]
def exponent_bias(d:DType) -> int: return {dtypes.float64: 1022, dtypes.float32: 126, dtypes.float16: 14}[d]
def exponent_mask(d:DType) -> int: return {dtypes.float64: 0x7FF, dtypes.float32: 0xFF, dtypes.float16: 0x1F}[d]

def float_to_bits(d:UOp) -> UOp:
  assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
  cast_to = {dtypes.float64: dtypes.uint64, dtypes.float32: dtypes.uint32, dtypes.float16: dtypes.uint16}[d.dtype]
  return d.bitcast(cast_to)

def bits_to_float(d:UOp, float_dtype:DType) -> UOp:
  assert d.dtype in [dtypes.uint64, dtypes.uint32, dtypes.uint16]
  cast_to = {dtypes.uint64: dtypes.float64, dtypes.uint32: dtypes.float32, dtypes.uint16: float_dtype}[d.dtype]
  return d.bitcast(cast_to)
# **** utils ****
def shr(x:UOp, y:int) -> UOp: return x // (2**y)
def shl(x:UOp, y:int) -> UOp: return x * (2**y)

def rintk(d:UOp) -> UOp:
  """ceiling(d:float) -> int"""
  assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
  return_t = {dtypes.float64: dtypes.int64, dtypes.float32: dtypes.int32, dtypes.float16: dtypes.int16}[d.dtype]
  return (d + d.lt(0.0).where(d.const(-0.5), d.const(0.5))).cast(return_t)

def pow2if(q:UOp, float_dtype:DType):
  """cast(2^q, float_dtype) where q is any integer in the range of [-126, 127]"""
  assert q.dtype in (dtypes.int64, dtypes.int32, dtypes.int16, dtypes.uint32)
  final_dtype = {dtypes.int64: dtypes.float64, dtypes.int32: dtypes.float32, dtypes.int16: float_dtype, dtypes.uint32: dtypes.float32}[q.dtype]
  return shl((q + (exponent_bias(final_dtype)+1)), significand_bits(final_dtype)).bitcast(final_dtype)

def ilogb2k(d:UOp) -> UOp:
  """calculate the integer part of log2(d), where d is normalized fp value in the range of [0, +inf)."""
  assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
  dint = d.bitcast({dtypes.float64: dtypes.int64, dtypes.float32: dtypes.int32, dtypes.float16: dtypes.int16}[d.dtype])
  # -1 <= ilog2bk(d) <= 128
  # ((float_to_bits(d) >> significand_bits(dtype)) & exponent_mask(dtype)) - exponent_bias(dtype)
  return (shr(dint, significand_bits(d.dtype)) & exponent_mask(d.dtype)) - (exponent_bias(d.dtype)+1)

def ldexp3k(d:UOp, e:UOp) -> UOp:
  """d*2^e. e is a number obtained by casting an integer in the range [-127, 127] to a float. d is any float number."""
  assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES and e.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
  dtype = d.dtype
  cast_map = {dtypes.float64: dtypes.int64, dtypes.float32: dtypes.int32, dtypes.float16: dtypes.int16}
  e = e.cast(cast_map[d.dtype])
  m1 = d.bitcast(cast_map[d.dtype])
  m2 = shl(e, significand_bits(d.dtype))
  return (m1 + m2).bitcast(d.dtype).cast(dtype)

def ldexp2k(d:UOp, e:UOp) -> UOp:
  """d*2^e. much faster than ldexp3k but risky. d > 0 and d is not denormal."""
  assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES and e.dtype in (dtypes.int16, dtypes.int32, dtypes.int64)
  return (d * pow2if(shr(e, 1), d.dtype)) * pow2if(e - shr(e, 1), d.dtype)

def frexp(v:UOp) -> Tuple[UOp, UOp]:
  """frexp(v) -> (mantissa, exponent)"""
  assert v.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
  # m1 = masks for mantissa, m2 = masks to normalize the mantissa.
  m1 = {dtypes.float64: 0x000FFFFFFFFFFFFF, dtypes.float32: 0x807FFFFF, dtypes.float16: 0x83FF}[v.dtype]
  m2 = {dtypes.float64: 0x3FE0000000000000, dtypes.float32: 0x3F000000, dtypes.float16: 0x3C00}[v.dtype]
  bias = {dtypes.float64: 1022, dtypes.float32: 126, dtypes.float16: 15}[v.dtype]
  bits = float_to_bits(v)
  exponent = shr(bits, significand_bits(v.dtype)) & exponent_mask(v.dtype)
  exponent_zero = exponent.ne(0.0)
  result_f = bits_to_float((bits & m1) | m2, v.dtype)
  value = exponent_zero.where(result_f, v)
  exp = exponent + (-bias)
  exp = exponent_zero.where(exp, exp.const(0))
  if v.dtype == dtypes.float16: exp = exp.bitcast(dtypes.int16)
  return value, exp

def mla(x:UOp, y:UOp, z:UOp) -> UOp: return x * y + z

def polyN(u:UOp, s:UOp, coeffs:List[float]) -> UOp: return functools.reduce(lambda u,c: mla(u, s, u.const(c)), coeffs, u)
# *** reduction algorithms for sine ***
def payne_hanek_reduction(d:UOp) -> Tuple[UOp, UOp]:
  """
  Performs Payne-Hanek Reduction: computes the remainder of `d` modulo pi/2 for the values `d` where
    39800.0 <= d <= +Inf
  Returns a tuple of `(r, q)`:
  - `r`[d.dtype] is the reminder value corresponding to `round_to_nearest(x % pi/2)`.
    ensuring that `r` is in the range of [0, pi/2).
  - `q`[int32] is an integer taking values 0,1,2 or 3, corresponding to the quadrant of the original angle `d`.
  """
  assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
  two_over_pi_f = [0x00000000,0x28be60db,0x9391054a,0x7f09d5f4,0x7d4d3770,0x36d8a566,0x4f10e410]

  input_dtype: DType = d.dtype
  dtype_via = dtypes.float32 if d.dtype == dtypes.float16 else d.dtype
  acc_dtype = dtypes.uint64

  f, e = frexp(d)
  ia = (f.cast(dtype_via) * 4.294967296e9).cast(dtypes.uint64)
  i = shr(e.cast(dtypes.uint64), 5)
  e = (e.cast(dtypes.uint64) & 31).cast(dtypes.uint32)
  offset = -e + 32

  def _eq(arr:UOp, eq_to:int) -> UOp: return arr.ne(eq_to)
  def _take(an:UOp, offset:int, count:int=0) -> UOp:
    """an = two_over_pi_f[i+offset]"""
    if count+offset <= len(two_over_pi_f[0:-2]):
      an = _eq(i, count).where(_take(an, offset, count=count+1), an.const(two_over_pi_f[count+offset]))
    return an
  def _exact_pow2if(x): return pow2if(x, input_dtype).cast(acc_dtype)
  def _shl_lazy(x, y): return (x.cast(acc_dtype) * _exact_pow2if(y)).cast(dtypes.uint32)
  def _shr_lazy(x, y): return (x.cast(acc_dtype) // _exact_pow2if(y)).cast(dtypes.uint32)
  # a_n = (two_over_pi_f[Int(i) + n] << e) | (two_over_pi_f[Int(i) + n+1] >> (nbits - e))
  a1 = _take(i.const(0).cast(dtypes.uint32), 0)
  a2 = _take(i.const(0).cast(dtypes.uint32), 1)
  a3 = _take(i.const(0).cast(dtypes.uint32), 2)
  a4 = _take(i.const(0).cast(dtypes.uint32), 3)
  # Note: e >= 1 for all numbers d >= 1.0. assume e != 0
  hi = _shl_lazy(a1, e) | _shr_lazy(a2, offset)
  mi = _shl_lazy(a2, e) | _shr_lazy(a3, offset)
  lo = _shl_lazy(a3, e) | _shr_lazy(a4, offset)

  def _hp_mul(x:UOp, y:UOp) -> UOp: return x.cast(dtypes.uint64) * y.cast(dtypes.uint64)
  p = _hp_mul(ia, lo)
  p = _hp_mul(ia, mi) + shr(p, 32)
  p = shl(_hp_mul(ia, hi), 32) + p

  q = shr(p, 62).cast(dtypes.int32)
  p = p & 0x3fffffffffffffff
  r = (p.cast(dtype_via) * (3.4061215800865545e-19)).cast(input_dtype)

  # if fraction >= 0.5, r -= pi/2, q += 1
  return f.lt(0.5).where(r, r + r.const(-math.pi / 2)), f.lt(0.5).where(q, q + 1)

def cody_waite_reduction(d:UOp) -> Tuple[UOp, UOp]:
  """
  Performs Cody-Waite Reduction: computes the reminder of `d` modulo pi/2 for the values `d` where
      0 <= abs(d) <= 39800.0
  Returns a tuple of `(r, q)`, where the output format is the same as that of `payne_hanek_reduction`.
  """
  m_1_pi = 0.318309886183790671537767526745028724
  qdh = (d * (m_1_pi / 16777216)).cast(dtypes.int64).cast(d.dtype) * 16777216.0
  def _quadrant(x:UOp) -> UOp:
    if x.dtype == dtypes.float64: return rintk(mla(d, d.const(m_1_pi), -qdh)).cast(x.dtype)
    return rintk(x * m_1_pi).cast(x.dtype)
  def _reduce_d(x:UOp, q:UOp):
    if x.dtype == dtypes.float64:
      d = mla(qdh, x.const(-3.1415926218032836914), x)
      d = mla(q, x.const(-3.1415926218032836914), d)
      d = mla(qdh, x.const(-3.1786509424591713469e-08), d)
      d = mla(q, x.const(-3.1786509424591713469e-08), d)
      d = mla(qdh, x.const(-1.2246467864107188502e-16), d)
      d = mla(q, x.const(-1.2246467864107188502e-16), d)
      d = mla(qdh + q, x.const(-1.2736634327021899816e-24), d)
    elif x.dtype == dtypes.float16:
      # [FIXME] when reducing `d`, FP16 needs FP32 precision to achieve 1.0 ULP precision.
      d = _reduce_d(x.cast(dtypes.float32), q.cast(dtypes.float32)).cast(dtypes.float16)
    else:
      d = mla(q, x.const(-3.1414794921875), x)
      d = mla(q, x.const(-0.00011315941810607910156), d)
      d = mla(q, x.const(-1.9841872589410058936e-09), d)
      d = mla(q, x.const(-1.2154201256553420762e-10), d)
    return d
  return _reduce_d(d, (q := _quadrant(d))), q.cast(dtypes.int32)
# *** approximate sine on small angle. ***
def trig_poly(d:UOp, coeff32, coeff64):
  u = None
  s = d * d
  if d.dtype == dtypes.float64:
    s2 = s * s
    s4 = s2 * s2
    def __poly4(x:UOp, x2:UOp, c3, c2, c1, c0) -> UOp: return mla(x2, mla(x, x.const(c3), x.const(c2)), mla(x, x.const(c1), x.const(c0)))
    def __poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0) -> UOp: return mla(x4, __poly4(x, x2, c7, c6, c5, c4), __poly4(x, x2, c3, c2, c1, c0))
    u = __poly8(s, s2, s4, *coeff64[:-1])
    u = mla(u, s, d.const(coeff64[-1]))
  else:
    u = polyN(s.const(coeff32[0]), s, coeff32[1:])
  return mla(s, u * d, d)
# approximate sine on [-pi/2, pi/2]
def sin_poly(d:UOp) -> UOp: return trig_poly(d, [2.6083159809786593541503e-06, -0.0001981069071916863322258, 0.00833307858556509017944336, -0.166666597127914428710938], [-7.97255955009037868891952e-18, 2.81009972710863200091251e-15, -7.64712219118158833288484e-13, 1.60590430605664501629054e-10, -2.50521083763502045810755e-08, 2.75573192239198747630416e-06, -0.000198412698412696162806809, 0.00833333333333332974823815, -0.166666666666666657414808]) # noqa: E501

def sin_poly_small(d:UOp, q:UOp) -> UOp:
  def _ifand(n:int): return (q & n).ne(0)
  r = sin_poly(d)
  return r * _ifand(1).where(r.const(-1), r.const(1))

def sin_poly_large(d:UOp, q:UOp) -> UOp:
  def _ifand(n:int): return (q & n).ne(0)
  d = d + _ifand(1).where(d.const(math.pi / 2), d.const(0))
  r = sin_poly(d)
  return r * _ifand(2).where(r.const(-1), r.const(1))
# *** toplevel functions for xsin/xlog2/xexp2 ***
def xsin(d:UOp, fast:bool=False, switch_over:float=30.0) -> UOp:
  """
  Implements a 1.0 ULP approximation for UnaryOps.SIN.
  - fast=True assumes x <= switch_over.
  - switch_over is the threshold for switching to payne_hanek_reduction.
  """
  assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
  reduction_algo = cody_waite_reduction if fast else payne_hanek_reduction
  # mask +-inf/nan as zero
  x = _lazy_map_numbers(d, d.const(0.0), d.const(0.0), d.const(0.0), d)
  # x_sign = sign(x)
  x_sign = x.ne(0).where(x.lt(0).where(x.const(-1), x.const(1)), x.const(0))
  x_abs = x * x_sign
  r, q = reduction_algo(x_abs)
  if fast: result = sin_poly_small(r, q)
  else:
    # Payne Hanek Reduction assumes abs(x) >= pi/4, so for smaller values, use cody_waite_reduction.
    switch_over_map = x_abs.lt(switch_over)
    r_fast, q_fast = cody_waite_reduction(x_abs)
    r = switch_over_map.where(r_fast, r)
    q = switch_over_map.where(q_fast, q)
    result = switch_over_map.where(sin_poly_small(r, q), sin_poly_large(r, q))
  result = result * x_sign # adjusts the sign for abs(x).
  # sin(Inf) = NaN, sin(-Inf) = NaN, sin(NaN) = NaN
  return _lazy_map_numbers(d, d.const(math.nan), d.const(math.nan), d.const(math.nan), result)

def xexp2(x:UOp) -> UOp:
  """
  Implements a 1.0 ULP approximation for UnaryOps.EXP2
  - Paper: https://arxiv.org/pdf/2001.09258
  """
  assert x.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
  fp64_p = x.dtype == dtypes.float64
  # mask +=inf/nan as zero.
  d = _lazy_map_numbers(x, x.const(0.0), x.const(0.0), x.const(0.0), x)
  q = rintk(d)
  # s = d - round(d)
  s = d - q.cast(d.dtype)
  # a polynomial approximation with 13 non-zero terms in the range of [−(log 2)/2,(log 2)/2].
  if fp64_p:
    u = polyN(s.const(0.4434359082926529454e-9), s, [0.7073164598085707425e-8, 0.1017819260921760451e-6, 0.1321543872511327615e-5, 0.1525273353517584730e-4, 0.1540353045101147808e-3, 0.1333355814670499073e-2, 0.9618129107597600536e-2, 0.5550410866482046596e-1, 0.2402265069591012214e+0, 0.6931471805599452862e+0, 0.1000000000000000000e+1]) # noqa: E501
  else:
    u = polyN(s.const(0.1535920892e-3), s, [0.1339262701e-2, 0.9618384764e-2, 0.5550347269e-1, 0.2402264476e+0, 0.6931471825e+0, 0.1000000000e+1])
  u = ldexp2k(u, q) # u*2^q
  upper = {dtypes.float64: 1024, dtypes.float32: 128, dtypes.float16: 23.0}[d.dtype]
  lower = {dtypes.float64: -2000, dtypes.float32: -150, dtypes.float16: -22}[d.dtype]
  # Replace x >= upper with +inf
  u = d.ne(upper).where(u, d.const(math.inf))
  u = d.lt(upper).where(u, d.const(math.inf))
  # Replace x <= lower with zero.
  u = d.lt(lower).where(d.const(0.0), u)
  # x=NaN never satisfies x < Inf. (for fastmode)
  u = d.lt(math.inf).where(u, u.const(math.nan))
  # exp2(Inf) = Inf, exp2(-Inf) = 0, exp2(NaN) = NaN
  return _lazy_map_numbers(x, x.const(math.inf), x.const(0.0), x.const(math.nan), u)

def xlog2(d:UOp) -> UOp:
  """
  Implements a 1.0 ULP approximation for UnaryOps.LOG2
  Paper: https://arxiv.org/pdf/2001.09258
  """
  assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
  fp64_p = d.dtype == dtypes.float64
  FLT_MIN = d.const(1e-6 if d.dtype == dtypes.float16 else 1e-4)
  d_orig = d
  denormal_map = d.lt(FLT_MIN)
  for _ in range(8): d = denormal_map.where(d * (2 ** 8), d)

  e = ilogb2k(d * (1.0 / 0.75)).cast(d.dtype)
  m = ldexp3k(d, -e)
  e = denormal_map.where(e + (-64), e)

  if fp64_p:
    x = (m - 1.0) * (m + 1.0).recip()
    x2 = x * x
    t = polyN(x.const(0.2211941750456081490e+0), x2, [0.2200768693152277689e+0, 0.2623708057488514656e+0, 0.3205977477944495502e+0, 0.4121985945485324709e+0, 0.5770780162997058982e+0, 0.96179669392608091449]) # noqa: E501
    s_hi, s_lo = dfadd2_f2_f2_f2(e, e.const(0), *dfmul2_f2_f2_f2(t.const(2.885390081777926774), t.const(0), x, x.const(0)))
    r = mla(t, x * x2, s_hi + s_lo)
  else:
    xx, xy = dfdiv2_f2_f2_f2(*dfadd2_f2_f2_f2(m.const(-1), m.const(0), m, m.const(0)), *dfadd2_f2_f2_f2(m.const(1), m.const(0), m, m.const(0)))
    x2 = xx * xx
    t = polyN(d.const(0.4374550283e+0), x2, [0.5764790177e+0, 0.9618012905120])
    sx, sy = dfadd2_f2_f2_f2(e, e.const(0), *dfmul2_f2_f2_f2(xx, xy, xx.const(2.8853900432586669922), xy.const(3.2734474483568488616e-08)))
    sx, sy = dfadd2_f2_f2_f2(sx, sy, x2.const(0), (x2 * xx) * t)
    r = sx + sy
  # log2(Inf) = Inf
  r = d_orig.ne(math.inf).where(r, r.const(math.inf))
  # log2(x=-0.01) = NaN. where x < 0
  r = d_orig.lt(-0.0).where(r.const(math.nan), r)
  # log2(0) = -Inf, but we will compare using the value of y because 1e-200==0 is true.
  # log2_zero = the value of unmasked xlog2(0.0).
  log2_zero = {dtypes.float64: -1087, dtypes.float32: -191, dtypes.float16: -79, None: -math.inf}[d.dtype]
  r = r.ne(log2_zero).where(r, r.const(-math.inf))
  # log(NaN) = NaN, using for all real number x, either of x < Inf, x == Inf becomes True.
  r = d_orig.lt(math.inf).where(r, d_orig.ne(math.inf).where(d.const(math.nan), d))
  # log(-0.0) = -Inf. In certain devices like PTX, x == -0.0 won't be true. so making reciprocal.
  return d_orig.recip().ne(-math.inf).where(r, r.const(-math.inf))
