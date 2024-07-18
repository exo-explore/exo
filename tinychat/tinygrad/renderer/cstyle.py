from typing import Dict, List, Optional, Tuple, Union, DefaultDict, cast, Literal, Callable
import os, math
from collections import defaultdict, Counter
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.helpers import strip_parens, getenv, prod, dedup
from tinygrad.dtype import ImageDType, dtypes, DType, PtrDType, ConstType
from tinygrad.codegen.uops import UOps, UOp
from tinygrad.codegen.uopgraph import UOpGraph
from tinygrad.renderer import Renderer, TensorCore

class CStyleLanguage(Renderer):
  kernel_prefix: str = ""
  buffer_prefix: str = ""
  buffer_suffix: str = ""
  smem_align: str = ""
  smem_prefix: str = ""
  smem_prefix_for_cast: bool = True
  arg_int_prefix: str = "const int"
  barrier: str = ""
  code_for_workitem: Dict[Union[Literal["g"], Literal["l"], Literal["i"]], Callable] = {}
  extra_args: List[str] = []
  float4: Optional[str] = None
  uses_vload: bool = False
  uses_ptr_arithmetic: bool = False
  type_map: Dict[DType, str] = {}
  code_for_op: Dict = {
    UnaryOps.NEG: lambda x,dtype: f"(!{x})" if dtype == dtypes.bool else f"(-{x})", UnaryOps.SQRT: lambda x,dtype: f"sqrt({x})",
    UnaryOps.RECIP: lambda x,dtype: f"(1/{x})",
    UnaryOps.EXP2: lambda x,dtype: f"exp2({x})", UnaryOps.LOG2: lambda x,dtype: f"log2({x})", UnaryOps.SIN: lambda x,dtype: f"sin({x})",
    BinaryOps.ADD: lambda a,b,dtype: f"({a}+{b})", BinaryOps.MAX: lambda a,b,dtype: f"max({a},{b})",
    BinaryOps.IDIV: lambda a,b,dtype: f"({a}/{b})", BinaryOps.MUL: lambda a,b,dtype: f"({a}*{b})", BinaryOps.MOD: lambda a,b,dtype: f"({a}%{b})",
    BinaryOps.CMPLT: lambda a,b,dtype: f"({a}<{b})", BinaryOps.CMPNE: lambda a,b,dtype: f"({a}!={b})", BinaryOps.XOR: lambda a,b,dtype: f"({a}^{b})",
    BinaryOps.AND: lambda a,b,dtype: f"({a}&{b})", BinaryOps.OR: lambda a,b,dtype: f"({a}|{b})",
    TernaryOps.WHERE: lambda a,b,c,dtype: f"({a}?{b}:{c})"}

  # returns a str expression of the casted xs with the given type
  def render_cast(self, x:str, var_dtype:DType, bitcast=False) -> str:
    if bitcast: return f"(*(({self.buffer_prefix}{self.render_dtype(var_dtype)}*)&{x}))"
    return f"({self.render_dtype(var_dtype)})({x})"

  # returns a str expression of the vectorized xs with the given type
  def render_vectorize(self, x:List[str], var_dtype:DType) -> str:
    assert len(x) == var_dtype.count, f"cast is wrong size {len(x)} != {var_dtype.count}"
    assert self.float4 is not None, "vectorized cast is not supported on this platform"
    return f"{self.float4.replace('float4', self.render_dtype(var_dtype))}({','.join(x)})"

  # returns a str expression of the const with the given type
  def render_const(self, x:ConstType, dtype:DType) -> str:
    if math.isnan(x): val = "NAN"
    elif math.isinf(x): val = ("-" if x < 0 else "") + "INFINITY"
    elif dtype.scalar() == dtypes.bool: val = "1" if x else "0"
    elif dtype.scalar() == dtypes.float: val = f"{x}f"
    else: val = str(x)
    if dtype.count > 1: return self.render_vectorize([val] * dtype.count, dtype)
    return (self.render_cast(val, dtype) if dtype not in [dtypes.float, dtypes.int, dtypes.bool] else val)

  # returns a str expression of the loaded value with the output type
  def render_load(self, output_dtype, buf_name, buf_dtype, idx, local=False) -> str:
    if isinstance(buf_dtype, ImageDType):
      assert output_dtype == dtypes.float.vec(4), f"images must be float4, getting {output_dtype}"
      return f"read_imagef({buf_name}, smp, {idx})"
    if self.uses_vload and buf_dtype.scalar() == dtypes.float16 and output_dtype.scalar() != dtypes.float16:
      return f"vload_half{'' if output_dtype.count == 1 else str(output_dtype.count)}(0, {buf_name}+{idx})"
    if output_dtype.count > 1:
      return f"*(({self.smem_prefix if local and self.smem_prefix_for_cast else self.buffer_prefix}{self.render_dtype(buf_dtype)}{output_dtype.count}*)({buf_name}+{idx}))"  # noqa: E501
    return f"*({buf_name}+{idx})" if self.uses_ptr_arithmetic else f"{buf_name}[{idx}]"

  def get_kernel_modifier(self, uops:UOpGraph) -> str: return ""
  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,Tuple[DType,bool]]], uops:UOpGraph, prefix=None) -> str:
    tmp = "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n" if any(isinstance(dtype, ImageDType) for _,(dtype,_) in bufs) else ""  # noqa: E501
    buftypes = [(name,f"{'write_only' if mutable else 'read_only'} image2d_t" if dtype.name.startswith('image') else
                ("" if mutable else "const ")+self.buffer_prefix+self.render_dtype(dtype)+"*"+self.buffer_suffix if isinstance(dtype, PtrDType) else
                self.arg_int_prefix if dtype == dtypes.int else None) for name,(dtype,mutable) in bufs]
    prg = ''.join([f"{self.kernel_prefix}void {self.get_kernel_modifier(uops)}{function_name}(",] +
    [', '.join([f'{t} {name}' for name,t in buftypes] + self.extra_args)] +
    [") {\n" + tmp] + ['\n'.join(kernel), "\n}"])
    return prg if prefix is None else "\n".join(prefix)+f"\n{prg}"

  # returns a str statement that does the store
  def render_store(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType, idx:str, local=False) -> str:
    if isinstance(buf_dtype, ImageDType):
      assert var_dtype == dtypes.float.vec(4), f"images must be float4, getting {var_dtype}"
      return f"write_imagef({buf_name}, {idx}, {var_name});"
    if self.uses_vload and buf_dtype.scalar() == dtypes.float16 and var_dtype.scalar() != dtypes.float16:
      return f"vstore_half{'' if var_dtype.count == 1 else str(var_dtype.count)}({var_name}, 0, {buf_name}+{idx});"
    if var_dtype.count > 1:
      prefix = self.smem_prefix if local and self.smem_prefix_for_cast else self.buffer_prefix
      return f"*(({prefix}{self.render_dtype(buf_dtype)}{var_dtype.count}*)({buf_name}+{idx})) = {var_name};"
    return f"*({buf_name}+{idx}) = {var_name};" if self.uses_ptr_arithmetic else f"{buf_name}[{idx}] = {var_name};"

  def render_local(self, name:str, dtype:DType, size:int): return self.smem_align + self.smem_prefix + f"{self.render_dtype(dtype)} {name}[{size}];"
  def render_dtype(self, var_dtype:DType) -> str: return self.type_map.get(var_dtype, var_dtype.name)

  def render(self, name:str, uops:UOpGraph) -> str:
    kernel = []
    bufs: List[Tuple[str, Tuple[DType, bool]]] = []
    depth = 1
    def kk(s): kernel.append("  "*depth+s)

    c: DefaultDict[str, int] = defaultdict(int)
    r: Dict[UOp, str] = {}

    def ssa(prefix:str, u:Optional[UOp]=None):
      nonlocal c, r
      ret = f"{prefix}{c[prefix]}"
      if u is not None: r[u] = ret
      c[prefix] += 1
      return ret

    child_count = Counter(v for ru in uops for v in ru.src)

    seen_vars = set()
    for u in uops:
      uop,dtype,src,args = u.op,u.dtype,u.src,u.arg
      # these four uops don't have output dtypes
      if uop is UOps.IF:
        kk(f"if ({r[src[0]]}) {{")
        depth += 1
      elif uop is UOps.BARRIER: kk(self.barrier)
      elif uop in {UOps.ENDRANGE, UOps.ENDIF}:
        depth -= 1
        kk("}")
      elif uop is UOps.STORE:
        assert src[0].dtype is not None and src[2].dtype is not None
        rendered_store = self.render_store(r[src[0]], src[0].dtype, r[src[2]], src[2].dtype, strip_parens(r[src[1]]), src[0].op is UOps.DEFINE_LOCAL)
        kk(f"if ({r[src[3]]}) {{ {rendered_store} }}" if len(src) > 3 else rendered_store)
      else:
        assert dtype is not None, f"None dtype for uop {uop}"
        if uop is UOps.RANGE:
          kk(f"for (int {(expr := ssa('ridx',u))} = {r[src[0]]}; {expr} < {r[src[1]]}; {expr}++) {{")
          depth += 1
        elif uop is UOps.ALU:
          # remove parens if ALU types are the same. TODO: can do more here
          if args in {BinaryOps.ADD,BinaryOps.MUL,BinaryOps.XOR}: operands = [strip_parens(r[v]) if v.arg == args else r[v]for v in src]
          else: operands = [r[v] for v in src]
          val = self.code_for_op[args](*operands, dtype)
          assert child_count[u] != 0, f"childless ALU op found {u}"
          # TODO: fix index rendering issue. fix clang nested max macro issue
          if child_count[u] <= 1 and args is not BinaryOps.MAX and not getenv("EXPAND_SSA"): r[u] = val
          else: kk(f"{self.render_dtype(dtype)} {ssa('alu',u)} = {val};")
        elif uop is UOps.SPECIAL:
          kk(f"int {args[1]} = {self.code_for_workitem[args[1][0]](args[0])}; /* {args[2]} */")
          r[u] = args[1]
        elif uop is UOps.LOAD:
          val = self.render_load(dtype, r[src[0]], src[0].dtype, strip_parens(r[src[1]]), src[0].op is UOps.DEFINE_LOCAL)
          # NOTE: this relies on the load not happening if it's in the unselected branch
          if len(src) > 3 and src[2].op is UOps.ALU: val = self.code_for_op[TernaryOps.WHERE](r[src[2]], val, r[src[3]], dtype)
          kk(f"{self.render_dtype(dtype)} {ssa('val',u)} = {val};")
        elif uop is UOps.PHI:
          kk(f"{r[src[0]]} = {r[src[1]]};")
          r[u] = r[src[0]]
        elif uop in {UOps.CAST, UOps.BITCAST, UOps.VECTORIZE}:
          assert len(src) == 1 or (uop is UOps.VECTORIZE and len(src) > 1), "Invalid source length for operation"
          if uop is UOps.BITCAST:
            precast = ssa('precast')
            kk(f"{self.render_dtype(cast(DType, src[0].dtype))} {precast} = {r[src[0]]};")
            val = self.render_cast(precast, dtype, bitcast=True)
          elif uop is UOps.CAST: val = self.render_cast(r[src[0]], dtype, bitcast=False)
          else: val = self.render_vectorize([r[x] for x in src], dtype)
          if child_count[u] <= 1: r[u] = val
          else: kk(f"{self.render_dtype(dtype)} {ssa('cast',u)} = {val};")
        elif uop is UOps.DEFINE_LOCAL:
          kk(self.render_local(args[0], dtype, args[1]))
          r[u] = args[0]
        elif uop is UOps.DEFINE_VAR:
          assert args.expr not in seen_vars, f"duplicate variable {args.expr}"
          seen_vars.add(args.expr)
          bufs.append((args.expr, (dtype,False)))
          r[u] = args.expr
        elif uop is UOps.DEFINE_GLOBAL:
          bufs.append((nm:=f"data{args[0]}", (dtype,args[1])))
          r[u] = nm
        elif uop is UOps.WMMA: kk(f"{self.render_dtype(dtype)} {ssa('wmma',u)} = __{args[0]}({r[src[0]]}, {r[src[1]]}, {r[src[2]]});")
        elif uop is UOps.DEFINE_ACC: kk(f"{self.render_dtype(dtype)} {ssa('acc',u)} = {self.render_const(src[0].arg, dtype)};")
        elif uop is UOps.CONST: r[u] = self.render_const(args, dtype) if args >= 0 else f"({self.render_const(args, dtype)})"
        elif uop is UOps.GEP:
          assert src[0].dtype is not None
          from_ssa = src[0].op in {UOps.LOAD, UOps.WMMA, UOps.DEFINE_ACC}
          r[u] = (r[src[0]] if from_ssa else f"{(r[src[0]])}") + (f"[{args}]" if src[0].dtype.count > 4 else f".{'xyzw'[args]}")
        else: raise RuntimeError(f"failed to render {u}")

    return self.render_kernel(name, kernel, bufs, uops)

class ClangRenderer(CStyleLanguage):
  device = "CLANG"
  supports_float4 = False
  has_local = False
  global_max = None

  # language options
  buffer_suffix = " restrict"
  type_map = {dtypes.bool:"_Bool", dtypes.half:"__fp16"}
  code_for_op = {**CStyleLanguage().code_for_op, BinaryOps.MAX: lambda a,b,dtype: f"(({a}>{b})?{a}:{b})"}

class OpenCLRenderer(CStyleLanguage):
  device = "GPU"

  # language options
  kernel_prefix = "__kernel "
  buffer_prefix = "__global "
  smem_align = "__attribute__ ((aligned (16))) "
  smem_prefix = "__local "
  barrier = "barrier(CLK_LOCAL_MEM_FENCE);"
  float4 = "(float4)"
  code_for_workitem = {"g": lambda x: f"get_group_id({x})", "l": lambda x: f"get_local_id({x})", "i": lambda x: f"get_global_id({x})"}
  uses_vload = True
  type_map = { dtypes.uint8: "uchar", dtypes.uint32: "uint", dtypes.uint16: "ushort", dtypes.uint64: "ulong" }
  def render_cast(self, x, var_dtype, bitcast=False) -> str:
    return f"as_{self.render_dtype(var_dtype)}({x})" if bitcast else super().render_cast(x, var_dtype)

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    if any(uop.dtype == dtypes.half for uop in uops): prefix = ["#pragma OPENCL EXTENSION cl_khr_fp16 : enable"]
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

class MetalRenderer(CStyleLanguage):
  device = "METAL"
  shared_max = 32768
  tensor_cores = [TensorCore(dims=(8,8,8), threads=[(0,2),(1,4),(0,2),(1,2)], thread_local_sizes=[[2],[2],[2]], dtype_in=di, dtype_out=do) for (di, do) in [(dtypes.float, dtypes.float), (dtypes.half, dtypes.float), (dtypes.half, dtypes.half)]] # noqa: E501
  def __init__(self): self.tensor_cores = MetalRenderer.tensor_cores if os.uname().machine == "arm64" else []

  # language options
  kernel_prefix = "kernel "
  buffer_prefix = "device "
  smem_prefix = "threadgroup "
  arg_int_prefix = "constant int&"
  barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);"
  float4 = "float4"
  uses_ptr_arithmetic = True
  code_for_workitem = {"g": lambda x: f"gid.{chr(120+x)}", "l": lambda x: f"lid.{chr(120+x)}"}
  # uint3 used for gid/lid - TODO: this should probably be `ushort3 lid [[thread_position_in_threadgroup]]`
  extra_args = ['uint3 gid [[threadgroup_position_in_grid]]', 'uint3 lid [[thread_position_in_threadgroup]]']
  type_map = {dtypes.bfloat16: "bfloat"}
  code_for_op = {**CStyleLanguage().code_for_op,
    BinaryOps.MAX: lambda a,b,dtype: f"(bfloat)max((float){a},(float){b})" if dtype == dtypes.bfloat16 else f"max({a},{b})",
    UnaryOps.SQRT: lambda x,dtype: f"(bfloat)sqrt({x})" if dtype == dtypes.bfloat16 else f"sqrt({x})",
    UnaryOps.EXP2: lambda x,dtype: f"(bfloat)exp2({x})" if dtype == dtypes.bfloat16 else f"exp2({x})",
    UnaryOps.LOG2: lambda x,dtype: f"(bfloat)log2({x})" if dtype == dtypes.bfloat16 else f"log2({x})",
    UnaryOps.SIN: lambda x,dtype: f"(bfloat)precise::sin({x})" if dtype == dtypes.bfloat16 else f"precise::sin({x})",}

  def render_cast(self, x:str, var_dtype:DType, bitcast=False) -> str:
    return f"as_type<{self.render_dtype(var_dtype)}>({x})" if bitcast else super().render_cast(x, var_dtype)

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
    prefix, wmma_args = ["#include <metal_stdlib>","using namespace metal;"], set([uop.arg for uop in uops if uop.op is UOps.WMMA])
    for arg in wmma_args: prefix.append(f"""{arg[3].name}2 __{arg[0]}({arg[2].name}2 m, {arg[2].name}2 n, {arg[3].name}2 o) {{
  simdgroup_{arg[3].name}8x8 a,b,c; a.thread_elements()[0] = m.x; a.thread_elements()[1] = m.y; b.thread_elements()[0] = n.x;
  b.thread_elements()[1] = n.y; c.thread_elements()[0] = o.x; c.thread_elements()[1] = o.y; simdgroup_multiply_accumulate(c, a, b, c);
  return {arg[3].name}2(c.thread_elements()[0], c.thread_elements()[1]);\n}}""")
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

code_for_op_half = {UnaryOps.RECIP: lambda x,dtype: f"hrcp({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"1/{x}",
                    BinaryOps.MAX: lambda a,b,dtype: f"__hmax({a},{b})" if dtype in (dtypes.half, dtypes.bfloat16) else f"max({a},{b})",
                    UnaryOps.SQRT: lambda x,dtype: f"hsqrt({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"sqrt({x})",
                    UnaryOps.SIN: lambda x,dtype: f"hsin({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"sin({x})",
                    UnaryOps.LOG2: lambda x,dtype: f"hlog2({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"log2({x})",
                    UnaryOps.EXP2: lambda x,dtype: f"hexp2({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"exp2({x})",}

_nms = "xyzwabcdefghijkl"
def _make_cuda_dtype(base_type, name, cnt):
  vec, elems, header = f"{name}{cnt}", ', '.join(_nms[:cnt]), ', '.join([f"{base_type} {x}" for x in _nms[:cnt]])
  return f"struct {vec} {{ {base_type} {elems}; }}; __device__ {vec} make_{vec}({header}) {{ {vec} r={{{elems}}}; return r; }}"

class CUDARenderer(CStyleLanguage):
  device = "CUDA"
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 64)
  shared_max = 49152
  tensor_cores = [TensorCore(dims=(8,16,16), threads=[(0,2),(0,2),(1,2),(1,2),(0,2)], thread_local_sizes=[[2,2,2],[2,2],[2,2]], dtype_in=di, dtype_out=do) for (di, do) in ([(dtypes.half, dtypes.float), (dtypes.bfloat16, dtypes.float)])]  # noqa: E501
  def __init__(self, arch:str): self.tensor_cores = CUDARenderer.tensor_cores if int(arch[3:]) >= 80 else []

  # language options
  kernel_prefix = "extern \"C\" __global__ "
  smem_prefix = "__shared__ "
  smem_prefix_for_cast = False
  barrier = "__syncthreads();"
  float4 = "make_float4"
  code_for_workitem = {"g": lambda x: f"blockIdx.{chr(120+x)}", "l": lambda x: f"threadIdx.{chr(120+x)}",
                       "i": lambda x: f"(blockIdx.{chr(120+x)}*blockDim.{chr(120+x)}+threadIdx.{chr(120+x)})"}
  code_for_op = {**CStyleLanguage().code_for_op, **code_for_op_half}
  type_map = {dtypes.bfloat16: "nv_bfloat16"}

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
    # TODO: why is dtypes.bfloat16.name == "__bf16"? would be easier not override dtypes.name
    dt_map = { dtypes.float: ("float","f32"), dtypes.half: ("half","f16"), dtypes.bfloat16: ("bfloat16","bf16"), }

    prefix = ["#define INFINITY (__int_as_float(0x7f800000))","#define NAN (__int_as_float(0x7fffffff))"]
    if any(uop.dtype == dtypes.half for uop in uops):
      prefix += ["#include <cuda_fp16.h>"] + [_make_cuda_dtype("half", "half", x) for x in [4, 8]]

    if any(uop.dtype == dtypes.bfloat16 for uop in uops):
      prefix += ["#include <cuda_bf16.h>"] + [_make_cuda_dtype("nv_bfloat16", "bfloat16", x) for x in [4, 8]]

    # TODO: this has to be way better to generate for arbitrary M,N,K: use arg[1] for MNK, use arg[4] for vec sizes, encode register packing
    for arg in dedup([uop.arg for uop in uops if uop.op is UOps.WMMA]):
      fn, ti, to, ci, co = arg[0], dt_map[arg[2]][0], dt_map[arg[3]][0], dt_map[arg[2]][1], dt_map[arg[3]][1]
      prefix.append(f"""__device__ {to}4 __{fn}({ti}8 a, {ti}4 b, {to}4 c) {{ int *a_pk = (int *) (&a), *b_pk = (int *) (&b);
asm( "mma.sync.aligned.m16n8k16.row.col.{co}.{ci}.{ci}.{co} {{ %0, %1, %2, %3 }}, {{ %4, %5, %6, %7 }}, {{ %8, %9 }}, {{ %0, %1, %2, %3 }};"
  : "+f"(c.x), "+f"(c.y), "+f"(c.z), "+f"(c.w) : "r"(a_pk[0]), "r"(a_pk[1]), "r"(a_pk[2]),  "r"(a_pk[3]), "r"(b_pk[0]), "r"(b_pk[1]) );
return c;}}""")

    return super().render_kernel(function_name, kernel, bufs, uops, prefix=prefix)

code_for_op_hip = { UnaryOps.SQRT: lambda x,dtype: f"__ocml_sqrt_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
                    UnaryOps.SIN: lambda x,dtype: f"__ocml_sin_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
                    UnaryOps.LOG2: lambda x,dtype: f"__ocml_log2_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
                    UnaryOps.EXP2: lambda x,dtype: f"__ocml_exp2_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
                    # TODO: MAX with int uses fmax_f32?
                    BinaryOps.MAX: lambda a,b,dtype: f"__ocml_fmax_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32) }({a},{b})",}

def _make_hip_code_for_op():
  def wrapper(key, func):
    def cast_bf16(*args):
      if args[-1] == dtypes.bfloat16:
        operands = tuple(f"(float)({arg})" for arg in (args[1:-1] if key is TernaryOps.WHERE else args[:-1]))
        return f"(hip_bfloat16)({func(*(((args[0],) if key is TernaryOps.WHERE else ()) + operands), dtypes.float)})"
      return func(*args)
    return cast_bf16
  return { k:wrapper(k,v) for k,v in {**CStyleLanguage().code_for_op, **code_for_op_hip}.items() }

def _make_hip_dtype(base_type, name, cnt):
  elems, header = ', '.join(_nms[:cnt]), ', '.join([f"{base_type} {x}" for x in _nms[:cnt]])
  return f"typedef {base_type} {name}{cnt} __attribute__((ext_vector_type({cnt})));\n" + \
         f"static inline __attribute__((device)) {name}{cnt} make_{name}{cnt}({header}) {{ return {{{elems}}}; }}"

class AMDRenderer(CStyleLanguage):
  device = "AMD"
  shared_max = 65536
  tensor_cores = [TensorCore(dims=(16,16,16), threads=[(0,8),(0,2),(1,2)], thread_local_sizes=[[16],[16],[4,2]], dtype_in=di, dtype_out=do) for (di, do) in [(dtypes.half, dtypes.float), (dtypes.half, dtypes.half)]] # noqa: E501

  # language options
  kernel_prefix = """extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_size(unsigned int);
extern "C" {\n""" + "".join([
f"""  __attribute__((device)) __attribute__((const)) {dt} __ocml_fmax_f{n}({dt}, {dt});
  __attribute__((device)) __attribute__((pure)) {dt} __ocml_exp2_f{n}({dt});
  __attribute__((device)) __attribute__((pure)) {dt} __ocml_log2_f{n}({dt});
  __attribute__((device)) __attribute__((const)) {dt} __ocml_sqrt_f{n}({dt});
  __attribute__((device)) {dt} __ocml_sin_f{n}({dt});\n""" for dt,n in [("float",32), ("double",64), ("_Float16",16)]]) +\
'}\nextern "C" __attribute__((global))'
  code_for_workitem = {"g": lambda x: f"__ockl_get_group_id({x})", "l": lambda x: f"__ockl_get_local_id({x})",
                       "i": lambda x: f"(__ockl_get_group_id({x})*__ockl_get_local_size({x})+__ockl_get_local_id({x}))"}
  code_for_op = _make_hip_code_for_op()
  smem_prefix = "__attribute__((shared))"
  barrier = '__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");' + '__builtin_amdgcn_s_barrier();' + \
            '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");'
  float4 = "make_float4"
  uses_ptr_arithmetic = False  # NOTE: this fixes TestLinearizerOverflowAlt
  type_map = {dtypes.bfloat16: "hip_bfloat16"}

  def render_kernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
    prefix = ["#define INFINITY (__builtin_inff())", "#define NAN (__builtin_nanf(\"\"))", "typedef long unsigned int size_t;"]
    vec_dts = [("float", "float", 2), ("float", "float", 4), ("float", "float", 8), ("signed int", "int", 4), ("signed int", "int", 2)]

    # TODO: add BF16 vec dts
    if any(uop.dtype == dtypes.bfloat16 for uop in uops): prefix.append("""
struct hip_bfloat16 {
  unsigned short data;
  inline __attribute__((device)) hip_bfloat16(float val) {
    union { float fp32; unsigned int u32; } u = {val};
    if (~u.u32 & 0x7f800000) { u.u32 += 0x7fff + ((u.u32 >> 16) & 1); } else if (u.u32 & 0xffff) { u.u32 |= 0x10000; }
    data = (u.u32 >> 16);
  }
  inline __attribute__((device)) operator float() const {
    unsigned int uval = data << 16;
    return *reinterpret_cast<float*>(&uval);
  }
};
static inline __attribute__((device)) bool operator<(hip_bfloat16 a, hip_bfloat16 b) { return ((float)a) < ((float)b); }
static inline __attribute__((device)) bool operator==(hip_bfloat16 a, hip_bfloat16 b) { return ((float)a) == ((float)b); }
""")

    if any(uop.dtype == dtypes.half for uop in uops):
      prefix.append("#define half _Float16")
      vec_dts += [("_Float16", "half", 2), ("_Float16", "half", 4), ("_Float16", "half", 8), ("_Float16", "half", 16)]

    prefix += [_make_hip_dtype(*x) for x in vec_dts]

    for arg in dedup([uop.arg for uop in uops if uop.op is UOps.WMMA]): # TODO: handle TCs f32_bf16 and bf16_bf16 w/ wrapper
      if arg[3] == dtypes.float: prefix.append(f"#define __{arg[0]} __builtin_amdgcn_wmma_f32_16x16x16_f16_w32")
      else: prefix.append(f"static inline __attribute__((device)) half8 __{arg[0]}"+"""(half16 a, half16 b, half8 c) {
  half16 c_frag = {}; half8 d; for (int n = 0; n < 8; n++) { c_frag[n*2] = c[n]; }
  c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c_frag, false);
  for (int n = 0; n < 8; n++) { d[n] = c_frag[n*2]; } return d;\n}""")
    return super().render_kernel(function_name, kernel, bufs, uops, prefix)

  def get_kernel_modifier(self, uops:UOpGraph) -> str:
    requiredMaxThreadsPerBlock = prod(u.arg[2] for u in uops if u.op is UOps.SPECIAL and u.arg[1][0] == "l")
    # https://clang.llvm.org/docs/AttributeReference.html#amdgpu-flat-work-group-size
    # NOTE: this makes hlb_cifar10 twice as fast, there may be more gains in tweaking these parameters
    return f"__attribute__((amdgpu_flat_work_group_size(1, {requiredMaxThreadsPerBlock})))"

class NVRenderer(CUDARenderer): device = "NV"
class HIPRenderer(AMDRenderer): device = "HIP"
