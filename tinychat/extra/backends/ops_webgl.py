import numpy as np
import functools
from tinygrad.dtype import dtypes, ImageDType
from tinygrad.device import Compiled, Allocator, CompilerOptions
from tinygrad.codegen.kernel import OptOps
from tinygrad.renderer.cstyle import uops_to_cstyle
from tinygrad.renderer.cstyle import GLSLLanguage
import moderngl

ctx = moderngl.create_standalone_context()
max_dims = 4096
dtype_map = { dtypes.float64: "f8", dtypes.float: "f4", dtypes.half: "f2", dtypes.int32: "i4", dtypes.uint32: "u4", dtypes.bool: "i1"}
vertex_shader="#version 330\nprecision highp float;\nin vec2 in_position;in vec2 in_uv;out vec2 uv;void main(){\
gl_Position=vec4(in_position,0.0,1.0);uv=in_uv;}"
class WebGLProgram:
  def __init__(self, name: str, prg: str, bufs:int=0, vars:int=0):
    self.name, self.prg = name, ctx.program(vertex_shader=vertex_shader, fragment_shader=prg)
  def __call__(self, *bufs, global_size, local_size=None, vals=(), wait=False):
    vert = ctx.buffer(np.asarray([-1, 1, -1, -1, 1, 1, 1, -1], dtype='f4').tobytes())
    uv = ctx.buffer(np.asarray([0, 1, 0, 0, 1, 1, 1, 0], dtype='f4').tobytes())
    self.vao = ctx.vertex_array(self.prg, [])
    self.vao.bind(self.prg["in_position"].location if "in_position" in self.prg else 0, buffer=vert, cls='f', fmt='2f4')
    self.vao.bind(self.prg["in_uv"].location if "in_uv" in self.prg else 1, buffer=uv, cls='f', fmt='2f4')
    self.vao.vertices = vert.size//4//2
    self.fbo = ctx.framebuffer(color_attachments=[bufs[0]])

    for i, x in enumerate(bufs[1:], start=1):
      if f"data{i}" in self.prg:
        self.prg[f"data{i}"] = i
        x.use(i)

    if ("width" in self.prg): self.prg["width"].value = self.fbo.size[0]
    ctx.viewport = (0, 0, self.fbo.size[0], self.fbo.size[1])
    self.fbo.use()
    self.vao.render(mode=moderngl.TRIANGLE_STRIP)

class RawWebGLAllocator(Allocator):
  def _alloc_image(self, dtype:ImageDType):
    tex = ctx.texture(dtype.shape, 1, dtype=dtype_map[dtype.base])
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    return tex
  def copyin(self, dest:moderngl.Texture, src: memoryview): dest.write(src)
  def copyout(self, dest:memoryview, src: moderngl.Texture):
    src.read_into(dest)
    return dest

class WebGlDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(RawWebGLAllocator(),
      CompilerOptions(device="WEBGL", global_max=[4096*4096,1,1], unsupported_opts=[OptOps.UPCAST, OptOps.UPCASTMID],
      supports_float4=False, supports_float4_alu=False, has_local=False, has_shared=False, dont_use_locals=True),
      functools.partial(uops_to_cstyle, GLSLLanguage()), lambda x: x, WebGLProgram)
