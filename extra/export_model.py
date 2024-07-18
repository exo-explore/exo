from typing import Tuple, Dict, List
from tinygrad.dtype import DType
from tinygrad.renderer import Program
from tinygrad.tensor import Device, Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.state import get_state_dict
from tinygrad.helpers import Context
from tinygrad.dtype import dtypes
import json

EXPORT_SUPPORTED_DEVICE = ["WEBGPU", "WEBGL", "CLANG", "CUDA", "GPU"]
web_utils = {
  "getTensorBuffer":
  """const getTensorBuffer = (safetensorBuffer, tensorMetadata) => {
    return safetensorBuffer.subarray(...tensorMetadata.data_offsets);
  }""",
  "getTensorMetadata": """const getTensorMetadata = (safetensorBuffer) => {
    const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
    const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
    return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}]));
  };"""
}

def compile_net(run:TinyJit, special_names:Dict[int,str]) -> Tuple[Dict[str,str],List[Tuple[str,List[str],List[int]]],Dict[str,Tuple[int,DType,int]],Dict[str,Tensor]]:
  functions, bufs, bufs_to_save, statements, bufnum = {}, {}, {}, [], 0
  for ji in run.jit_cache:
    fxn: Program = ji.prg.p
    functions[fxn.function_name] = fxn.src   # NOTE: this assumes all with the same name are the same
    cargs = []
    for i,arg in enumerate(ji.bufs):
      key = id(arg)
      if key not in bufs:
        if key in special_names:
          bufs[key] = (special_names[key], arg.size*arg.dtype.itemsize, arg.dtype, key)
        else:
          bufs[key] = (f"buf_{bufnum}", arg.size*arg.dtype.itemsize, arg.dtype, key)
          bufnum += 1
          if i > 0: bufs_to_save[bufs[key][0]] = arg   # if first usage of a buffer is not an output, and it's not a special name
      cargs.append(bufs[key][0])
    statements.append((fxn.function_name, cargs, fxn.global_size, fxn.local_size))

  return functions, statements, {name:(size, dtype, key) for (name,size,dtype,key) in bufs.values()}, bufs_to_save

def jit_model(model, *args) -> Tuple[TinyJit,Dict[int,str]]:
  assert hasattr(model, "forward") or callable(model), "model needs a forward function"
  @TinyJit
  def run(*x):
    out = model.forward(*x) if hasattr(model, "forward") else model(*x)
    assert isinstance(out, tuple) or isinstance(out, list) or isinstance(out, Tensor), "model output must be a Tensor, tuple, or a list of Tensors for export"
    out = [out] if isinstance(out, Tensor) else out
    return [o.realize() for o in out]

  # twice to run the JIT
  for _ in range(2): the_output = run(*args)
  special_names = {}

  # hack to put the inputs back
  for (j,i),idx in run.input_replace.items():
    realized_input = args[idx].lazydata.base.realized
    run.jit_cache[j].bufs[i] = realized_input
    special_names[id(realized_input)] = f'input{idx}'

  # TODO: fetch this from the jit in self.input_replace and self.ret (hint: use get_parameters on self.ret)
  for i, output in enumerate(the_output):
    special_names[id(output.lazydata.base.realized)] = f'output{i}'
  return run, special_names

def export_model_clang(functions:Dict[str,str], statements:Dict[str,Tuple[str,int,int]], bufs:Dict[str,Tuple[str,int,int]], bufs_to_save:Dict[str,Tensor], input_names:List[str], output_names:List[str]) -> str:
  cprog = ["#include <tgmath.h>"]

  for name,cl in bufs_to_save.items():
    weight = ''.join(["\\x%02X"%x for x in bytes(cl._buf)])
    cprog.append(f"unsigned char {name}_data[] = \"{weight}\";")

  inputs = ", ".join([f'float* {input}' for input in input_names])
  outputs = ", ".join([f'float* {output}' for output in output_names])
  cprog += [f"float {name}[{len}];" if name not in bufs_to_save else f"float *{name} = (float *){name}_data;" for name,(len,dtype,_key) in bufs.items() if name not in ['input', 'outputs']]
  cprog += list(functions.values())
  cprog += [f"void net({inputs}, {outputs}) {{"] + [f"{name}({', '.join(args)});" for (name, args, _global_size, _local_size) in statements] + ["}"]
  return '\n'.join(cprog)

def export_model_webgl(functions, statements, bufs, bufs_to_save, weight_names, input_names, output_names) -> str:
  header = f"""
  function setupNet(gl, safetensor) {{
    function createShaderProgram(gl, code) {{
      const vertexShader = loadShader(gl, gl.VERTEX_SHADER, '#version 300 es\\nin vec2 in_position;in vec2 in_uv;out vec2 uv;void main(){{gl_Position=vec4(in_position,0.0,1.0);uv=in_uv;}}');
      const fragmentShader = loadShader(gl, gl.FRAGMENT_SHADER, code);
      const shaderProgram = gl.createProgram();
      gl.attachShader(shaderProgram, vertexShader);
      gl.attachShader(shaderProgram, fragmentShader);
      gl.linkProgram(shaderProgram);

      if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {{
        console.log(`Unable to initialize the shader program: ${{gl.getProgramInfoLog(shaderProgram)}}`);
        return null;
      }}

      return shaderProgram;
    }}

    function loadShader(gl, type, source) {{
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);

      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {{
        console.log(`An error occurred compiling the shaders: ${{gl.getShaderInfoLog(shader)}}`);
        gl.deleteShader(shader);
        return null;
      }}

      return shader;
    }}

    function setupVertexData(gl, program, vertices) {{
      let vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      let vertexBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
      const positionLocation = gl.getAttribLocation(program, 'in_position');
      const uvLocation = gl.getAttribLocation(program, 'in_uv');
      gl.enableVertexAttribArray(positionLocation);
      gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 4 * 4, 0);
      gl.enableVertexAttribArray(uvLocation);
      gl.vertexAttribPointer(uvLocation, 2, gl.FLOAT, false, 4 * 4, 2 * 4);
      gl.bindVertexArray(null);

      return vao;
    }}

    function runProgram(gl, kernelName, program, textures) {{
      let framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, textures[0].tex, 0);
      gl.useProgram(program);
      gl.uniform1i(gl.getUniformLocation(program, "width"), textures[0].width);

      const vao = setupVertexData(gl, program, [-1, 1, 0, 1, -1, -1, 0, 0, 1, 1, 1, 1, 1, -1, 1, 0]);
      gl.bindVertexArray(vao);
      // Texture 0 is the framebuffer texture, so we skip that
      for (let i = 1; i < textures.length; i++) {{
        gl.activeTexture(gl.TEXTURE0 + i-1);
        gl.bindTexture(gl.TEXTURE_2D, textures[i].tex);
        gl.uniform1i(gl.getUniformLocation(program, 'data' + i), i-1);
      }}

      gl.viewport(0, 0, textures[0].width, textures[0].height);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      for (let i = 1; i < textures.length; i++) {{
        gl.activeTexture(gl.TEXTURE0 + i-1);
        gl.bindTexture(gl.TEXTURE_2D, null);
      }}

      console.log("Finished running: " + kernelName);
    }}

    function limitTextureDims(size, threshold) {{
      if (size <= threshold) {{ return [size, 1] }};

      for (let i = 2; i < threshold + 1; i++) {{
        if ((size % i == 0) && (Math.floor(size / i) <= threshold)) {{
          return [Math.floor(size / i), i];
        }}
      }}

      return [size, 1];
    }}

    function updateTextureData(gl, texture, data, isHalf) {{
      gl.bindTexture(gl.TEXTURE_2D, texture.tex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, texture.width, texture.height, gl.RED, (isHalf) ? gl.HALF_FLOAT : gl.FLOAT, data);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }}

    function readTextureData(gl, texture) {{
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture.tex, 0);

      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {{
        throw new Error('Framebuffer not complete');
      }}

      let data = new Float32Array(texture.width * texture.height);
      gl.readPixels(0, 0, texture.width, texture.height, gl.RED, gl.FLOAT, data);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.deleteFramebuffer(framebuffer);

      return data;
    }}

    function createTexture(gl, size, isHalf, tensorBuffer) {{
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      const internalFormat = gl.RGBA;
      const texSize = limitTextureDims(size, gl.getParameter(gl.MAX_TEXTURE_SIZE));
      let weights;

      if (tensorBuffer != null) {{
        if (!isHalf)
          weights = new Float32Array(tensorBuffer.buffer, tensorBuffer.byteOffset, tensorBuffer.byteLength / Float32Array.BYTES_PER_ELEMENT);
        else
          weights = new Uint16Array(tensorBuffer.buffer, tensorBuffer.byteOffset, tensorBuffer.byteLength / Uint16Array.BYTES_PER_ELEMENT);
      }} else {{
        if (!isHalf)
          weights = new Float32Array(size).fill(0.0);
        else
          weights = new Uint16Array(size).fill(0.0);
      }}

      if (size != weights.length)
        console.log("Weights length: " + weights.length + ", texsize: " + texSize[0]*texSize[1]);

      gl.texImage2D(gl.TEXTURE_2D, 0, (isHalf) ? gl.R16F : gl.R32F, texSize[0], texSize[1], 0, gl.RED, (isHalf) ? gl.HALF_FLOAT : gl.FLOAT, weights);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.bindTexture(gl.TEXTURE_2D, null);
      return {{ tex: texture, width: texSize[0], height: texSize[1] }};
    }}

    {web_utils["getTensorBuffer"]}
    {web_utils["getTensorMetadata"]}

    const metadata = getTensorMetadata(safetensor);
  """

  textures = '\n    '.join([f"const {name} = " + (f"createTexture(gl, {size/(2 if dtype == dtypes.half else 4)}, {'true' if dtype == dtypes.half else 'false'});" if _key not in weight_names else f"createTexture(gl, {size/(2 if dtype == dtypes.half else 4)}, {'true' if dtype == dtypes.half else 'false'}, getTensorBuffer(safetensor, metadata['{weight_names[_key]}']))") + ";"  for name,(size,dtype,_key) in bufs.items()])
  kernels = '\n\n'.join([f"const {key} = `{code.replace(key, 'main').replace('version 330', 'version 300 es')}`;" for key, code in functions.items()])
  kernel_names = ', '.join([name for (name, _args, _global_size, _local_size) in statements])
  kernel_calls = '\n        '.join([f"runProgram(gl, '{name}', programs[{i}], [{', '.join(args)}]);" for i, (name, args, _global_size, _local_size) in enumerate(statements) ])
  copy_inputs = "\n".join([f'updateTextureData(gl, {name}, _{name}, {"true" if dtype == dtypes.half else "false"});' for name,(size,dtype,_key) in bufs.items() if "input" in name])
  entry_point = f"""
    return function({",".join([f"_{name}" for name,(size,dtype,_key) in bufs.items() if "input" in name])}) {{
      const ext = gl.getExtension('EXT_color_buffer_float');
      {copy_inputs}
      {kernel_calls}

      return readTextureData(gl, output0);
    }}
  """
  programs = f"let programs = [{kernel_names}].map((code) => createShaderProgram(gl, code));"
  return f"{header}\n{kernels}\n{textures}\n{programs}\n{entry_point}}}"

def export_model_webgpu(functions, statements, bufs, bufs_to_save, weight_names, input_names, output_names) -> Tuple[str,int,int]:
  kernel_code = '\n\n'.join([f"const {key} = `{code.replace(key, 'main')}`;" for key, code in functions.items()])
  kernel_names = ', '.join([name for (name, _args, _global_size, _local_size) in statements])
  kernel_calls = '\n        '.join([f"addComputePass(device, commandEncoder, piplines[{i}], [{', '.join(args)}], {global_size});" for i, (_name, args, global_size, _local_size) in enumerate(statements) ])
  _bufs =  '\n    '.join([f"const {name} = " + (f"createEmptyBuf(device, {size});" if _key not in weight_names else f"createWeightBuf(device, {size}, getTensorBuffer(safetensor, metadata['{weight_names[_key]}']))") + ";"  for name,(size,dtype,_key) in bufs.items()])
  gpu_write_bufs =  '\n    '.join([f"const gpuWriteBuffer{i} = device.createBuffer({{size:{input_name}.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE }});" for i,input_name in enumerate(input_names)])
  input_writers = '\n    '.join([f"await gpuWriteBuffer{i}.mapAsync(GPUMapMode.WRITE);\n        new Float32Array(gpuWriteBuffer{i}.getMappedRange()).set(" + f'_{inp_name});' + f"\n        gpuWriteBuffer{i}.unmap();\n        commandEncoder.copyBufferToBuffer(gpuWriteBuffer{i}, 0, {inp_name}, 0, gpuWriteBuffer{i}.size);"  for i,inp_name in enumerate(input_names)])
  gpu_read_bufs = '\n    '.join([f"const gpuReadBuffer{i} = device.createBuffer({{size:{output_name}.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }});" for i,output_name in enumerate(output_names)])
  outbuf_copies = '\n        '.join([f"commandEncoder.copyBufferToBuffer({output_name}, 0, gpuReadBuffer{i}, 0, output{i}.size);" for i,output_name in enumerate(output_names)])
  output_readers = '\n        '.join([f"await gpuReadBuffer{i}.mapAsync(GPUMapMode.READ);\n        const resultBuffer{i} = new Float32Array(gpuReadBuffer{i}.size);\n        resultBuffer{i}.set(new Float32Array(gpuReadBuffer{i}.getMappedRange()));\n        gpuReadBuffer{i}.unmap();" for i in range(len(output_names))])
  output_return = '[{}]'.format(",".join([f'resultBuffer{i}' for i in range(len(output_names))]))
  return f"""
{web_utils["getTensorBuffer"]}

{web_utils["getTensorMetadata"]}

const createEmptyBuf = (device, size) => {{
    return device.createBuffer({{size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }});
}};

const createWeightBuf = (device, size, data) => {{
  const buf = device.createBuffer({{ mappedAtCreation: true, size, usage: GPUBufferUsage.STORAGE }});
  new Uint8Array(buf.getMappedRange()).set(data);
  buf.unmap();
  return buf;
}};

const addComputePass = (device, commandEncoder, pipeline, bufs, workgroup) => {{
  const bindGroup = device.createBindGroup({{layout: pipeline.getBindGroupLayout(0), entries: bufs.map((buffer, index) => ({{ binding: index, resource: {{ buffer }} }}))}});
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroup);
  passEncoder.end();
}};

{kernel_code}

const setupNet = async (device, safetensor) => {{
    const metadata = getTensorMetadata(safetensor);

    {_bufs}

    {gpu_write_bufs}

    {gpu_read_bufs}

    const kernels = [{kernel_names}];
    const piplines = await Promise.all(kernels.map(name => device.createComputePipelineAsync({{layout: "auto", compute: {{ module: device.createShaderModule({{ code: name }}), entryPoint: "main" }}}})));

    return async ({",".join([f"_{input_name}" for input_name in input_names])}) => {{
        const commandEncoder = device.createCommandEncoder();
        {input_writers}
        {kernel_calls}
        {outbuf_copies}
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        {output_readers}
        return {output_return};
    }}
}}
  """ + f"\n\nconst loadNet = async (device) => {{ return await fetch('net.safetensors').then(x => x.arrayBuffer()).then(x => setupNet(device, new Uint8Array(x))); }}"

def export_model(model, target:str, *inputs):
  assert Device.DEFAULT in EXPORT_SUPPORTED_DEVICE, "only WEBGPU, WEBGL, CLANG, CUDA, GPU, METAL are supported"
  with Context(JIT=2): run,special_names = jit_model(model, *inputs)
  functions, statements, bufs, bufs_to_save = compile_net(run, special_names)
  state = get_state_dict(model)
  weight_names = {id(x.lazydata.base.realized): name for name, x in state.items()}
  input_names = [name for _,name in special_names.items() if "input" in name]
  output_names = [name for _,name in special_names.items() if "output" in name]
  prg = ""
  if target == "clang":
    prg = export_model_clang(functions, statements, bufs, bufs_to_save, input_names, output_names)
  elif target == "webgpu":
    prg = export_model_webgpu(functions, statements, bufs, bufs_to_save, weight_names, input_names, output_names)
  elif target == "webgl":
    prg = export_model_webgl(functions, statements, bufs, bufs_to_save, weight_names, input_names, output_names)
  else:
    prg = json.dumps({
      "backend": Device.DEFAULT,
      "inputs": [{
        "size": bufs[name][0],
        "dtype": bufs[name][1].name
      } for name in input_names],
      "outputs": [{
        "size": bufs[name][0],
        "dtype": bufs[name][1].name
      } for name in output_names],
      "functions": functions,
      "statements": [{
        "kernel": kernel,
        "args": args,
        "global_size": global_size,
        "local_size": local_size
      } for (kernel, args, global_size, local_size) in statements],
      "buffers": {
        name: {
          "size": size,
          "dtype": dtype.name,
          "id": weight_names[_key] if _key in weight_names else ""
        } for name, (size,dtype,_key) in bufs.items() if name not in ["input", "outputs"]
      }
    })

  return prg, {input:bufs[input][0] for input in input_names}, {output:bufs[output][0] for output in output_names}, state
