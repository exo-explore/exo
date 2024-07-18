import os
from extra.export_model import compile_net, jit_model
from examples.stable_diffusion import StableDiffusion
from tinygrad.nn.state import get_state_dict, safe_save, safe_load_metadata, torch_load, load_state_dict
from tinygrad.tensor import Tensor
from tinygrad import Device
from tinygrad.helpers import fetch
from typing import NamedTuple, Any, List
from pathlib import Path
import argparse
import numpy as np

def convert_f32_to_f16(input_file, output_file):
  with open(input_file, 'rb') as f:
    metadata_length_bytes = f.read(8)
    metadata_length = int.from_bytes(metadata_length_bytes, byteorder='little', signed=False)
    metadata_json_bytes = f.read(metadata_length)
    float32_values = np.fromfile(f, dtype=np.float32)

  first_text_model_offset = 3772703308
  num_elements = int((first_text_model_offset)/4)
  front_float16_values = float32_values[:num_elements].astype(np.float16)
  rest_float32_values = float32_values[num_elements:]

  with open(output_file, 'wb') as f:
    f.write(metadata_length_bytes)
    f.write(metadata_json_bytes)
    front_float16_values.tofile(f)
    rest_float32_values.tofile(f)

def split_safetensor(fn):
  _, json_len, metadata = safe_load_metadata(fn)
  text_model_offset = 3772703308
  chunk_size = 536870912

  for k in metadata:
    # safetensor is in fp16, except for text moel
    if (metadata[k]["data_offsets"][0] < text_model_offset):
      metadata[k]["data_offsets"][0] = int(metadata[k]["data_offsets"][0]/2)
      metadata[k]["data_offsets"][1] = int(metadata[k]["data_offsets"][1]/2)

  last_offset = 0
  part_end_offsets = []

  for k in metadata:
    offset = metadata[k]['data_offsets'][0]

    if offset == text_model_offset:
      break

    part_offset = offset - last_offset

    if (part_offset >= chunk_size):
      part_end_offsets.append(8+json_len+offset)
      last_offset = offset

  text_model_start = int(text_model_offset/2)
  net_bytes = bytes(open(fn, 'rb').read())
  part_end_offsets.append(text_model_start+8+json_len)
  cur_pos = 0

  for i, end_pos in enumerate(part_end_offsets):
    with open(f'./net_part{i}.safetensors', "wb+") as f:
      f.write(net_bytes[cur_pos:end_pos])
      cur_pos = end_pos

  with open(f'./net_textmodel.safetensors', "wb+") as f:
    f.write(net_bytes[text_model_start+8+json_len:])

  return part_end_offsets

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run Stable Diffusion', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--remoteweights', action='store_true', help="Use safetensors from Huggingface, or from local")
  args = parser.parse_args()
  Device.DEFAULT = "WEBGPU"

  Tensor.no_grad = True
  model = StableDiffusion()

  # load in weights
  load_state_dict(model, torch_load(fetch('https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt', 'sd-v1-4.ckpt'))['state_dict'], strict=False)

  class Step(NamedTuple):
    name: str = ""
    input: List[Tensor] = []
    forward: Any = None

  sub_steps = [
    Step(name = "textModel", input = [Tensor.randn(1, 77)], forward = model.cond_stage_model.transformer.text_model),
    Step(name = "diffusor", input = [Tensor.randn(1, 77, 768), Tensor.randn(1, 77, 768), Tensor.randn(1,4,64,64), Tensor.rand(1), Tensor.randn(1), Tensor.randn(1), Tensor.randn(1)], forward = model),
    Step(name = "decoder", input = [Tensor.randn(1,4,64,64)], forward = model.decode)
  ]

  prg = ""

  def compile_step(model, step: Step):
    run, special_names = jit_model(step, *step.input)
    functions, statements, bufs, _ = compile_net(run, special_names)
    state = get_state_dict(model)
    weights = {id(x.lazydata.base.realized): name for name, x in state.items()}
    kernel_code = '\n\n'.join([f"const {key} = `{code.replace(key, 'main')}`;" for key, code in functions.items()])
    kernel_names = ', '.join([name for (name, _, _, _) in statements])
    kernel_calls = '\n        '.join([f"addComputePass(device, commandEncoder, piplines[{i}], [{', '.join(args)}], {global_size});" for i, (_name, args, global_size, _local_size) in enumerate(statements) ])
    bufs =  '\n    '.join([f"const {name} = " + (f"createEmptyBuf(device, {size});" if _key not in weights else f"createWeightBuf(device, {size}, getTensorBuffer(safetensor, metadata['{weights[_key]}'], '{weights[_key]}'))") + ";"  for name,(size,dtype,_key) in bufs.items()])
    gpu_write_bufs =  '\n    '.join([f"const gpuWriteBuffer{i} = device.createBuffer({{size:input{i}.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE }});" for i,(_,value) in enumerate(special_names.items()) if "output" not in value])
    input_writer = '\n    '.join([f"await gpuWriteBuffer{i}.mapAsync(GPUMapMode.WRITE);\n    new Float32Array(gpuWriteBuffer{i}.getMappedRange()).set(" + f'data{i});' + f"\n    gpuWriteBuffer{i}.unmap();\ncommandEncoder.copyBufferToBuffer(gpuWriteBuffer{i}, 0, input{i}, 0, gpuWriteBuffer{i}.size);"  for i,(_,value) in enumerate(special_names.items()) if value != "output0"])
    return f"""\n    var {step.name} = function() {{

    {kernel_code}

    return {{
      "setup": async (device, safetensor) => {{
        const metadata = getTensorMetadata(safetensor[0]);

        {bufs}

        {gpu_write_bufs}
        const gpuReadBuffer = device.createBuffer({{ size: output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }});

        const kernels = [{kernel_names}];
        const piplines = await Promise.all(kernels.map(name => device.createComputePipelineAsync({{layout: "auto", compute: {{ module: device.createShaderModule({{ code: name }}), entryPoint: "main" }}}})));

        return async ({",".join([f'data{i}' for i,(k,v) in enumerate(special_names.items()) if v != "output0"])}) => {{
            const commandEncoder = device.createCommandEncoder();

            {input_writer}

            {kernel_calls}
            commandEncoder.copyBufferToBuffer(output0, 0, gpuReadBuffer, 0, output0.size);
            const gpuCommands = commandEncoder.finish();
            device.queue.submit([gpuCommands]);

            await gpuReadBuffer.mapAsync(GPUMapMode.READ);
            const resultBuffer = new Float32Array(gpuReadBuffer.size/4);
            resultBuffer.set(new Float32Array(gpuReadBuffer.getMappedRange()));
            gpuReadBuffer.unmap();
            return resultBuffer;
        }}
      }}
    }}
  }}
  """

  for step in sub_steps:
    print(f'Executing step={step.name}')
    prg += compile_step(model, step)

    if step.name == "diffusor":
      if args.remoteweights:
        base_url = "https://huggingface.co/wpmed/tinygrad-sd-f16/resolve/main"
      else:
        state = get_state_dict(model)
        safe_save(state, os.path.join(os.path.dirname(__file__), "net.safetensors"))
        convert_f32_to_f16("./net.safetensors", "./net_conv.safetensors")
        split_safetensor("./net_conv.safetensors")
        os.remove("net.safetensors")
        os.remove("net_conv.safetensors")
        base_url = "."

  prekernel = f"""
    window.MODEL_BASE_URL= "{base_url}";
    const getTensorMetadata = (safetensorBuffer) => {{
      const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
      const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
      return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {{...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}}]));
    }};

  const getTensorBuffer = (safetensorParts, tensorMetadata, key) => {{
    let selectedPart = 0;
    let counter = 0;
    let partStartOffsets = [1131408336, 2227518416, 3308987856, 4265298864];
    let correctedOffsets = tensorMetadata.data_offsets;
    let prev_offset = 0;

    for (let start of partStartOffsets) {{
      prev_offset = (counter == 0) ? 0 : partStartOffsets[counter-1];

      if (tensorMetadata.data_offsets[0] < start) {{
        selectedPart = counter;
        correctedOffsets = [correctedOffsets[0]-prev_offset, correctedOffsets[1]-prev_offset];
        break;
      }}

      counter++;
    }}

    let allZero = true;
    let out = safetensorParts[selectedPart].subarray(...correctedOffsets);

    for (let i = 0; i < out.length; i++) {{
        if (out[i] !== 0) {{
            allZero = false;
            break;
        }}
    }}

    if (allZero) {{
        console.log("Error: weight '" + key + "' is all zero.");
    }}

    return safetensorParts[selectedPart].subarray(...correctedOffsets);
  }}

  const getWeight = (safetensors, key) => {{
    let uint8Data = getTensorBuffer(safetensors, getTensorMetadata(safetensors[0])[key], key);
    return new Float32Array(uint8Data.buffer, uint8Data.byteOffset, uint8Data.byteLength / Float32Array.BYTES_PER_ELEMENT);
  }}

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
  }};"""

  with open(os.path.join(os.path.dirname(__file__), "net.js"), "w") as text_file:
    text_file.write(prekernel + prg)
