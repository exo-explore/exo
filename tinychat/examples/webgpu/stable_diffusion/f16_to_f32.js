const f16tof32 = `
fn u16_to_f16(x: u32) -> f32 {
    let sign = f32((x >> 15) & 0x1);
    let exponent = f32((x >> 10) & 0x1F);
    let fraction = f32(x & 0x3FF);

    let sign_multiplier = select(1.0, -1.0, sign == 1.0);
    if (exponent == 0.0) {
        return sign_multiplier * 6.103515625e-5 * (fraction / 1024.0);
    } else {
        return sign_multiplier * exp2(exponent - 15.0) * (1.0 + fraction / 1024.0);
    }
}

@group(0) @binding(0) var<storage,read_write> data0: array<u32>;
@group(0) @binding(1) var<storage,read_write> data1: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gidx = gid.x;
    let outgidx = gidx*2;

    if (gidx >= arrayLength(&data0)) {
        return;
    }

    let oo = data0[gidx];
    let oo1 = (oo >> 16);
    let oo2 = oo & 0xFFFFu;

    let f1 = u16_to_f16(oo2);
    let f2 = u16_to_f16(oo1);
    
    data1[outgidx] = f1;
    data1[outgidx + 1] = f2;
}`;

window.f16tof32GPU = async(device, inf16) => {
    const input = device.createBuffer({size: inf16.length, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({size: inf16.length*2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    const gpuWriteBuffer = device.createBuffer({size: input.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });
    const gpuReadBuffer = device.createBuffer({ size: output.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const commandEncoder = device.createCommandEncoder();
    await gpuWriteBuffer.mapAsync(GPUMapMode.WRITE);

    const alignedUint32View = new Uint32Array(inf16.buffer, inf16.byteOffset, inf16.length / 4);
    new Uint32Array(gpuWriteBuffer.getMappedRange()).set(alignedUint32View);

    gpuWriteBuffer.unmap();
    commandEncoder.copyBufferToBuffer(gpuWriteBuffer, 0, input, 0, gpuWriteBuffer.size);
    const pipeline = await device.createComputePipelineAsync({layout: "auto", compute: { module: device.createShaderModule({ code: f16tof32 }), entryPoint: "main" }});

    addComputePass(device, commandEncoder, pipeline, [input, output], [Math.ceil(inf16.length/(4*256)), 1, 1]);

    commandEncoder.copyBufferToBuffer(output, 0, gpuReadBuffer, 0, output.size);
    const gpuCommands = commandEncoder.finish();
    device.queue.submit([gpuCommands]);

    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    const resultBuffer = new Float32Array(gpuReadBuffer.size/4);
    resultBuffer.set(new Float32Array(gpuReadBuffer.getMappedRange()));
    gpuReadBuffer.unmap();

    return resultBuffer;
}
