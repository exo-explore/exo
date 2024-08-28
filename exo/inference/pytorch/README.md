# PyTorch & HuggingFace inference engine
Experimental, still under development


## Install
Install needed py modules, make sure to be using CUDA 12.4 for the PyTorch install

```console
$ pip install torch --index-url https://download.pytorch.org/whl/cu124
$ pip install transformers accelerate
```

After installing accelerate you get hit with a dependency error, for now ignore until we can fix this as exo works fine with 1.26.4

```console
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
exo 0.0.1 requires numpy==2.0.0, but you have numpy 1.26.4 which is incompatible.
```

## Low VRAM Notes

- When trying to do disk_offload getting the error "Cannot copy out of meta tensor; no data!", looking up the error it is tied to (low vram)[https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13087#issuecomment-2080272004]

## Multiple GPU in 1 Notes
### Running multiple GPUs on 1 machine
- Getting error "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1! (when checking argument for argument tensors in method wrapper_CUDA_cat)"
