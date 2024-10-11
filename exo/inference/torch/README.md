# PyTorch & HuggingFace inference engine

## Notes/Issues
### 10/10/2024
- To select a pytorch device via environment variables, set the variable TORCH_DEVICE
  - XLA is currently not installed and will need to be added to inference.py, looking into doing this on a TPU VM
  - With pytorch, CUDA and ROCm are the same so specifying CUDA also enables ROCm support. See this [post](https://github.com/pytorch/pytorch/issues/55223#issuecomment-812587373)
  - Looking into adding mobile device support properly
- If device is not CPU the data type defaults to float32 else float16.
