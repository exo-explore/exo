from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.dtype import _from_torch_dtype, _from_np_dtype
import torch

def main() -> None:
    tensor1 = torch.tensor([1.0, 2.0, 3.0], device=torch.device("mps"))
    tiny_tensor1 = Tensor.from_blob(tensor1.data_ptr(), tensor1.shape, dtype=_from_torch_dtype(tensor1.dtype), device='METAL')
    
    # Before tinygrad calculations, mps needs to be synchronized to make sure data is valid.
    if tensor1.device.type == "mps": torch.mps.synchronize()
    else: torch.cuda.synchronize()
    
    x = (tiny_tensor1 + 1).realize()
    
    print(x)

if __name__=="__main__":
    main()