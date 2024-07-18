import gzip
from tinygrad.tensor import Tensor
from tinygrad.helpers import fetch

def _fetch_mnist(file, offset): return Tensor(gzip.open(fetch("https://storage.googleapis.com/cvdf-datasets/mnist/"+file)).read()[offset:])
def mnist():
  return _fetch_mnist("train-images-idx3-ubyte.gz", 0x10).reshape(-1, 1, 28, 28), _fetch_mnist("train-labels-idx1-ubyte.gz", 8), \
         _fetch_mnist("t10k-images-idx3-ubyte.gz", 0x10).reshape(-1, 1, 28, 28), _fetch_mnist("t10k-labels-idx1-ubyte.gz", 8)
