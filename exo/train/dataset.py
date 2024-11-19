#from https://github.com/ml-explore/mlx-examples
from pathlib import Path
import numpy as np
import json
from functools import partial, reduce
def compose(*funcs):    
  return reduce(lambda f, g: lambda x: f(g(x)), funcs, lambda x : x)

def batch_with_lengths(tokens, maxlen = None):
  lengths = [len(x) for x in tokens]
  batch_size = len(lengths)
  if maxlen is None:
    maxlen = max(lengths)
  else:
    lengths = [min(maxlen, l) for l in lengths]

  # Pad to the max length
  batch_arr = np.zeros((batch_size, maxlen), np.int32)

  for j in range(batch_size):
    batch_arr[j, : lengths[j]] = tokens[j]
  batch = np.array(batch_arr)
  return batch[:, :-1], batch[:, 1:], np.array(lengths)

def batch_chunk(batch_size):
  return lambda d, i: d[i:i + batch_size]
  

def iterate_batches(dset, batch_size, train=False, uniform_length=True):
# Shuffle indices
  make_batch = lambda b: batch_with_lengths(b, maxlen=dset._maxlen if uniform_length else None)
  chunk = batch_chunk(batch_size)
  while True:
    indices = np.arange(len(dset))
    if train:
      indices = np.random.permutation(indices)
    batch = compose(make_batch, lambda i: [dset[k] for k in i], partial(chunk, indices))

    # Collect batches from dataset
    for i in range(0, len(indices) - batch_size + 1, batch_size):
      yield batch(i)

    if not train:
      break

class Dataset:
  def __init__(self, path: Path, preprocess=lambda item: item, loadline=json.loads, metrics={}):
    if not path.exists():
      self._data = None
    else:
      self.preprocess = preprocess
      with open(path, "r") as fid:
        self._data = [loadline(l) for l in fid]
        self._maxlen = max([len(preprocess(x)) for x in self._data])
        # Check if any sequence is longer than 2048 tokens
        if self._maxlen > 2048:
          print("You've got sequences with over 2048 tokens in here! Split your data fool!")


  def __getitem__(self, idx: int):
    return self.preprocess(self._data[idx])

  def __len__(self):
    return len(self._data)


def load_dataset(data_path: str, preprocess=lambda i: i, loadline=json.loads):
  def load_and_check(name):
    dataset_path = Path(data_path) / f"{name}.jsonl"
    try:
      return Dataset(dataset_path, preprocess=preprocess, loadline=loadline)
    except Exception as e:
      print(f"Unable to build dataset {dataset_path} ({e})")
      raise

  names = ("train", "valid", "test")
  train, valid, test = (load_and_check(n) for n in names)

  return train, valid, test

