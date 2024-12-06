#from https://github.com/ml-explore/mlx-examples
from pathlib import Path
import numpy as np
import json

def make_batch(tokens):
  lengths = [len(x) for x in tokens]
  batch_size = len(lengths)

  # Check if any sequence is longer than 2048 tokens
  if max(lengths) > 2048:
    print("You've got sequences with over 2048 tokens in here! Split your data fool!")

  # Pad to the max length
  batch_arr = np.zeros((batch_size, max(lengths)), np.int32)

  for j in range(batch_size):
    batch_arr[j, : lengths[j]] = tokens[j]
  batch = np.array(batch_arr)
  return batch[:, :-1], batch[:, 1:], np.array(lengths)

def iterate_batches(dset, tokenizer, batch_size, train=False):
# Shuffle indices
  while True:
    indices = np.arange(len(dset))
    if train:
      indices = np.random.permutation(indices)

    # Collect batches from dataset
    for i in range(0, len(indices) - batch_size + 1, batch_size):
      # Encode batch
      yield make_batch([tokenizer.encode(dset[indices[i + j]]) for j in range(batch_size)])

    if not train:
      break

class Dataset:
  """
  Light-weight wrapper to hold lines from a jsonl file
  """

  def __init__(self, path: Path, key: str = "text"):
    if not path.exists():
      self._data = None
    else:
      with open(path, "r") as fid:
        self._data = [json.loads(l) for l in fid]
    self._key = key

  def __getitem__(self, idx: int):
    return self._data[idx][self._key]

  def __len__(self):
    return len(self._data)


def load_dataset(data_path: str):
  def load_and_check(name):
    dataset_path = Path(data_path) / f"{name}.jsonl"
    try:
      return Dataset(dataset_path)
    except Exception as e:
      print(f"Unable to build dataset {dataset_path} ({e})")
      raise

  names = ("train", "valid", "test")
  train, valid, test = (load_and_check(n) for n in names)

  return train, valid, test

