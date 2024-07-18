import os, json, pathlib, zipfile, pickle, tarfile, struct
from typing import Dict, Union, List, Optional, Any, Tuple
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import prod, argsort, DEBUG, Timing, CI, unwrap, GlobalCounters, tqdm
from tinygrad.shape.view import strides_for_shape
from tinygrad.multi import MultiLazyBuffer

safe_dtypes = {"BOOL":dtypes.bool, "I8":dtypes.int8, "U8":dtypes.uint8, "I16":dtypes.int16, "U16":dtypes.uint16, "I32":dtypes.int, "U32":dtypes.uint,
               "I64":dtypes.int64, "U64":dtypes.uint64, "F16":dtypes.float16, "BF16":dtypes.bfloat16, "F32":dtypes.float32, "F64":dtypes.float64}
inverse_safe_dtypes = {v:k for k,v in safe_dtypes.items()}

def safe_load_metadata(fn:Union[Tensor,str]) -> Tuple[Tensor, int, Any]:
  """
  Loads a .safetensor file from disk, returning the data, metadata length, and metadata.
  """
  t = fn if isinstance(fn, Tensor) else Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}")
  json_len = t[0:8].bitcast(dtypes.int64).item()
  return t, json_len, json.loads(t[8:8+json_len].numpy().tobytes())

def safe_load(fn:Union[Tensor,str]) -> Dict[str, Tensor]:
  """
  Loads a .safetensor file from disk, returning the state_dict.

  ```python
  state_dict = nn.state.safe_load("test.safetensor")
  ```
  """
  t, json_len, metadata = safe_load_metadata(fn)
  ret = {}
  for k,v in metadata.items():
    if k == "__metadata__": continue
    dtype = safe_dtypes[v['dtype']]
    sz = (v['data_offsets'][1]-v['data_offsets'][0])
    ret[k] = t[8+json_len+v['data_offsets'][0]:8+json_len+v['data_offsets'][0]+sz].bitcast(dtype).reshape(v['shape'])
  return ret

def safe_save(tensors:Dict[str, Tensor], fn:str, metadata:Optional[Dict[str, Any]]=None):
  """
  Saves a state_dict to disk in a .safetensor file with optional metadata.

  ```python
  t = Tensor([1, 2, 3])
  nn.state.safe_save({'t':t}, "test.safetensor")
  ```
  """
  headers, offset = {}, 0
  if metadata: headers['__metadata__'] = metadata
  for k,v in tensors.items():
    headers[k] = {'dtype': inverse_safe_dtypes[v.dtype], 'shape': list(v.shape), 'data_offsets':[offset, offset+v.nbytes()]}
    offset += v.nbytes()
  j = json.dumps(headers, separators=(',', ':'))
  j += "\x20"*((8-len(j)%8)%8)
  pathlib.Path(fn).unlink(missing_ok=True)
  t = Tensor.empty(8+len(j)+offset, dtype=dtypes.uint8, device=f"disk:{fn}")
  t[0:8].bitcast(dtypes.int64).assign([len(j)])
  t[8:8+len(j)].assign(list(j.encode('utf-8')))
  for k,v in safe_load(t).items(): v.assign(tensors[k])

# state dict

from collections import OrderedDict
def get_state_dict(obj, prefix:str='', tensor_type=Tensor) -> Dict[str, Tensor]:
  """
  Returns a state_dict of the object, with optional prefix.

  ```python exec="true" source="above" session="tensor" result="python"
  class Net:
    def __init__(self):
      self.l1 = nn.Linear(4, 5)
      self.l2 = nn.Linear(5, 6)

  net = Net()
  print(nn.state.get_state_dict(net).keys())
  ```
  """
  if isinstance(obj, tensor_type): return {prefix.strip('.'):obj}
  if hasattr(obj, '_asdict'): return get_state_dict(obj._asdict(), prefix, tensor_type)  # namedtuple
  if isinstance(obj, OrderedDict): return get_state_dict(dict(obj), prefix, tensor_type)
  if hasattr(obj, '__dict__'): return get_state_dict(obj.__dict__, prefix, tensor_type)
  state_dict = {}
  if isinstance(obj, (list, tuple)):
    for i,x in enumerate(obj): state_dict.update(get_state_dict(x, f"{prefix}{str(i)}.", tensor_type))
  elif isinstance(obj, dict):
    for k,v in obj.items(): state_dict.update(get_state_dict(v, f"{prefix}{str(k)}.", tensor_type))
  return state_dict
def get_parameters(obj) -> List[Tensor]:
  """
  ```python exec="true" source="above" session="tensor" result="python"
  class Net:
    def __init__(self):
      self.l1 = nn.Linear(4, 5)
      self.l2 = nn.Linear(5, 6)

  net = Net()
  print(len(nn.state.get_parameters(net)))
  ```
  """
  return list(get_state_dict(obj).values())

def load_state_dict(model, state_dict:Dict[str, Tensor], strict=True, verbose=True, consume=False) -> None:
  """
  Loads a state_dict into a model.

  ```python
  class Net:
    def __init__(self):
      self.l1 = nn.Linear(4, 5)
      self.l2 = nn.Linear(5, 6)

  net = Net()
  state_dict = nn.state.get_state_dict(net)
  nn.state.load_state_dict(net, state_dict)
  ```
  """
  start_mem_used = GlobalCounters.mem_used
  with Timing("loaded weights in ", lambda et_ns: f", {(GlobalCounters.mem_used-start_mem_used)/1e9:.2f} GB loaded at {(GlobalCounters.mem_used-start_mem_used)/et_ns:.2f} GB/s"):  # noqa: E501
    model_state_dict = get_state_dict(model)
    if DEBUG >= 1 and len(state_dict) > len(model_state_dict):
      print("WARNING: unused weights in state_dict", sorted(list(state_dict.keys() - model_state_dict.keys())))
    for k,v in (t := tqdm(model_state_dict.items(), disable=CI or not verbose)):
      t.desc = f"ram used: {GlobalCounters.mem_used/1e9:5.2f} GB, {k:50s}: "
      if k not in state_dict and not strict:
        if DEBUG >= 1: print(f"WARNING: not loading {k}")
        continue
      if isinstance((mlb:=v.lazydata), MultiLazyBuffer):
        if isinstance(state_dict[k].lazydata, MultiLazyBuffer): v.replace(state_dict[k]).realize()
        else: v.replace(state_dict[k].shard(mlb.device, mlb.axis)).realize()
      else: v.replace(state_dict[k].to(v.device)).realize()
      if consume: del state_dict[k]

# torch support!

def torch_load(fn:str) -> Dict[str, Tensor]:
  """
  Loads a torch .pth file from disk.

  ```python
  state_dict = nn.state.torch_load("test.pth")
  ```
  """
  t = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}")

  offsets: Dict[Union[str, int], int] = {}
  lens: Dict[Union[str, int], int] = {}
  def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=None, backward_hooks=None, metadata=None):
    #print(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata)
    lens[storage[2]] = storage[4] * storage[1].itemsize
    if storage[2] not in offsets: return None
    byte_offset = offsets[storage[2]]+storage_offset*storage[1].itemsize
    ret = t[byte_offset:byte_offset+prod(size)*storage[1].itemsize].bitcast(storage[1])

    # 7 lines to deal with permuted tensors. NOTE: this currently requires reading off the disk
    shape_strides = [(s, st) for s,st in zip(size, stride) if s != 1]
    permute_indexes = [len(shape_strides)-1-y for y in argsort([x[1] for x in shape_strides])]
    if tuple(permute_indexes) != tuple(range(len(permute_indexes))):
      intermediate_shape = tuple([shape_strides[x][0] for x in argsort(permute_indexes)])
      assert tuple([shape_strides[i][1] for i in argsort(permute_indexes)]) == strides_for_shape(intermediate_shape), "nonpermutable strides"
      if DEBUG >= 3: print(f"WARNING: this torch load is slow. CLANG to permute {intermediate_shape} with {permute_indexes}")
      assert storage[1] != dtypes.bfloat16, "can't CLANG permute BF16"
      # TODO: find a nice way to support all shapetracker on disktensors
      ret = ret.clang().reshape(intermediate_shape).permute(permute_indexes)

    return ret.reshape(size)

  class Parameter:
    def __setstate__(self, state): self.tensor = state[0]

  deserialized_objects: Dict[str, Any] = {}
  intercept = {"HalfStorage": dtypes.float16, "FloatStorage": dtypes.float32, "BFloat16Storage": dtypes.bfloat16, "IntStorage": dtypes.int32,
               "LongStorage": dtypes.int64, "_rebuild_tensor_v2": _rebuild_tensor_v2, "FloatTensor": None, "Parameter": Parameter}
  whitelist = {"torch", "collections", "numpy", "_codecs"}  # NOTE: this is not for security, only speed
  class Dummy: pass
  class TorchPickle(pickle.Unpickler):
    def find_class(self, module, name):
      module_root = module.split(".")[0]
      if module_root not in whitelist:
        if DEBUG >= 2: print(f"WARNING: returning Dummy for {module} {name}")
        return Dummy
      return intercept[name] if module_root == "torch" else super().find_class(module, name)
    def persistent_load(self, pid): return deserialized_objects.get(pid, pid)

  if zipfile.is_zipfile(fn):
    myzip = zipfile.ZipFile(fn, 'r')
    base_name = myzip.namelist()[0].split('/', 1)[0]
    for n in myzip.namelist():
      if n.startswith(f'{base_name}/data/'):
        with myzip.open(n) as myfile:
          offsets[n.split("/")[-1]] = myfile._orig_compress_start # type: ignore
    with myzip.open(f'{base_name}/data.pkl') as myfile:
      return TorchPickle(myfile).load()
  elif tarfile.is_tarfile(fn):
    with tarfile.open(fn, "r") as tar:
      storages_offset = tar.getmember('storages').offset_data
      f = unwrap(tar.extractfile('storages'))
      for i in range(TorchPickle(f).load()):  # num_storages
        (key, _, storage_type), sz = TorchPickle(f).load(), struct.unpack('<q', f.read(8))[0]
        offsets[key] = storages_offset + f.tell()
        f.seek(sz*storage_type.itemsize, 1)
      f = unwrap(tar.extractfile('tensors'))
      for _ in range(TorchPickle(f).load()):  # num_tensors
        (key, storage_id, _), ndim, _ = TorchPickle(f).load(), struct.unpack('<i', f.read(4))[0], f.read(4)
        size, stride = struct.unpack(f'<{ndim}q', f.read(8 * ndim)), struct.unpack(f'<{ndim}q', f.read(8 * ndim))
        storage_offset = struct.unpack('<q', f.read(8))[0]
        deserialized_objects[str(key)] = _rebuild_tensor_v2((None, storage_type, storage_id, None, -1), storage_offset, size, stride)
      return {k:v.tensor if isinstance(v, Parameter) else v for k,v in TorchPickle(unwrap(tar.extractfile('pickle'))).load().items()}
  else:
    with open(fn, "rb") as f:
      pkl = TorchPickle(f)
      _, _, _, rwd, _, ids, base_offset = pkl.load(), pkl.load(), pkl.load(), f.tell(), pkl.load(), pkl.load(), f.tell()
      for i in ids:
        offsets[i] = base_offset + 8
        base_offset += 8 + lens[i]
      f.seek(rwd)
      return TorchPickle(f).load()
