from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
from tinygrad import Device, Tensor
from tinygrad.device import Buffer
from tinygrad.dtype import DTypeLike, to_dtype
from tinygrad.tensor import all_tensors


def _export_realized_storage(array: Any) -> dict[str, Any]:
  mx.eval(array)
  return mx.metal._unsafe_export_storage(array)


def _mlx_dtype_name(dtype: Any) -> str:
  return repr(dtype).removeprefix("mlx.core.")


def _bytes_view(mv: memoryview) -> memoryview:
  return mv if mv.format == "B" and mv.ndim == 1 else mv.cast("B")


def _iter_tensors(obj: Any, seen: set[int] | None = None):
  if seen is None: seen = set()
  obj_id = id(obj)
  if obj_id in seen: return
  seen.add(obj_id)

  if isinstance(obj, Tensor):
    yield obj
    return
  if isinstance(obj, dict):
    for value in obj.values():
      yield from _iter_tensors(value, seen)
    return
  if isinstance(obj, (list, tuple, set, frozenset)):
    for value in obj:
      yield from _iter_tensors(value, seen)


def _snapshot_live_tensors() -> dict[int, Tensor]:
  return {id(t): t for tref in list(all_tensors) if (t := tref()) is not None}


def _tensor_base_buffer(t: Tensor) -> Buffer | None:
  if not t.uop.has_buffer_identity(): return None
  try:
    return cast(Buffer, t.uop.buffer).base
  except Exception:
    return None


def _uop_depends_on(uop: Any, target: Any) -> bool:
  seen: set[Any] = set()
  stack = [uop]
  while stack:
    cur = stack.pop()
    if cur is target: return True
    if cur in seen: continue
    seen.add(cur)
    stack.extend(cur.src)
  return False


def _reject_scoped_tensor_escapes(borrowed: Tensor, returned_tensors: tuple[Tensor, ...], pre_live_ids: set[int]) -> None:
  borrowed_base = _tensor_base_buffer(borrowed)

  if any(t is borrowed for t in returned_tensors):
    raise RuntimeError("callback returned the borrowed tensor directly; return a copied or independently realized result instead")
  if borrowed_base is not None and any(_tensor_base_buffer(t) is borrowed_base for t in returned_tensors):
    raise RuntimeError("callback returned tensor(s) that still alias the borrowed slot; copy or realize independent storage before returning")

  for tensor in returned_tensors:
    tensor.realize()

  returned_ids = {id(t) for t in returned_tensors}
  escaped: list[Tensor] = []
  for tensor_id, tensor in _snapshot_live_tensors().items():
    if tensor_id in pre_live_ids or tensor_id in returned_ids or tensor is borrowed:
      continue
    if borrowed_base is not None and _tensor_base_buffer(tensor) is borrowed_base:
      escaped.append(tensor)
      continue
    if _uop_depends_on(tensor.uop, borrowed.uop):
      escaped.append(tensor)
  if escaped:
    raise RuntimeError(
      "callback leaked tensor(s) derived from the borrowed tensor; only independent realized outputs may escape the callback"
    )


def _run_with_scoped_lease(lease: "MlxToTinygradLease", fn, *, synchronize_on_release: bool | None = None):
  borrowed = lease.tensor
  pre_live_ids = set(_snapshot_live_tensors())
  try:
    result = fn(borrowed)
    returned_tensors = tuple(_iter_tensors(result))
    _reject_scoped_tensor_escapes(borrowed, returned_tensors, pre_live_ids)
    return result
  finally:
    if not lease._released:
      lease.release(synchronize=synchronize_on_release)


def _evict_lru_idle_pool(pools: "OrderedDict[Any, Any]", *, max_pools: int | None) -> None:
  if max_pools is None or len(pools) < max_pools: return
  for key, pool in list(pools.items()):
    if pool.in_flight == 0:
      del pools[key]
      return
  raise RuntimeError(f"pool cache is full ({max_pools} keys) and every pool is still in use")


@dataclass(frozen=True)
class MlxToTinygradLeaseKey:
  shape: tuple[int, ...]
  mlx_dtype_name: str
  tinygrad_dtype_name: str
  byte_offset: int

  @staticmethod
  def from_storage(storage: dict[str, Any], *, tg_dtype: DTypeLike) -> "MlxToTinygradLeaseKey":
    return MlxToTinygradLeaseKey(
      shape=tuple(int(dim) for dim in storage["shape"]),
      mlx_dtype_name=_mlx_dtype_name(storage["dtype"]),
      tinygrad_dtype_name=to_dtype(tg_dtype).base.name,
      byte_offset=int(storage["offset_bytes"]),
    )


@dataclass(frozen=True)
class MlxToTinygradCopyKey:
  shape: tuple[int, ...]
  mlx_dtype_name: str
  tinygrad_dtype_name: str

  @staticmethod
  def from_storage(storage: dict[str, Any], *, tg_dtype: DTypeLike) -> "MlxToTinygradCopyKey":
    return MlxToTinygradCopyKey(
      shape=tuple(int(dim) for dim in storage["shape"]),
      mlx_dtype_name=_mlx_dtype_name(storage["dtype"]),
      tinygrad_dtype_name=to_dtype(tg_dtype).base.name,
    )


@dataclass
class _AliasLeaseSlot:
  borrower: Any
  tensor: Tensor
  generation: int = 0
  in_use: bool = False


@dataclass
class _CopyLeaseSlot:
  tensor: Tensor
  generation: int = 0
  in_use: bool = False


class MlxToTinygradLease:
  __slots__ = ("_pool", "_slot_index", "_generation", "_tensor", "_released")

  def __init__(self, pool: Any, slot_index: int, generation: int, tensor: Tensor):
    self._pool, self._slot_index, self._generation = pool, slot_index, generation
    self._tensor, self._released = tensor, False

  @property
  def generation(self) -> int: return self._generation

  @property
  def key(self) -> Any: return self._pool.key

  @property
  def tensor(self) -> Tensor:
    # This is the unsafe low-level lease surface. The preferred production API
    # is `run_with_mlx_tensor(...)`, which scopes use and release together.
    if self._released: raise RuntimeError("lease already released")
    return self._tensor

  def release(self, *, synchronize: bool | None = None) -> None:
    if self._released: raise RuntimeError("lease already released")
    self._pool._release(self._slot_index, self._generation, synchronize=synchronize)
    self._released = True

  def __enter__(self) -> "MlxToTinygradLease": return self

  def __exit__(self, exc_type, exc, tb) -> bool:
    if not self._released: self.release()
    return False


class MlxToTinygradLeasePool:
  __slots__ = ("key", "tg_dtype", "capacity", "_slots", "_next_slot", "_synchronize_on_release")

  def __init__(self, *, key: MlxToTinygradLeaseKey, tg_dtype: DTypeLike, template_storage: dict[str, Any], template_owner: Any,
               capacity: int = 4, synchronize_on_release: bool = True):
    if capacity <= 0: raise ValueError(f"capacity must be positive, got {capacity}")
    self.key, self.tg_dtype, self.capacity = key, to_dtype(tg_dtype), capacity
    self._next_slot, self._synchronize_on_release = 0, synchronize_on_release
    self._slots = [
      _AliasLeaseSlot(
        borrower := Tensor._unsafe_metal_borrower(
          int(template_storage["mtl_buffer_ptr"]),
          tuple(template_storage["shape"]),
          dtype=self.tg_dtype,
          byte_offset=int(template_storage["offset_bytes"]),
          buffer_nbytes=int(template_storage["buffer_nbytes"]),
          owner=template_owner,
        ),
        borrower.tensor,
      ) for _ in range(capacity)
    ]

  @classmethod
  def from_mlx(cls, array: Any, *, tg_dtype: DTypeLike, owner: Any | None = None, capacity: int = 4,
               synchronize_on_release: bool = True) -> "MlxToTinygradLeasePool":
    storage = _export_realized_storage(array)
    owner_obj = array if owner is None else owner
    return cls(
      key=MlxToTinygradLeaseKey.from_storage(storage, tg_dtype=tg_dtype),
      tg_dtype=tg_dtype,
      template_storage=storage,
      template_owner=owner_obj,
      capacity=capacity,
      synchronize_on_release=synchronize_on_release,
    )

  def _next_available_slot(self) -> tuple[int, _AliasLeaseSlot]:
    for _ in range(self.capacity):
      slot_index = self._next_slot
      self._next_slot = (self._next_slot + 1) % self.capacity
      slot = self._slots[slot_index]
      if not slot.in_use: return slot_index, slot
    raise RuntimeError(
      f"all {self.capacity} lease slots for key={self.key} are still in use; "
      "release leases or increase pool capacity"
    )

  def _validate_storage(self, storage: dict[str, Any]) -> None:
    incoming = MlxToTinygradLeaseKey.from_storage(storage, tg_dtype=self.tg_dtype)
    if incoming != self.key:
      raise ValueError(f"pool key mismatch: expected {self.key}, got {incoming}")

  def acquire_from_storage(self, storage: dict[str, Any], *, owner: Any) -> MlxToTinygradLease:
    self._validate_storage(storage)
    slot_index, slot = self._next_available_slot()
    tensor = slot.borrower.rebind(
      int(storage["mtl_buffer_ptr"]),
      owner=owner,
      shape=tuple(storage["shape"]),
      dtype_name=_mlx_dtype_name(storage["dtype"]),
      byte_offset=int(storage["offset_bytes"]),
      buffer_nbytes=int(storage["buffer_nbytes"]),
    )
    slot.tensor = tensor
    slot.generation += 1
    slot.in_use = True
    return MlxToTinygradLease(self, slot_index, slot.generation, tensor)

  def acquire_from_mlx(self, array: Any, *, owner: Any | None = None) -> MlxToTinygradLease:
    storage = _export_realized_storage(array)
    return self.acquire_from_storage(storage, owner=array if owner is None else owner)

  def run_with_mlx_tensor(self, array: Any, fn, *, owner: Any | None = None,
                          synchronize_on_release: bool | None = None):
    return _run_with_scoped_lease(
      self.acquire_from_mlx(array, owner=owner),
      fn,
      synchronize_on_release=synchronize_on_release,
    )

  def _release(self, slot_index: int, generation: int, *, synchronize: bool | None = None) -> None:
    slot = self._slots[slot_index]
    if not slot.in_use: raise RuntimeError(f"slot {slot_index} is not currently leased")
    if slot.generation != generation:
      raise RuntimeError(f"stale lease generation for slot {slot_index}: expected {slot.generation}, got {generation}")
    do_synchronize = synchronize if synchronize is not None else self._synchronize_on_release
    if do_synchronize:
      Device["METAL"].synchronize()
      slot.borrower.clear_owner()
    slot.in_use = False

  @property
  def in_flight(self) -> int: return sum(int(slot.in_use) for slot in self._slots)


class MlxToTinygradLeasePools:
  __slots__ = ("capacity_per_key", "max_pools", "_synchronize_on_release", "_pools")

  def __init__(self, *, capacity_per_key: int = 4, max_pools: int | None = 64, synchronize_on_release: bool = True):
    if capacity_per_key <= 0: raise ValueError(f"capacity_per_key must be positive, got {capacity_per_key}")
    if max_pools is not None and max_pools <= 0: raise ValueError(f"max_pools must be positive, got {max_pools}")
    self.capacity_per_key, self.max_pools, self._synchronize_on_release = capacity_per_key, max_pools, synchronize_on_release
    self._pools: OrderedDict[MlxToTinygradLeaseKey, MlxToTinygradLeasePool] = OrderedDict()

  def _get_or_create_pool(self, storage: dict[str, Any], *, tg_dtype: DTypeLike, owner_obj: Any) -> MlxToTinygradLeasePool:
    key = MlxToTinygradLeaseKey.from_storage(storage, tg_dtype=tg_dtype)
    pool = self._pools.get(key)
    if pool is not None:
      self._pools.move_to_end(key)
      return pool
    _evict_lru_idle_pool(self._pools, max_pools=self.max_pools)
    pool = MlxToTinygradLeasePool(
      key=key,
      tg_dtype=tg_dtype,
      template_storage=storage,
      template_owner=owner_obj,
      capacity=self.capacity_per_key,
      synchronize_on_release=self._synchronize_on_release,
    )
    self._pools[key] = pool
    return pool

  def acquire_from_mlx(self, array: Any, *, tg_dtype: DTypeLike, owner: Any | None = None) -> MlxToTinygradLease:
    storage = _export_realized_storage(array)
    owner_obj = array if owner is None else owner
    pool = self._get_or_create_pool(storage, tg_dtype=tg_dtype, owner_obj=owner_obj)
    return pool.acquire_from_storage(storage, owner=owner_obj)

  def run_with_mlx_tensor(self, array: Any, *, tg_dtype: DTypeLike, fn, owner: Any | None = None,
                          synchronize_on_release: bool | None = None):
    return _run_with_scoped_lease(
      self.acquire_from_mlx(array, tg_dtype=tg_dtype, owner=owner),
      fn,
      synchronize_on_release=synchronize_on_release,
    )

  @property
  def pool_count(self) -> int: return len(self._pools)

  def get_pool(self, key: MlxToTinygradLeaseKey) -> MlxToTinygradLeasePool | None:
    return self._pools.get(key)


class MlxToTinygradCopyLeasePool:
  __slots__ = ("key", "tg_dtype", "capacity", "_slots", "_next_slot", "_synchronize_on_release")

  def __init__(self, *, key: MlxToTinygradCopyKey, tg_dtype: DTypeLike, capacity: int = 4,
               synchronize_on_release: bool = True):
    if capacity <= 0: raise ValueError(f"capacity must be positive, got {capacity}")
    self.key, self.tg_dtype, self.capacity = key, to_dtype(tg_dtype), capacity
    self._next_slot, self._synchronize_on_release = 0, synchronize_on_release
    self._slots = [_CopyLeaseSlot(Tensor.empty(*key.shape, dtype=self.tg_dtype, device="METAL").realize()) for _ in range(capacity)]

  @classmethod
  def from_mlx(cls, array: Any, *, tg_dtype: DTypeLike, capacity: int = 4,
               synchronize_on_release: bool = True) -> "MlxToTinygradCopyLeasePool":
    storage = _export_realized_storage(array)
    return cls(
      key=MlxToTinygradCopyKey.from_storage(storage, tg_dtype=tg_dtype),
      tg_dtype=tg_dtype,
      capacity=capacity,
      synchronize_on_release=synchronize_on_release,
    )

  def _next_available_slot(self) -> tuple[int, _CopyLeaseSlot]:
    for _ in range(self.capacity):
      slot_index = self._next_slot
      self._next_slot = (self._next_slot + 1) % self.capacity
      slot = self._slots[slot_index]
      if not slot.in_use: return slot_index, slot
    raise RuntimeError(
      f"all {self.capacity} copy slots for key={self.key} are still in use; "
      "release leases or increase pool capacity"
    )

  def _validate_storage(self, storage: dict[str, Any]) -> None:
    incoming = MlxToTinygradCopyKey.from_storage(storage, tg_dtype=self.tg_dtype)
    if incoming != self.key:
      raise ValueError(f"copy pool key mismatch: expected {self.key}, got {incoming}")

  def acquire_from_mlx(self, array: Any) -> MlxToTinygradLease:
    storage = _export_realized_storage(array)
    self._validate_storage(storage)
    slot_index, slot = self._next_available_slot()
    cast(Buffer, slot.tensor.uop.buffer).ensure_allocated().copyin(_bytes_view(memoryview(array)))
    slot.generation += 1
    slot.in_use = True
    return MlxToTinygradLease(self, slot_index, slot.generation, slot.tensor)

  def run_with_mlx_tensor(self, array: Any, fn, *, synchronize_on_release: bool | None = None):
    return _run_with_scoped_lease(
      self.acquire_from_mlx(array),
      fn,
      synchronize_on_release=synchronize_on_release,
    )

  def _release(self, slot_index: int, generation: int, *, synchronize: bool | None = None) -> None:
    slot = self._slots[slot_index]
    if not slot.in_use: raise RuntimeError(f"slot {slot_index} is not currently leased")
    if slot.generation != generation:
      raise RuntimeError(f"stale lease generation for slot {slot_index}: expected {slot.generation}, got {generation}")
    do_synchronize = synchronize if synchronize is not None else self._synchronize_on_release
    if do_synchronize:
      Device["METAL"].synchronize()
    slot.in_use = False

  @property
  def in_flight(self) -> int: return sum(int(slot.in_use) for slot in self._slots)


class MlxToTinygradCopyLeasePools:
  __slots__ = ("capacity_per_key", "max_pools", "_synchronize_on_release", "_pools")

  def __init__(self, *, capacity_per_key: int = 4, max_pools: int | None = 64, synchronize_on_release: bool = True):
    if capacity_per_key <= 0: raise ValueError(f"capacity_per_key must be positive, got {capacity_per_key}")
    if max_pools is not None and max_pools <= 0: raise ValueError(f"max_pools must be positive, got {max_pools}")
    self.capacity_per_key, self.max_pools, self._synchronize_on_release = capacity_per_key, max_pools, synchronize_on_release
    self._pools: OrderedDict[MlxToTinygradCopyKey, MlxToTinygradCopyLeasePool] = OrderedDict()

  def _get_or_create_pool(self, storage: dict[str, Any], *, tg_dtype: DTypeLike) -> MlxToTinygradCopyLeasePool:
    key = MlxToTinygradCopyKey.from_storage(storage, tg_dtype=tg_dtype)
    pool = self._pools.get(key)
    if pool is not None:
      self._pools.move_to_end(key)
      return pool
    _evict_lru_idle_pool(self._pools, max_pools=self.max_pools)
    pool = MlxToTinygradCopyLeasePool(
      key=key,
      tg_dtype=tg_dtype,
      capacity=self.capacity_per_key,
      synchronize_on_release=self._synchronize_on_release,
    )
    self._pools[key] = pool
    return pool

  def acquire_from_mlx(self, array: Any, *, tg_dtype: DTypeLike) -> MlxToTinygradLease:
    storage = _export_realized_storage(array)
    return self._get_or_create_pool(storage, tg_dtype=tg_dtype).acquire_from_mlx(array)

  def run_with_mlx_tensor(self, array: Any, *, tg_dtype: DTypeLike, fn, synchronize_on_release: bool | None = None):
    return _run_with_scoped_lease(
      self.acquire_from_mlx(array, tg_dtype=tg_dtype),
      fn,
      synchronize_on_release=synchronize_on_release,
    )

  @property
  def pool_count(self) -> int: return len(self._pools)

  def get_pool(self, key: MlxToTinygradCopyKey) -> MlxToTinygradCopyLeasePool | None:
    return self._pools.get(key)
