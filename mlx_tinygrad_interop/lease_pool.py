from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from tinygrad import Device, Tensor
from tinygrad.dtype import DTypeLike, to_dtype
from tinygrad.runtime.ops_metal import wait_check


def _export_realized_storage(array: Any) -> dict[str, Any]:
  mx.eval(array)
  return mx.metal._unsafe_export_storage(array)


def _mlx_dtype_name(dtype: Any) -> str:
  return repr(dtype).removeprefix("mlx.core.")


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


def _capture_new_metal_command_buffers(start_idx: int):
  metal_dev = Device["METAL"]
  inflight = getattr(metal_dev, "mtl_buffers_in_flight", None)
  if inflight is None: return metal_dev, ()
  return metal_dev, tuple(inflight[start_idx:])


def _wait_for_scoped_metal_work(metal_dev: Any, command_buffers: tuple[Any, ...]) -> None:
  if not command_buffers: return
  wait_check(command_buffers[-1])
  inflight = getattr(metal_dev, "mtl_buffers_in_flight", None)
  if inflight is None: return
  for cbuf in command_buffers:
    with contextlib.suppress(ValueError):
      inflight.remove(cbuf)


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


@dataclass
class _LeaseSlot:
  borrower: Any
  generation: int = 0
  in_use: bool = False


class MlxToTinygradLease:
  __slots__ = ("_pool", "_slot_index", "_generation", "_tensor", "_released")

  def __init__(self, pool: "MlxToTinygradLeasePool", slot_index: int, generation: int, tensor: Tensor):
    self._pool, self._slot_index, self._generation = pool, slot_index, generation
    self._tensor, self._released = tensor, False

  @property
  def generation(self) -> int: return self._generation

  @property
  def key(self) -> MlxToTinygradLeaseKey: return self._pool.key

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
      _LeaseSlot(
        Tensor._unsafe_metal_borrower(
          int(template_storage["mtl_buffer_ptr"]),
          tuple(template_storage["shape"]),
          dtype=self.tg_dtype,
          byte_offset=int(template_storage["offset_bytes"]),
          buffer_nbytes=int(template_storage["buffer_nbytes"]),
          owner=template_owner,
        )
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

  def _next_available_slot(self) -> tuple[int, _LeaseSlot]:
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
    slot.generation += 1
    slot.in_use = True
    return MlxToTinygradLease(self, slot_index, slot.generation, tensor)

  def acquire_from_mlx(self, array: Any, *, owner: Any | None = None) -> MlxToTinygradLease:
    storage = _export_realized_storage(array)
    return self.acquire_from_storage(storage, owner=array if owner is None else owner)

  def run_with_mlx_tensor(self, array: Any, fn, *, owner: Any | None = None,
                          synchronize_on_release: bool | None = None):
    lease = self.acquire_from_mlx(array, owner=owner)
    metal_dev = Device["METAL"]
    start_idx = len(getattr(metal_dev, "mtl_buffers_in_flight", ()))
    try:
      result = fn(lease.tensor)
      returned_tensors = tuple(_iter_tensors(result))
      if any(t is lease.tensor for t in returned_tensors):
        raise RuntimeError("callback returned the borrowed tensor directly; return a realized copy or downstream result instead")
      for tensor in returned_tensors:
        tensor.realize()
      return result
    finally:
      if not lease._released:
        if synchronize_on_release is False:
          lease.release(synchronize=False)
        else:
          _, command_buffers = _capture_new_metal_command_buffers(start_idx)
          _wait_for_scoped_metal_work(metal_dev, command_buffers)
          slot = self._slots[lease._slot_index]
          if not slot.in_use: raise RuntimeError(f"slot {lease._slot_index} is not currently leased")
          if slot.generation != lease._generation:
            raise RuntimeError(f"stale lease generation for slot {lease._slot_index}: expected {slot.generation}, got {lease._generation}")
          slot.borrower.clear_owner()
          slot.in_use = False
          lease._released = True

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
  __slots__ = ("capacity_per_key", "_synchronize_on_release", "_pools")

  def __init__(self, *, capacity_per_key: int = 4, synchronize_on_release: bool = True):
    if capacity_per_key <= 0: raise ValueError(f"capacity_per_key must be positive, got {capacity_per_key}")
    self.capacity_per_key, self._synchronize_on_release = capacity_per_key, synchronize_on_release
    self._pools: dict[MlxToTinygradLeaseKey, MlxToTinygradLeasePool] = {}

  def acquire_from_mlx(self, array: Any, *, tg_dtype: DTypeLike, owner: Any | None = None) -> MlxToTinygradLease:
    storage = _export_realized_storage(array)
    owner_obj = array if owner is None else owner
    key = MlxToTinygradLeaseKey.from_storage(storage, tg_dtype=tg_dtype)
    if key not in self._pools:
      self._pools[key] = MlxToTinygradLeasePool(
        key=key,
        tg_dtype=tg_dtype,
        template_storage=storage,
        template_owner=owner_obj,
        capacity=self.capacity_per_key,
        synchronize_on_release=self._synchronize_on_release,
      )
    return self._pools[key].acquire_from_storage(storage, owner=owner_obj)

  def run_with_mlx_tensor(self, array: Any, *, tg_dtype: DTypeLike, fn, owner: Any | None = None,
                          synchronize_on_release: bool | None = None):
    storage = _export_realized_storage(array)
    owner_obj = array if owner is None else owner
    key = MlxToTinygradLeaseKey.from_storage(storage, tg_dtype=tg_dtype)
    if key not in self._pools:
      self._pools[key] = MlxToTinygradLeasePool(
        key=key,
        tg_dtype=tg_dtype,
        template_storage=storage,
        template_owner=owner_obj,
        capacity=self.capacity_per_key,
        synchronize_on_release=self._synchronize_on_release,
      )
    pool = self._pools[key]
    lease = pool.acquire_from_storage(storage, owner=owner_obj)
    metal_dev = Device["METAL"]
    start_idx = len(getattr(metal_dev, "mtl_buffers_in_flight", ()))
    try:
      result = fn(lease.tensor)
      returned_tensors = tuple(_iter_tensors(result))
      if any(t is lease.tensor for t in returned_tensors):
        raise RuntimeError("callback returned the borrowed tensor directly; return a realized copy or downstream result instead")
      for tensor in returned_tensors:
        tensor.realize()
      return result
    finally:
      if not lease._released:
        if synchronize_on_release is False:
          lease.release(synchronize=False)
        else:
          _, command_buffers = _capture_new_metal_command_buffers(start_idx)
          _wait_for_scoped_metal_work(metal_dev, command_buffers)
          slot = pool._slots[lease._slot_index]
          if not slot.in_use: raise RuntimeError(f"slot {lease._slot_index} is not currently leased")
          if slot.generation != lease._generation:
            raise RuntimeError(f"stale lease generation for slot {lease._slot_index}: expected {slot.generation}, got {lease._generation}")
          slot.borrower.clear_owner()
          slot.in_use = False
          lease._released = True

  @property
  def pool_count(self) -> int: return len(self._pools)

  def get_pool(self, key: MlxToTinygradLeaseKey) -> MlxToTinygradLeasePool | None:
    return self._pools.get(key)
