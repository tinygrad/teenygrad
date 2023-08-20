from __future__ import annotations
from typing import Tuple
from teenygrad.helpers import dtypes
from teenygrad.ops import BinaryOps, ReduceOps
import numpy as np

class Device:
  DEFAULT = "CPU"
  def canonicalize(x): return "CPU"

def shape_to_axis(old_shape:Tuple[int, ...], new_shape:Tuple[int, ...]) -> Tuple[int, ...]:
  assert len(old_shape) == len(new_shape), "reduce shapes must have same dimensions"
  return tuple(i for i,(a,b) in enumerate(zip(old_shape, new_shape)) if a != b)

class LazyBuffer:
  device = "CPU"
  dtype = dtypes.float32
  def __init__(self, buf): self._np = buf

  @property
  def shape(self): return self._np.shape

  def contiguous(x): return x

  @staticmethod
  def fromCPU(x): return LazyBuffer(x)

  @staticmethod
  def loadop(op, shape, dtype, device, arg=None, src=None) -> LazyBuffer:
    print(op, shape, dtype, device, arg, src)
    return LazyBuffer(np.empty(shape))

  def reshape(self, arg): return LazyBuffer(self._np.reshape(arg))
  def expand(self, arg): return LazyBuffer(np.broadcast_to(self._np, arg))
  def shrink(self, arg): return LazyBuffer(self._np[tuple(slice(p[0], p[1], None) for p in arg)])
  def permute(self, arg): return LazyBuffer(self._np.transpose(arg))

  def binary_op(self, op, y:LazyBuffer):
    if op == BinaryOps.MAX:
      return LazyBuffer(np.maximum(self._np, y._np))
    else:
      raise NotImplementedError(op)

  def reduce_op(self, op, new_shape):
    if op == ReduceOps.SUM:
      return LazyBuffer(self._np.sum(shape_to_axis(self.shape, new_shape), keepdims=True))
    else:
      raise NotImplementedError(op)

  def __add__(self, x:LazyBuffer): return LazyBuffer(self._np + x._np)
  def __mul__(self, x:LazyBuffer): return LazyBuffer(self._np * x._np)
