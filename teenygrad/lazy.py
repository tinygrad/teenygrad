from __future__ import annotations
from typing import Tuple
from teenygrad.helpers import dtypes
from teenygrad.ops import UnaryOps, BinaryOps, ReduceOps, TernaryOps, LoadOps
import numpy as np

def shape_to_axis(old_shape:Tuple[int, ...], new_shape:Tuple[int, ...]) -> Tuple[int, ...]:
  assert len(old_shape) == len(new_shape), "reduce shapes must have same dimensions"
  return tuple(i for i,(a,b) in enumerate(zip(old_shape, new_shape)) if a != b)

class LazyBuffer:
  device = "CPU"
  dtype = dtypes.float32
  realized = None

  def __init__(self, buf): self._np = buf

  @property
  def shape(self): return self._np.shape

  def contiguous(x): return x
  def realize(x): return x

  def const(self, x) -> LazyBuffer: return LazyBuffer(np.full_like(self._np, x))

  @staticmethod
  def fromCPU(x): return LazyBuffer(x)
  def toCPU(self): return self._np

  @staticmethod
  def loadop(op, shape, dtype, device, arg=None, src=None) -> LazyBuffer:
    if op == LoadOps.RAND: return LazyBuffer(np.random.default_rng(arg).random(size=shape, dtype=np.float32))
    elif op == LoadOps.CONST: return LazyBuffer(np.full(shape, arg))
    else: raise NotImplementedError(op)

  # MovementOps
  def reshape(self, arg): return LazyBuffer(self._np.reshape(arg))
  def expand(self, arg): return LazyBuffer(np.broadcast_to(self._np, arg))
  def shrink(self, arg): return LazyBuffer(self._np[tuple(slice(p[0], p[1], None) for p in arg)])
  def permute(self, arg): return LazyBuffer(self._np.transpose(arg))
  def pad(self, arg): return LazyBuffer(np.pad(self._np, arg))
  def stride(self, arg): return LazyBuffer(self._np[tuple(slice(None, None, i) for i in arg)])

  def e(self, op, *srcs):
    if op == UnaryOps.NEG: return LazyBuffer(-self._np)
    elif op == UnaryOps.EXP2: return LazyBuffer(np.exp2(self._np))
    elif op == UnaryOps.LOG2: return LazyBuffer(np.log2(self._np))
    elif op == UnaryOps.SIN: return LazyBuffer(np.sin(self._np))
    elif op == UnaryOps.SQRT: return LazyBuffer(np.sqrt(self._np))
    elif op == BinaryOps.ADD: return LazyBuffer(self._np + srcs[0]._np)
    elif op == BinaryOps.SUB: return LazyBuffer(self._np - srcs[0]._np)
    elif op == BinaryOps.MUL: return LazyBuffer(self._np * srcs[0]._np)
    elif op == BinaryOps.DIV: return LazyBuffer(self._np / srcs[0]._np)
    elif op == BinaryOps.MAX: return LazyBuffer(np.maximum(self._np, srcs[0]._np))
    elif op == BinaryOps.CMPLT: return LazyBuffer(self._np < srcs[0]._np)
    elif op == TernaryOps.WHERE: return LazyBuffer(np.where(self._np, srcs[0]._np, srcs[1]._np))
    else: raise NotImplementedError(op)

  def reduce_op(self, op, new_shape):
    if op == ReduceOps.SUM: return LazyBuffer(self._np.sum(shape_to_axis(self.shape, new_shape), keepdims=True))
    elif op == ReduceOps.MAX: return LazyBuffer(self._np.max(shape_to_axis(self.shape, new_shape), keepdims=True))
    else: raise NotImplementedError(op)
