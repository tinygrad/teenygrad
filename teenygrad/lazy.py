from __future__ import annotations
from teenygrad.helpers import DType, dtypes, DEBUG
from teenygrad.ops import UnaryOps, BinaryOps, ReduceOps, TernaryOps, LoadOps
import numpy as np

class RawCPUBuffer:
  def __init__(self, x): self.x = x
  def toCPU(self): return self.x

class LazyBuffer:
  device = "CPU"

  def __init__(self, buf: np.ndarray): self._np = buf

  @property
  def dtype(self): return dtypes.from_np(self._np.dtype)
  @property
  def realized(self): return RawCPUBuffer(self._np)
  @property
  def shape(self): return self._np.shape
  def __repr__(self): return f"<LB {self.shape} {self.dtype}>"

  def schedule(self, seen=None): return []
  def is_unrealized_const(self): return False

  @staticmethod
  def fromCPU(x): return LazyBuffer(x)

  @staticmethod
  def loadop(op, shape, dtype, device, arg=None, src=None) -> LazyBuffer:
    if op == LoadOps.RAND: return LazyBuffer(np.random.default_rng(arg).random(size=shape, dtype=dtype.np))
    elif op == LoadOps.CONST: return LazyBuffer(np.full(shape, arg, dtype=dtype.np))
    elif op == LoadOps.EMPTY: return LazyBuffer(np.empty(shape, dtype=dtype.np))
    else: raise NotImplementedError(op)

  def contiguous(x): return x
  def const(self, x) -> LazyBuffer: return LazyBuffer(np.full_like(self._np, x))

  def cast(self, dtype:DType, bitcast:bool=False): return LazyBuffer(self._np.view(dtype.np) if bitcast else self._np.astype(dtype.np))

  def e(self, op, *srcs:LazyBuffer):
    if DEBUG >= 1: print(op, self, srcs)
    if op == UnaryOps.NEG: ret = -self._np
    elif op == UnaryOps.EXP2: ret = np.exp2(self._np)
    elif op == UnaryOps.LOG2: ret = np.log2(self._np)
    elif op == UnaryOps.SIN: ret = np.sin(self._np)
    elif op == UnaryOps.SQRT: ret = np.sqrt(self._np)
    elif op == BinaryOps.ADD: ret = self._np + srcs[0]._np
    elif op == BinaryOps.SUB: ret = self._np - srcs[0]._np
    elif op == BinaryOps.MUL: ret = self._np * srcs[0]._np
    elif op == BinaryOps.DIV: ret = self._np / srcs[0]._np
    elif op == BinaryOps.MAX: ret = np.maximum(self._np, srcs[0]._np)
    elif op == BinaryOps.CMPLT: ret = self._np < srcs[0]._np
    elif op == TernaryOps.WHERE: ret = np.where(self._np, srcs[0]._np, srcs[1]._np)
    else: raise NotImplementedError(op)
    return LazyBuffer(ret.astype(self.dtype.np if len(srcs) == 0 else max(self.dtype, *[x.dtype for x in srcs]).np, copy=False))

  def r(self, op, new_shape):
    if DEBUG >= 1: print(op, self, new_shape)
    assert len(self.shape) == len(new_shape), "reduce shapes must have same dimensions"
    axis = tuple(i for i,(a,b) in enumerate(zip(self.shape, new_shape)) if a != b)
    if op == ReduceOps.SUM: return LazyBuffer(self._np.sum(axis, dtype=self._np.dtype, keepdims=True))
    elif op == ReduceOps.MAX: return LazyBuffer(self._np.max(axis, keepdims=True))
    else: raise NotImplementedError(op)

  # MovementOps
  def reshape(self, arg): return LazyBuffer(self._np.reshape(arg))
  def expand(self, arg): return LazyBuffer(np.broadcast_to(self._np, arg))
  def shrink(self, arg): return LazyBuffer(self._np[tuple(slice(p[0], p[1], None) for p in arg)])
  def permute(self, arg): return LazyBuffer(self._np.transpose(arg))
  def pad(self, arg): return LazyBuffer(np.pad(self._np, arg))
  def stride(self, arg): return LazyBuffer(self._np[tuple(slice(None, None, i) for i in arg)])
