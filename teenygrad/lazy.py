from __future__ import annotations
from teenygrad.helpers import DEBUG, prod
from teenygrad.dtype import DType, dtypes, least_upper_dtype
from teenygrad.device import Buffer
from teenygrad.ops import UnaryOps, BinaryOps, ReduceOps, TernaryOps, LoadOps
import numpy as np

class RawCPUBuffer:
  def __init__(self, x): self.x = x
  def toCPU(self): return self.x

class LazyBuffer:
  device = "CPU"
  def __init__(self, buf: np.ndarray): self.realized = Buffer("CPU", buf.size, dtypes.from_np(buf.dtype), buf)

  @property
  def base(self): return self
  @property
  def dtype(self): return self.realized.dtype
  @property
  def _np(self):
    if self.realized._buf is None: return np.array([], dtype=self.realized.dtype.np).reshape((0,))
    return self.realized._buf
  @property
  def shape(self): return self._np.shape
  def __repr__(self): return f"<LB {self.shape} {self.dtype}>"

  def is_unrealized_contiguous_const(self): return False
  def copy_to_device(self, device:str) -> LazyBuffer: return self

  @staticmethod
  def fromCPU(x): return LazyBuffer(x)

  @staticmethod
  def loadop(op, shape, dtype, device, arg=None, src=None, _buf=None) -> LazyBuffer:
    if op == LoadOps.CUSTOM:
      arg(ret := Buffer(device, prod(shape), dtype))
      return ret._buf.reshape(shape)
    elif op == LoadOps.CONST: return LazyBuffer(np.full(shape, arg, dtype=dtype.np))
    elif op == LoadOps.EMPTY: return LazyBuffer(_buf._buf if device == "EXT" and prod(shape) != 0 else np.empty(shape, dtype=dtype.np))
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
    elif op == BinaryOps.XOR: ret = self._np ^ srcs[0]._np
    elif op == BinaryOps.MAX: ret = np.maximum(self._np, srcs[0]._np)
    elif op == BinaryOps.CMPLT: ret = self._np < srcs[0]._np
    elif op == BinaryOps.CMPEQ: ret = self._np == srcs[0]._np
    elif op == TernaryOps.WHERE: ret = np.where(self._np, srcs[0]._np, srcs[1]._np)
    else: raise NotImplementedError(op)
    new_type = least_upper_dtype(self.dtype, *[x.dtype for x in srcs]) if op not in (BinaryOps.CMPLT, BinaryOps.CMPEQ) else dtypes.bool
    return LazyBuffer(ret.astype(new_type.np, copy=False))

  def r(self, op, axis):
    if DEBUG >= 1: print(op, self, axis)
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
