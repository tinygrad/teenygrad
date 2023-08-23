import ctypes
import numpy as np
from collections import defaultdict, deque
from typing import TypeVar, Type, Any, Dict, Deque, Tuple
from teenygrad.helpers import DType, dtypes, prod, GlobalCounters, ImageDType

_T = TypeVar("_T")
class RawBuffer:  # pylint: disable=abstract-method
  def __init__(self, size:int, dtype:DType, buf:Any=None, allocator:Any=None, **kwargs):
    self.size: int = size
    self.dtype: DType = dtype
    self._buf = buf if buf is not None else (allocator.alloc(size, dtype, **kwargs) if allocator else None) # If buf is provided, use it. Otherwise try to allocate from the allocator.
    self._memsz: int = size*dtype.itemsize
    self._allocator = allocator
    GlobalCounters.mem_used += self._memsz
  def __del__(self):  # NOTE: if it fails on init (bad dtype), it won't have a _memsz
    if hasattr(self, '_memsz'): GlobalCounters.mem_used -= self._memsz
    if hasattr(self, '_allocator') and self._allocator: self._allocator.free(self._buf)
  def __repr__(self): return f"buffer<{self.size}, {self.dtype}>"
  @property
  def key(self): return (self.size, self.dtype)

  # NOTE: this interface allows for 0 copy
  @classmethod
  def fromCPU(cls:Type[_T], x:np.ndarray) -> _T: raise NotImplementedError("must be implemented")
  def toCPU(self) -> np.ndarray: raise NotImplementedError("must be implemented")

class RawConst(RawBuffer): # pylint: disable=abstract-method
  def __repr__(self): return f"const<{self._buf}, {self.dtype}>"
  @property
  def key(self): return (str(self._buf), self.dtype)

def buf_is_kernel_arg(x) -> bool:
  return x.realized is not None and x.realized.__class__ is not RawConst

