from typing import Optional, Any
from teenygrad.dtype import DType
import numpy as np

class Device:
  DEFAULT = "CPU"
  _devices = ["CPU"]
  @staticmethod
  def canonicalize(device:Optional[str]) -> str: return "CPU"

class Buffer:
  def __init__(self, device:str, size:int, dtype:DType, opaque:Any=None, options=None):
    self.device, self.size, self.dtype, self._buf = device, size, dtype, opaque[1] if isinstance(opaque, tuple) else opaque
  def copyin(self, buf): self._buf = np.frombuffer(buf, dtype=self.dtype.np)
  def as_buffer(self): return np.require(self._buf, requirements=["C"]).data
