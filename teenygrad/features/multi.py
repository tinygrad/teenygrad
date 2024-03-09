# NOTE: this abstraction is wrong in tensor.py so we have to stub this
from typing import Tuple
from teenygrad.dtype import DType
class MultiLazyBuffer:
  device: Tuple[str, ...]
  dtype: DType
  shape: Tuple[int, ...]
  def __init__(self, lbs, axis, real=None):
    self.lbs, self.axis = lbs, axis
    raise NotImplementedError("no multibuffer support")
  @staticmethod
  def from_sharded(lb, devices, axis=None): raise NotImplementedError("no multibuffer support")
  def copy_to_device(self, device): pass