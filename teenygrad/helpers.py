from typing import Union, Tuple, Iterator, NamedTuple, Optional, Final
import os, functools
import numpy as np
from math import prod # noqa: F401 # pylint:disable=unused-import

def dedup(x): return list(dict.fromkeys(x))   # retains list orderi
def argfix(*x): return tuple(x[0]) if x and x[0].__class__ in (tuple, list) else x
def make_pair(x:Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else x
def flatten(l:Iterator): return [item for sublist in l for item in sublist]
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python

@functools.lru_cache(maxsize=None)
def getenv(key, default=0): return type(default)(os.getenv(key, default))

DEBUG = getenv("DEBUG")
CI = os.getenv("CI", "") != ""

class DType(NamedTuple):
  priority: int  # this determines when things get upcasted
  itemsize: int
  name: str
  np: Optional[type]  # TODO: someday this will be removed with the "remove numpy" project
  sz: int = 1
  def __repr__(self): return f"dtypes.{self.name}"
  @property
  def key(self): return (self.name)

class dtypes:
  float32: Final[DType] = DType(4, 4, "float", np.float32)
  bool: Final[DType] = DType(0, 1, "bool", np.bool_)
