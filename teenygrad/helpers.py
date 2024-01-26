from typing import Union, Tuple, Iterator, Optional, Final, Any
import os, functools, platform
import numpy as np
from math import prod # noqa: F401 # pylint:disable=unused-import
from dataclasses import dataclass

OSX = platform.system() == "Darwin"
def dedup(x): return list(dict.fromkeys(x))   # retains list ordering
def argfix(*x): return tuple(x[0]) if x and x[0].__class__ in (tuple, list) else x
def make_pair(x:Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else x
def flatten(l:Iterator): return [item for sublist in l for item in sublist]
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def all_int(t: Tuple[Any, ...]) -> bool: return all(isinstance(s, int) for s in t)
def round_up(num, amt:int): return (num+amt-1)//amt * amt

@functools.lru_cache(maxsize=None)
def getenv(key, default=0): return type(default)(os.getenv(key, default))

DEBUG = getenv("DEBUG")
CI = os.getenv("CI", "") != ""

@dataclass(frozen=True, order=True)
class DType:
  priority: int  # this determines when things get upcasted
  itemsize: int
  name: str
  np: Optional[type]  # TODO: someday this will be removed with the "remove numpy" project
  sz: int = 1
  def __repr__(self): return f"dtypes.{self.name}"

class dtypes:
  @staticmethod # static methds on top, or bool in the type info will refer to dtypes.bool
  def is_int(x: DType)-> bool: return x in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
  @staticmethod
  def is_float(x: DType) -> bool: return x in (dtypes.float16, dtypes.float32, dtypes.float64)
  @staticmethod
  def is_unsigned(x: DType) -> bool: return x in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
  @staticmethod
  def from_np(x) -> DType: return DTYPES_DICT[np.dtype(x).name]
  bool: Final[DType] = DType(0, 1, "bool", np.bool_)
  float16: Final[DType] = DType(9, 2, "half", np.float16)
  half = float16
  float32: Final[DType] = DType(10, 4, "float", np.float32)
  float = float32
  float64: Final[DType] = DType(11, 8, "double", np.float64)
  double = float64
  int8: Final[DType] = DType(1, 1, "char", np.int8)
  int16: Final[DType] = DType(3, 2, "short", np.int16)
  int32: Final[DType] = DType(5, 4, "int", np.int32)
  int64: Final[DType] = DType(7, 8, "long", np.int64)
  uint8: Final[DType] = DType(2, 1, "unsigned char", np.uint8)
  uint16: Final[DType] = DType(4, 2, "unsigned short", np.uint16)
  uint32: Final[DType] = DType(6, 4, "unsigned int", np.uint32)
  uint64: Final[DType] = DType(8, 8, "unsigned long", np.uint64)

  # NOTE: bfloat16 isn't supported in numpy
  bfloat16: Final[DType] = DType(9, 2, "__bf16", None)

DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if not k.startswith('__') and not callable(v) and not v.__class__ == staticmethod}

PtrDType, ImageDType, IMAGE = None, None, 0  # junk to remove
