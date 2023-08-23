from __future__ import annotations
import time, importlib, inspect, functools, pathlib
from enum import Enum, auto
from typing import TYPE_CHECKING, Union, Type, Tuple, Any, List, Optional, Dict, Callable, cast
from teenygrad.helpers import ansilen, prod, DEBUG, getenv, GlobalCounters, DType, colored, dedup, merge_dicts
if TYPE_CHECKING:
  from teenygrad.lazy import LazyBuffer

# these are the llops your accelerator must implement, along with toCpu
# the Enum class doesn't work with mypy, this is static. sorry it's ugly
# NOTE: MOD, CMPLT don't have to be implemented on vectors, just scalars
# NOTE: rdna3 only has RECIP and not DIV. DIV and POW are on the chopping block
class UnaryOps(Enum): NOOP = auto(); EXP2 = auto(); LOG2 = auto(); CAST = auto(); SIN = auto(); SQRT = auto(); RECIP = auto() # noqa: E702
class BinaryOps(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto() # noqa: E702
class ReduceOps(Enum): SUM = auto(); MAX = auto() # noqa: E702
class TernaryOps(Enum): MULACC = auto(); WHERE = auto() # noqa: E702
class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); STRIDE = auto() # noqa: E702
class LoadOps(Enum): EMPTY = auto(); RAND = auto(); CONST = auto(); FROM = auto(); CONTIGUOUS = auto(); CUSTOM = auto() # noqa: E702

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps, LoadOps, TernaryOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[MovementOps], Type[LoadOps], Type[TernaryOps]]

class LazyOp:
  __slots__ = "op", "src", "arg", "buffers", "__weakref__"
  op: Op
  src: Tuple[Union[LazyOp, LazyBuffer], ...]
  arg: Any
  buffers: Tuple[LazyBuffer, ...]

  def __init__(self, op: Op, src: Tuple[Union[LazyOp, LazyBuffer], ...], arg: Any = None):
    self.op = op
    self.src = src
    self.arg = arg
    try:
      self.buffers = tuple([y for x in src for y in x.buffers])
    except AttributeError:
      # NOTE: the linearizer's key function maps the buffers to ints, and LOCAL_BUFFER is used. we don't care about buffers in these cases
      pass

  def __repr__(self): return f"LazyOp(op={self.op}, src={self.src}, arg={self.arg})"
  def __eq__(self, __value: object) -> bool:
    if __value.__class__ is not LazyOp: return False
    __value = cast(LazyOp, __value)
    return self.op == __value.op and self.src == __value.src and self.arg == __value.arg
  def __hash__(self) -> int: return hash((self.op, self.src, self.arg))
  @property
  def key(self): return (self.op, tuple(map(lambda x: getattr(x, "key", x), self.src)), getattr(self.arg, "key", self.arg))

  # Any == Union[LazyBuffer, DeviceBuffer]
  def map_buffers(self, real_srcs: Dict[Any, Any]) -> LazyOp: return LazyOp(self.op, tuple([y.map_buffers(real_srcs) for y in self.src]), self.arg)
  def get_lazyops(self) -> List[LazyOp]: return [self] + [item for x in self.src for item in x.get_lazyops()]

  def replace_with_movement_ops(self:LazyOp, ops:List[Tuple[MovementOps, Tuple[Any, ...]]]) -> 'LazyBuffer':
    assert self.op in BinaryOps or self.op in UnaryOps or self.op in TernaryOps
    srcs = [z.replace_with_movement_ops(ops) for z in self.src]
    return srcs[0].e(self.op, *srcs[1:], arg=self.arg)   # type: ignore

  @property
  def st(self): raise NotImplementedError
  @property
  def children(self): raise NotImplementedError
  @property
  def shape(self): raise NotImplementedError
  @property
  def realized(self): raise NotImplementedError
  @property
  def optype(self): raise NotImplementedError
  def realize(self): raise NotImplementedError

  # movement ops
  def reshape(self, _): raise NotImplementedError
  def pad(self, _): raise NotImplementedError
  def expand(self, _): raise NotImplementedError
  def permute(self, _): raise NotImplementedError
  def shrink(self, _): raise NotImplementedError
  def stride(self, _): raise NotImplementedError

# **************** Device ****************

class _Device:
  def __init__(self) -> None:
    self._buffers: List[str] = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]
    self.DEFAULT: str = functools.reduce(lambda val, ele: ele if getenv(ele) == 1 else val, self._buffers, None) or self._default_device()
  def canonicalize(self, device:Optional[str]) -> str: return (device.split(":", 1)[0].upper() + ((":"+device.split(":", 1)[1]) if ':' in device else '')).replace(":0", "") if device is not None else self.DEFAULT
  @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def __getitem__(self, x:str) -> Union[Interpreted, Compiled]:
    x = x.split(":")[0].upper()
    return [cls for cname, cls in inspect.getmembers(importlib.import_module(f'teenygrad.runtime.ops_{x.lower()}')) if (cname.lower() == x.lower() + "buffer") and x in self._buffers][0]
  def _default_device(self) -> str:
    for device in ["METAL", "CUDA", "GPU"]:
      try:
        if self[device]: return device
      except Exception: pass
    return "CPU"
Device = _Device()

# **************** for Interpreted Buffers ****************

class Interpreted:
  def __init__(self, buffer, fxn_for_op: Dict[Op, Callable], from_lazybuffer=lambda x: x.realized, to_underlying=lambda x: x._buf, from_underlying=None):
    self.buffer, self.fxn_for_op, self.from_lazybuffer, self.to_underlying = buffer, fxn_for_op, from_lazybuffer, to_underlying
    self.from_underlying = buffer if from_underlying is None else from_underlying
    self.synchronize = lambda: None
    self.codegen = None

  def exec_ast(self, ast:LazyOp, output=None, context=None, **kwargs):
    if TernaryOps.MULACC in self.fxn_for_op and ast.op == ReduceOps.SUM and ast.src[0].__class__ is LazyOp and ast.src[0].op == BinaryOps.MUL:
      ast = LazyOp(TernaryOps.MULACC, cast(LazyOp, ast.src[0]).src, ast.arg)
    created_context = context is None
    if context is None: context = dict()
    if not created_context and ast in context: return context[ast]
    srcs = [self.exec_ast(cast(LazyOp, x), context=context, **kwargs) if x.__class__ is LazyOp else self.from_lazybuffer(x) for x in ast.src]
    if DEBUG >= 3: st = time.perf_counter()
    ret = self.from_underlying(self.fxn_for_op[ast.op](*([self.to_underlying(x) for x in srcs] + ([ast.arg] if ast.arg is not None else []))))
    if output is not None and ret.dtype != output.dtype and UnaryOps.CAST in self.fxn_for_op: ret = self.from_underlying(self.fxn_for_op[UnaryOps.CAST](self.to_underlying(ret), (output.dtype, False))) # Do manual casting of ret if it does not match the required output dtype.
    if DEBUG >= 3: print(f"*** {'exec' if created_context else '    '} {GlobalCounters.mem_used/1e9:5.2f} GB {(time.perf_counter()-st)*1e3:7.2f} ms op: {ast.op:20s} out({ret.dtype.name}): {str(ret._buf.shape) if hasattr(ret._buf, 'shape') else str(len(ret._buf)):30s} in({len(srcs)}):", list(set(x._buf.shape if hasattr(x._buf, 'shape') else len(x._buf) for x in srcs)), ast.arg if ast.arg is not None else "")
    if not created_context: context[ast] = ret
    if output is not None and output.output_buffer is not None:
      assert output.output_buffer.size == ret.size, output.output_buffer.dtype == ret.dtype
      output.output_buffer._buf = ret._buf
      return output.output_buffer
    return ret

