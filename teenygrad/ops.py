from enum import Enum, auto

class UnaryOps(Enum): NOOP = auto(); EXP2 = auto(); LOG2 = auto(); CAST = auto(); SIN = auto(); SQRT = auto(); RECIP = auto(); NEG = auto() # noqa: E702
class BinaryOps(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto() # noqa: E702
class ReduceOps(Enum): SUM = auto(); MAX = auto() # noqa: E702
class TernaryOps(Enum): MULACC = auto(); WHERE = auto() # noqa: E702
class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); STRIDE = auto() # noqa: E702
class LoadOps(Enum): EMPTY = auto(); RAND = auto(); CONST = auto(); FROM = auto(); CONTIGUOUS = auto(); CUSTOM = auto() # noqa: E702

class Device:
  DEFAULT = "CPU"
  _buffers = ["CPU"]
  def canonicalize(x): return "CPU"
