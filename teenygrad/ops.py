from enum import Enum, auto

class UnaryOps(Enum): EXP2 = auto(); LOG2 = auto(); CAST = auto(); SIN = auto(); SQRT = auto(); NEG = auto() # noqa: E702
class BinaryOps(Enum):
  ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto(); CMPEQ = auto(); XOR = auto() # noqa: E702
class TernaryOps(Enum): WHERE = auto() # noqa: E702
class ReduceOps(Enum): SUM = auto(); MAX = auto() # noqa: E702
class LoadOps(Enum): EMPTY = auto(); CONST = auto(); COPY = auto(); CONTIGUOUS = auto(); CUSTOM = auto(); SYNC = auto(); WAIT = auto() # noqa: E702