from enum import Enum, auto
from typing import Optional

class UnaryOps(Enum): NOOP, EXP2, LOG2, CAST, SIN, SQRT, RECIP, NEG = range(1, 9) # noqa: E702
class BinaryOps(Enum): ADD, SUB, MUL, DIV, MAX, MOD, CMPLT = range(1, 8) # noqa: E702
class ReduceOps(Enum): SUM, MAX = range(1, 3) # noqa: E702
class TernaryOps(Enum): MULACC, WHERE = range(1, 3) # noqa: E702
class MovementOps(Enum): RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, STRIDE = range(1, 7) # noqa: E702
class LoadOps(Enum): EMPTY, RAND, CONST, FROM, CONTIGUOUS, CUSTOM = range(1, 7) # noqa: E702

class Device:
  DEFAULT = "CPU"
  _buffers = ["CPU"]
  @staticmethod
  def canonicalize(device:Optional[str]) -> str: return "CPU"
