from typing import Union, Tuple, Sequence, Any, Iterable, Dict, TypeVar
import os, functools, platform, operator

T = TypeVar("T")
U = TypeVar("U")
OSX = platform.system() == "Darwin"
def prod(x:Iterable[T]) -> Union[T,int]: return functools.reduce(operator.mul, x, 1)
def dedup(x): return list(dict.fromkeys(x))   # retains list orderi
def argfix(*x): return tuple(x[0]) if x and x[0].__class__ in (tuple, list) else x
def make_pair(x:Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else x
def flatten(l:Iterable[Iterable[T]]): return [item for sublist in l for item in sublist]
def fully_flatten(l): return [item for sublist in l for item in (fully_flatten(sublist) if isinstance(sublist, (tuple, list)) else [sublist])]
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def all_int(t: Sequence[Any]) -> bool: return all(isinstance(s, int) for s in t)
def round_up(num, amt:int): return (num+amt-1)//amt * amt
def merge_dicts(ds:Iterable[Dict[T,U]]) -> Dict[T,U]:
  assert len(kvs:=set([(k,v) for d in ds for k,v in d.items()])) == len(set(kv[0] for kv in kvs)), f"cannot merge, {kvs} contains different values for the same key"  # noqa: E501
  return {k:v for d in ds for k,v in d.items()}
def flat_mv(mv:memoryview): return mv if len(mv) == 0 else mv.cast("B", shape=(mv.nbytes,))

@functools.lru_cache(maxsize=None)
def getenv(key, default=0): return type(default)(os.getenv(key, default))

DEBUG, WINO, IMAGE = getenv("DEBUG"), getenv("WINO"), 0
CI = os.getenv("CI", "") != ""
