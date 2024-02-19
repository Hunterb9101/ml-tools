from typing import Optional, Union, Tuple, Any, Sequence, Callable
import string
from datetime import datetime as dt

import numpy as np

IntScalarOrTuple = Optional[Union[int, Tuple[int]]]
Number = Union[float, int]

def _iterable_multiply(a: Sequence[Number]) -> Number:
    res: Number = 1
    for i in a:
        res *= i
    return res


def _ndim_random(fn: Callable, size: IntScalarOrTuple, **kwargs) -> Union[Any, np.ndarray]:
    if not size:
        return fn(**kwargs)
    sz = tuple([size]) if isinstance(size, int) else size
    objs = int(_iterable_multiply(sz))
    arr: np.ndarray = np.array([fn(**kwargs) for _ in range(objs)]).reshape(sz)
    return arr


def rand_alphanum_item(length: int = 12) -> str:
    if length <= 0:
        raise ValueError(f"Invalid length={length}")
    alphanum = list(string.ascii_letters + string.digits)
    return ''.join(np.random.choice(alphanum, size=length))


def rand_date_item(start: str, end:str, fmt:str) -> str:
    """ Note that start, end, must both be pre-formatted as fmt"""
    start_dt, end_dt = dt.strptime(start, fmt), dt.strptime(end, fmt)
    start_ts, end_ts = dt.timestamp(start_dt), dt.timestamp(end_dt)
    val_ts = np.random.randint(low=int(start_ts), high=int(end_ts))
    val_dt = dt.fromtimestamp(val_ts)
    return dt.strftime(val_dt, fmt)


def rand_alphanum(length: int = 12, size: IntScalarOrTuple = None) -> Union[str, np.ndarray]:
    return _ndim_random(rand_alphanum_item, size=size, length=length)


def rand_date(start: str, end: str, fmt:str, size: IntScalarOrTuple = None) -> Union[str, np.ndarray]:
    return _ndim_random(rand_date_item, size=size, start=start, end=end, fmt=fmt)
