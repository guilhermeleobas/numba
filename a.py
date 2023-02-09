from numba.core.types import *
from numba.core import types
from numba.core.typing import signature

import numpy as np
from numba import njit

@njit
def add(a, b):
    sz = len(a)
    y = np.zeros(sz, dtype=np.int64)
    for i in range(sz):
        y[i] = a[i] + b - 123
    return y

sig0 = signature(none, types.Array(int64, 1, 'C', False, aligned=True), types.int64)
sig1 = signature(none, types.Array(float64, 1, 'C', False, aligned=True), types.float64)
# sig = ...
add.compile(sig0.args)
print(add.signatures)