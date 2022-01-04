from numba import njit
from numba.extending import constexpr
from numba.core import typing, types
import numpy as np

from numba.core.typing import typeof, templates
from numba.core import extending, types

from types import SimpleNamespace
a = SimpleNamespace(b=2)

@constexpr
def bar():
    return a.b

@njit
def foo():
    a = bar()
    return a + 1

print('result', foo())