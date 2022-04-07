import timeit
import numpy as np
from numba import njit

@njit
def foo(l, incr):
    s = 0
    sz = len(l)
    for i in range(0, sz, incr):
        s += l[i]
    return s


@njit
def bar(l, incr):
    s = 0
    sz = len(l)
    i = 0
    while i < sz:
        s += l[i]
        i += incr
    return s


def measure(func, *args):
    def setup():
        return func(*args)

    t = timeit.Timer(setup=setup)
    return min(t.repeat(repeat=5, number=1))

# l = np.arange(400_000_000)
# incr = 2

# print(measure(foo, l, incr))
# print(measure(bar, l, incr))

@njit
def baz():
    a = np.arange(6)
    return a.clip(5)

print(baz())
print(baz.py_func())