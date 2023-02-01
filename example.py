from numba import njit, types
from numba.core import extending
from numba.typed import List
from functools import partial
import numpy as np
import time
import timeit
import os

def enable_fastpath():
    os.environ['USE_FASTPATH'] = '1'

def disable_fastpath():
    os.environ['USE_FASTPATH'] = ''

def _measure(size):
    # a = List(np.repeat(np.array(['foo', 'bar', 'baz']), size))
    a = List(np.repeat(np.array([1]), size))
    print(len(a))

def measure(size):
    ratio = []
    for fn in [enable_fastpath, disable_fastpath]:
        fn()
        _measure(size)
        m = partial(_measure, size)
        t = timeit.timeit(m, number=4)
        print(t)
        ratio.append(t)
        # now = time.time()
        # for i in range(4):
        #     _measure(size)
        # print(time.time() - now)

    print('ratio', ratio[0] / ratio[1])

size = 100_000
# size = 1
enable_fastpath()
_measure(size)
# measure(size)