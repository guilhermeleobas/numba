from numba import cuda, int32, float64, void, types
from numba.core.typing.templates import Registry, make_overload_template
import numpy as np
import math

from numba.cuda.cudadecl import registry, register, register_global, register_attr

def overload(func, jit_options={}, strict=True, inline='never',
             prefer_literal=False):

    # set default options
    _overload_default_jit_options = {'no_cpython_wrapper': True}
    opts = _overload_default_jit_options.copy()
    opts.update(jit_options)  # let user options override

    def decorate(overload_func):
        template = make_overload_template(func, overload_func, opts, strict,
                                          inline, prefer_literal)
        register(template)
        if callable(func):
            register_global(func, types.Function(template))
        return overload_func

    return decorate

def foo(a):
    return a + 10

@overload(foo)
def foo_impl(a):
    def impl(a):
        return a + 1.0
    return impl

# from numba import jit
@cuda.jit(void(float64[:], float64[:], float64[:]))
# @jit(nopython=True)
def f(r, x, y):
    r[0] = foo(2.0)
    # r[0] = math.log(x[0] + y[0])


x = np.zeros(1, dtype=np.float64)
y = np.zeros_like(x)
r = np.zeros(1, dtype=np.float64)

x[0] = 5.0
y[0] = 2.4

f[1, 1](r, x, y)
# f(r, x, y)

print(r[0])