import os
import inspect
import textwrap
import numpy as np
from functools import partial
from numba import jit, literal_unroll
from numba.core.dispatcher import Dispatcher
from numba.tests.cache_usecases import (
    add_usecase,
    inner,
    outer_uncached,
    use_c_sin,
    use_c_sin_nest1,
    use_c_sin_nest2,
    use_big_array,
    record_return,
    c_sin,
    biggie,
    packed_arr,
    aligned_arr,
    packed_record_type,
    aligned_record_type,
    self_test,
)

if os.environ.get("USE_Z_CACHE", False):
    jit = partial(jit, _target="Z")

Z = 1


def incr(x):
    return x + 1


def decr(x):
    return x - 1


def incr_by_Z(x):
    return x + Z


def decr_by_Z(x):
    return x - Z


def identity(x):
    return x


@jit(nopython=True, cache=True)
def primitive_add(x, y):
    return x + y


@jit(nopython=True, cache=True)
def tuple_append(t, e):
    return t + (e,)


@jit(nopython=True, cache=True)
def typed_list_append(lst, e):
    lst.append(e)
    return lst


@jit(nopython=True, cache=True)
def typed_dict_add(d, k, v):
    d[k] = v
    return d


@jit(nopython=True, cache=True)
def ndarray(arr):
    if arr.sum() > 10:
        return np.ones_like(arr)
    else:
        return np.zeros_like(arr)


@jit(nopython=True, cache=True)
def func(fn, x):
    return fn(x)


jit_incr = jit(nopython=True)(incr)
jit_decr = jit(nopython=True)(decr)
jit_incr_by_Z = jit(nopython=True)(incr_by_Z)
jit_decr_by_Z = jit(nopython=True)(decr_by_Z)


jit_fns = (jit_incr, jit_decr, jit_incr_by_Z, jit_decr_by_Z)


@jit(nopython=True, cache=True)
def incr_or_decr(kind, x):
    if kind == "incr":
        return jit_incr(x)
    else:
        return jit_decr(x)


@jit(nopython=True, cache=True)
def incr_or_decr_by_Z(kind, x):
    if kind == "incr":
        return jit_incr_by_Z(x)
    else:
        return jit_decr_by_Z(x)


@jit(nopython=True, cache=True)
def all_incr_decr_fns(x):
    for fn in literal_unroll(jit_fns):
        x = fn(x)
    return x


@jit(nopython=True)
def inner1(x):
    return x + Z


@jit(nopython=True, cache=True)
def outer0(f, x):
    return f(x)


def make_str_closure(x):
    ns = {"jit": jit, "x": x}
    fc_txt = """
    @jit(cache=True, nopython=True)
    def closure(y):
        return x + y
    """
    fc_txt = textwrap.dedent(fc_txt)
    exec(fc_txt, ns, ns)
    return ns["closure"]


str_closure1 = make_str_closure(3)
str_closure2 = make_str_closure(5)
str_closure3 = make_str_closure(7)
str_closure4 = make_str_closure(9)

# String Source Functions
# Many functions above will be reconstructed as string-source function to
# test the ability to cache this type of functions. The original name is
# prepended with "str"

all_fns = [
    add_usecase,
    inner,
    outer_uncached,
    use_c_sin,
    use_c_sin_nest1,
    use_c_sin_nest2,
    use_big_array,
    record_return,
    primitive_add,
    typed_list_append,
    typed_dict_add,
    tuple_append,
    ndarray,
    func,
    incr_or_decr,
    incr_or_decr_by_Z,
    all_incr_decr_fns,
    outer0,
]


def load(fns):
    if not isinstance(fns, (list, tuple)):
        fns = [fns]

    for fc in fns:
        if isinstance(fc, Dispatcher):
            fc = fc.py_func
        fc_txt = inspect.getsource(fc).replace("def ", "def str_")
        fn_name = f"str_{fc.__name__}"
        exec(fc_txt, globals(), locals())
        globals()[fn_name] = locals()[fn_name]


def load_all():
    return load(all_fns)


fc_txt = """@jit(cache=True, nopython=True)
def str_outer(x, y):
    return str_inner(-y, x)
"""
exec(fc_txt)

fc_txt = """@jit(cache=True, nopython=True)
def ambiguous_function(x):
    return x + 2
"""
ns = {"jit": jit}
exec(fc_txt, ns, ns)
str_renamed_function1 = ns["ambiguous_function"]

fc_txt = """@jit(cache=True, nopython=True)
def ambiguous_function(x):
    return x + 6
"""
ns = {"jit": jit}
exec(fc_txt, ns, ns)
str_renamed_function2 = ns["ambiguous_function"]
