import warnings
from numba.core.errors import NumbaWarning
from numba.core.registry import CPUDispatcher, cpu_target
from numba.core import types
from numba.core.caching import FunctionCache, CompileResultCacheImpl
from numba.core.environment import lookup_environment
from numba.core.target_extension import (
    dispatcher_registry,
    target_registry,
    CPU,
)


# class ZCacheImpl(CompileResultCacheImpl):
#     def check_cachable(self, cres):
#         return True
#         mod = cres.library._final_module
#         # Find environments
#         for gv in mod.global_variables:
#             gvn = gv.name
#             env = lookup_environment(gvn)
#             if env is not None:
#                 if not env.can_cache():
#                     qualname = cres.fndesc.qualname.split(".")[-1]
#                     msg = (
#                         f'Cannot cache compiled function "{qualname}" as the '
#                         "environment cannot be cached."
#                     )
#                     warnings.warn_explicit(
#                         msg, NumbaWarning, self._locator._py_file, self._lineno
#                     )
#                     return False
#         return super().check_cachable(cres)


class ZFunctionCache(FunctionCache):
    """
    How to Use Z Function Caching in Numba is a hackish way to cache Higher
    Order Functions (HOPs). Internally this will change how Numba compute the
    caching key when a function is involved in an overload signature. For more
    info, see how the `_index_key` is implemented.

    ### How to use it

    The main change involves importing the `custom_dispatcher.py` file and
    modifying the `@jit` decorator to include `_target="Z"`.

    1. Import `custom_dispatcher.py` to import the modified dispatcher to cache
    higher order string functions.

    2. Apply the `@jit` decorator to your function, adding the `_target="Z"`
    parameter to enable Z Function caching. Here is an example:

    ```python
    import custom_dispatcher

    @jit(nopython=True)
    def pow(x):
        return x ** 2

    @jit(_target="Z", cache=True, nopython=True)
    def my_function(f, x):
        return f(x) + 2 * x + 1
    ```

    Targetting the "Z" dispatcher, "my_function" will automatically be able
    to cache not only string functions but any HOP. Use with
    caution!!!
    """

    # _impl_class = ZCacheImpl

    def _index_key(self, sig, codegen):
        def _is_first_class_function(typ):
            return isinstance(typ, types.Dispatcher)

        def _compute_custom_key(typ):
            py_func = typ.key().py_func
            return (py_func.__module__, py_func.__qualname__)

        def map_only(types, func, iterable):
            return type(iterable)(
                func(i) if isinstance(i, types) else i for i in iterable
            )

        key = super()._index_key(sig, codegen)
        if any(map(_is_first_class_function, sig)):
            sig = map_only(types.Dispatcher, _compute_custom_key, sig)
            return (sig,) + key[1:]

        return key


class Z(CPU): ...


class ZDispatcher(CPUDispatcher):
    targetdescr = cpu_target

    def enable_caching(self):
        self._cache = ZFunctionCache(self.py_func)


target_registry["Z"] = Z
dispatcher_registry[target_registry["Z"]] = ZDispatcher
