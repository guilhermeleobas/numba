import ctypes
import functools
import hashlib
from collections.abc import Iterable
from types import FunctionType
from numba.core.caching import CacheImpl, _CacheLocator
from numba.core.dispatcher import Dispatcher
from numba.core import serialize
from custom_dispatcher import Z


STRING_CACHE_DIR = "./string_pycache"


@functools.singledispatch
def custom_serialize(obj):
    if isinstance(obj, (int, str, complex, float)):
        return serialize.dumps(obj)
        # return [repr(obj)]

    if isinstance(obj, Iterable):
        r = b""
        for x in obj:
            r += custom_serialize(x)
        return r
    return serialize.dumps(obj)
    # return [repr(obj)]


@custom_serialize.register(ctypes._CFuncPtr)
def custom_serialize_dispatcher(obj):
    return serialize.dumps(obj)


@custom_serialize.register(Dispatcher)
def custom_serialize_dispatcher(obj):
    return custom_serialize(obj.py_func)


@custom_serialize.register(FunctionType)
def custom_serialize_pyfunc(obj):
    buf = b""

    # co_consts => tuple containing literals
    for x in obj.__code__.co_consts:
        buf += custom_serialize(x)

    # co_names => tuple of names other than arguments and function locals
    # Distinguish functions with the same name but with different closure
    # captured values. i.e. "str_closureX" test cases
    for x in obj.__code__.co_names:
        other = obj.__globals__.get(x)
        buf += custom_serialize(other)

    return buf


class _StringCacheLocator(_CacheLocator):
    def __init__(self, py_func, py_file):
        self._py_func = py_func
        self._py_file = "<string>"
        self._identifier = self._hash(py_func)

    def get_cache_path(self):
        return STRING_CACHE_DIR
        # return config.CACHE_DIR

    @classmethod
    def _hash(cls, py_func):
        buf = custom_serialize(py_func)
        const_bytes = buf
        data = py_func.__code__.co_code + const_bytes
        return hashlib.sha256(data).hexdigest()

    def get_disambiguator(self):
        return self._identifier

    def get_source_stamp(self):
        return self._identifier

    @classmethod
    def from_function(cls, py_func, py_file):
        if not py_file == "<string>":
            return
        fname = f"<string>-{py_func.__name__}-{cls._hash(py_func)}"
        self = cls(py_func, fname)
        # try:
        #     self.ensure_cache_path()
        # except OSError:
        #     # Cannot ensure the cache directory exists or is writable
        #     return
        return self


if _StringCacheLocator not in CacheImpl._locator_classes:
    CacheImpl._locator_classes.append(_StringCacheLocator)
