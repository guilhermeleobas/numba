"""
Challenges:
    - Dump types in a concise way
        + TODO
    - Include imports automatically
        + [DONE] Use pyflyby
    - Free/Global variables
        + [DONE] Insert into the global scope the missing variable

            def add(a):
                return a + b
                          vvv
                           "b" is missing from add scope

            b = @value at time "add" was created
            def add(a):
                return a + b
"""


import textwrap
import numpy as np
import logging
import string
import os
from numba import njit
from numba.core.bytecode import ByteCode
from typing import Any, Sequence, Mapping, Union
from types import ModuleType
from pyflyby._imports2s import fix_unused_and_missing_imports
from pyflyby import PythonBlock, logger


# change pyflyby logging warning to "ERROR"
# logger.setLevel(logging.ERROR)

class NumbaDumpLogger(logging.Logger):
    _LEVELS = dict( (k, getattr(logging, k))
                for k in ['DEBUG', 'INFO', 'WARNING', 'ERROR'] )

    def __init__(self, name, level):
        if isinstance(level, str):
            try:
                level = self._LEVELS[level.upper()]
            except KeyError:
                raise ValueError("Bad log level %r" % (level,))

        super().__init__(name, level)
        self.setLevel(level)


logger = NumbaDumpLogger('numba_dump_logger', os.getenv("NUMBA_DUMP_LOGGER") or "INFO")
# logger.info("teste")


class Formatter(string.Formatter):
    def __init__(self, mapping):
        super().__init__()
        self._mapping = mapping

    def _format_value(self, value):
        name = getattr(value, "__name__", str(value))
        return name

    def get_value(self, key, args, kwargs):
        try:
            return super().get_value(key, args, kwargs)
        except KeyError as e:
            value = self._mapping.get(key, None)
            if value is None:
                raise KeyError(e)

            value = self._format_value(value)
            logger.info(f"Replacing '{key}' by '{value}'")
            return value


class _FunctionIdentity:
    # mimics a Numba FunctionIdentity
    def __init__(self, code):
        self.code = code

    @property
    def __globals__(self):
        return globals()


class SourceCode:
    @classmethod
    def from_source_code(cls, source_code, mapping):
        self = object.__new__(cls)
        dedent = textwrap.dedent(source_code)
        self._src = cls.replace_placeholders(dedent, mapping)
        self._code = PythonBlock(self._src).compile()
        self._name = self._code.co_names[-1]
        return self

    @classmethod
    def replace_placeholders(cls, text, mapping):
        return Formatter(mapping).vformat(text, [], globals())

    def _get_used_globals(self):
        # XXX: Remove Numba dependency
        fi = _FunctionIdentity(self._code)
        bc = ByteCode(fi)
        unused_globals = ByteCode._compute_used_globals(
            fi, bc.table, bc.co_consts, bc.co_names
        )
        return unused_globals

    def add_missing_globals(self):
        # add global constants
        globs = self._get_used_globals()
        decls = ""
        for k, v in globs.items():
            if not (isinstance(v, ModuleType) or callable(v)):
                decl = f"{k} = {v}\n"
                logger.info(decl)
                decls += decl

        self._src = decls + self._src

    def add_missing_imports(self):
        # run pyflyby to add missing imports
        self._src = fix_unused_and_missing_imports(self._src).text.joined

    def __str__(self) -> str:
        return self._src

    def dump_to_file(self):
        filename = f"{self._function_name}.txt"
        logger.info(f"Logging function {self._name} to file {filename}")
        with open(filename, "w") as f:
            f.write(self._src)


def get_add():
    fn = """
    @{decorator}
    def add(a, b):
        sz = len(a)
        y = np.zeros(sz, dtype=np.int64)
        for i in range(sz):
            y[i] = a[i] + b - {c}
        return y
    """
    return fn


def get_func():
    func = """
    @{decorator}
    def bar(a):
        return a + np.abs(b) + {c}
    """
    return func


# argtys = (types.Array(types.int64, 1, 'C'), types.int64)

b = 3

for f in [get_func, get_add]:
    F = SourceCode.from_source_code(f(), {"decorator": njit, "c": 123})
    F.add_missing_globals()
    F.add_missing_imports()
    print(F)

# F.dump_to_file()

