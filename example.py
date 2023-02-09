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

import logging
import pathlib
import hashlib
import string
import textwrap
import os
from types import ModuleType

import numpy as np
from pyflyby import PythonBlock
from pyflyby import logger as pf_logger
from pyflyby._imports2s import fix_unused_and_missing_imports

from numba import njit
from numba.core.types import int64, float64, Array
from numba.core.typing import Signature, signature
from numba.core.bytecode import ByteCode


# Add common Numba imports to pyflyby
os.environ['PYFLYBY_PATH'] = './numba.py:-'

# change pyflyby logging warning to "ERROR"
# pf_logger.set_level('WARNING')
# logger = logging.getLogger(__name__)
logger = logging.getLogger()
logger.setLevel(logging.INFO)



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


class SourceCode:
    """
    """

    class _FunctionIdentity:
        """Mimics a Numba FunctionIdentity class. Code below if the minimum
        required by `_compute_used_globals` to work
        """
        # mimics a Numba FunctionIdentity
        def __init__(self, code):
            self.code = code

        @property
        def __globals__(self):
            return globals()

    def __init__(self, source_code, mapping):
        dedent = textwrap.dedent(source_code)
        self._src = self._replace_placeholders(dedent, mapping)
        self._code = PythonBlock(self._src).compile()
        self._name = self._code.co_names[-1]

    @property
    def name(self):
        return self._name

    def resolve(self):
        """
        """
        self._add_missing_globals()
        self._add_missing_imports()

    def _replace_placeholders(self, text, mapping):
        return Formatter(mapping).vformat(text, [], globals())

    def _get_used_globals(self):
        # XXX: Remove Numba dependency
        fi = self._FunctionIdentity(self._code)
        bc = ByteCode(fi)
        unused_globals = ByteCode._compute_used_globals(
            fi, bc.table, bc.co_consts, bc.co_names
        )
        return unused_globals

    def _add_missing_globals(self):
        # add global constants
        globs = self._get_used_globals()
        decls = ""
        for k, v in globs.items():
            if not (isinstance(v, ModuleType) or callable(v)):
                decl = f"{k} = {v}\n"
                logger.info(decl)
                decls += decl

        self._src = decls + self._src

    def _add_missing_imports(self):
        # run pyflyby to add missing imports
        self._src = fix_unused_and_missing_imports(self._src).text.joined

    def get_content(self) -> str:
        return self._src


class TypedSourceCode(SourceCode):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._signatures = []

    def add_signature(self, sig):
        msg = f'"sig" object must be an instance of Numba Signature. Got {type(sig)}'
        assert isinstance(sig, Signature), msg
        self._signatures.append(sig)

    @property
    def signatures(self):
        return self._signatures

    def get_content(self) -> str:
        """Return the content for"""
        top_comment = (
            f'# Reproducer for "{self._name}" to debug Numba compilation step. To execute,\n'
            f"# uncomment the last line: `# {self._name}.compile(sig)` and replace `sig`\n"
            "# by one of the available signatures\n"
        )
        content = ""
        content += f"{top_comment}\n"
        content += "from numba.core import types\n"
        content += "from numba.core.types import *\n"
        content += "from numba.core.typing import signature\n"
        content += super().get_content()
        if self._signatures:
            sigs = ""
            for idx, sig in enumerate(self._signatures):
                retty = sig.return_type
                argtys = ", ".join(map(lambda x: f"types.{x!r}", sig.args))
                sigs += f'sig{idx} = eval("signature({retty!r}, {argtys})")\n'
            s = "\n"
            s += f"{sigs}"
            s += f"# {self._name}.compile(sig.args)" + "\n"
            content += textwrap.dedent(s)

        return fix_unused_and_missing_imports(content, remove_unused=False).text.joined


def compute_hash(signatures: list[Signature]):
    hashes = []
    for sig in signatures:
        hash_ = hashlib.sha1(str(sig.args).encode()).hexdigest()
        hashes.append(hash_)
    final_hash = hashlib.sha1("".join(hashes).encode()).hexdigest()
    return final_hash


def save_source_code(F: TypedSourceCode, prefix_path=pathlib.Path(".")) -> None:
    hash_ = compute_hash(F.signatures)
    filename = prefix_path / f"{F.name}_{hash_}.py"

    logger.info(f"Saving file {filename}")

    content = F.get_content()
    with open(filename, "w") as file:
        file.write(content)


add = """
@{decorator}
def add(a, b):
    sz = len(a)
    y = np.zeros(sz, dtype=np.int64)
    for i in range(sz):
        y[i] = a[i] + b - {c}
    return np.sum(y)
"""

bar = """
@{decorator}
def bar(a):
    return a + np.abs(b) + {c}
"""


def func(source_code, mapping, sigs):
    F = TypedSourceCode(source_code, mapping)
    F.resolve()
    for sig in sigs:
        F.add_signature(sig)
    save_source_code(F)


b = 3


sig = signature(int64, int64)
func(bar, {"decorator": njit, "c": 123}, [sig])

sig = signature(int64, Array(int64, 1, "C"), int64)
sig2 = signature(float64, Array(float64, 1, "C"), float64)
func(add, {"decorator": njit, "c": 123}, [sig, sig2])
