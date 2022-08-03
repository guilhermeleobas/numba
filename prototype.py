import numpy as np
import llvmlite.binding as ll
from numba import njit
from numba.core import funcdesc, types, typing, extending
from numba.core.types import float64


ll.load_library_permanently("./my_module.cpython-39-x86_64-linux-gnu.so")


class AOTExternalFunction(types.Function):
    """
    A named native function (resolvable by LLVM) accepting an explicit
    signature. For internal use only.
    """

    def __init__(self, symbol, sig):
        self.symbol = symbol
        self.sig = sig
        template = typing.make_concrete_template(symbol, symbol, [sig])
        super().__init__(template)

    @property
    def key(self):
        return self.symbol, self.sig


def get_codegen(symbol):
    def codegen(context, builder, sig, args):
        fndesc = funcdesc.ExternalFunctionDescriptor(symbol, sig.return_type, sig.args)
        out = context.call_internal(builder, fndesc, sig, args)
        return out

    return codegen


def make_AOT_external_function(symbol, sig):
    code = f"def {symbol}(*args): pass"
    exec(code, globals())
    func = globals()[symbol]

    assert isinstance(symbol, str), "symbol must be a string"
    assert isinstance(sig, typing.Signature), "signature must be of type Signature"

    obj = AOTExternalFunction(symbol, sig)
    template = obj.templates[0]
    typing.templates.infer_global(func)(template)
    extending.lower_builtin(func, *sig.args)(get_codegen(symbol))

    return func


######

# square
sig = typing.signature(float64, float64)
square = make_AOT_external_function("square", sig)

# array_sum
arr = types.Array(types.float64, 1, "C")
sig = typing.signature(float64, arr)
array_sum = make_AOT_external_function("array_sum", sig)


# importing my_module will initialize NRT
# this is required!
import my_module

# np_arange
arr = types.Array(types.int64, 1, "C")
sig = typing.signature(arr, types.int32)
np_arange = make_AOT_external_function("np_arange", sig)

# np_ones_like
arr = types.Array(types.int32, 1, "C")
sig = typing.signature(arr, arr)
np_ones_like = make_AOT_external_function("np_ones_like", sig)


@njit
def call_intrinsic(x, y):
    return square(x) + array_sum(y)


@njit
def call_intrinsic_no_promotion():
    return square(2)  # since there's no type promotion, this returns 0


@njit
def arange(sz):
    return np_arange(sz)


@njit
def ones_like(arr):
    return np_ones_like(arr)


x = 2.0
y = np.arange(5, dtype=np.float64)
arr = np.arange(5, dtype=np.int32)

print(f"{call_intrinsic(x, y) = }")  # returns 14.0
print(f"{call_intrinsic_no_promotion() = }")  # returns 14.0
print(f"{arange(5) = }")  # returns array([0, 1, 2, 3, 4])
print(f"{ones_like(arr) = }")  # returns array([1, 1, 1, 1, 1], dtype=np.int32)
