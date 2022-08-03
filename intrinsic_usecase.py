import numpy as np
import llvmlite.binding as ll
from numba import njit
from numba.core import types, typing, funcdesc


# loads the dynamic library
ll.load_library_permanently('./my_module.cpython-39-x86_64-linux-gnu.so')


def get_codegen(symbol):
    def codegen(context, builder, sig, args):
        fndesc = funcdesc.ExternalFunctionDescriptor(
            symbol, sig.return_type, sig.args)
        return context.call_internal(builder, fndesc, sig, args)
    return codegen


# Generate a @intrinsic at compile-time
def make_intrinsic_wrapper(symbol, signature, module_globals):
    retty = signature.return_type
    argtys = signature.args

    argnames = [f'arg{idx}' for idx in range(len(argtys))]

    fn_str = f'''
from numba.core.extending import intrinsic
from numba.core import types
@intrinsic
def {symbol}(typingctx, {", ".join(argnames)}):
    signature = types.{retty}({", ".join(argnames)})
    codegen = get_codegen("{symbol}")
    return signature, codegen
'''
    exec(fn_str, module_globals)
    fn = module_globals[symbol]
    return fn


# declare square
square = make_intrinsic_wrapper(
    'square',typing.signature(types.float64, types.float64), globals())


# declare array_sum
arr = types.Array(dtype=types.float64, ndim=1, layout='C')
array_sum = make_intrinsic_wrapper(
    'array_sum', typing.signature(types.float64, arr), globals())


@njit
def call_intrinsic(x, y):
    return square(x) + array_sum(y)


@njit
def call_intrinsic_no_promotion():
    return square(2)  # since there's no type promotion, this returns 0

x = 2.0
y = np.arange(5, dtype=np.float64)
print(f'{call_intrinsic(x, y) = }')  # returns 14.0
print(f'{call_intrinsic_no_promotion() = }')  # returns 0.0
