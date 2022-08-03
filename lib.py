import os
import numpy as np
from llvmlite.binding import Linkage
from numba.pycc.cc import CC
from numba.pycc.compiler import ModuleCompiler
from numba.core.compiler import compile_extra, Flags
from numba.core.compiler_lock import global_compiler_lock
from numba.core.runtime import nrtdynmod
from distutils import log


class CustomModuleCompiler(ModuleCompiler):
    @global_compiler_lock
    def _cull_exports(self):
        """Read all the exported functions/modules in the translator
        environment, and join them into a single LLVM module.
        """
        self.exported_function_types = {}
        self.function_environments = {}
        self.environment_gvs = {}

        codegen = self.context.codegen()
        library = codegen.create_library(self.module_name)

        # Generate IR for all exported functions
        flags = Flags()
        flags.no_compile = True
        if not self.export_python_wrap:
            flags.no_cpython_wrapper = True
            flags.no_cfunc_wrapper = True
        if self.use_nrt:
            flags.nrt = True
            # Compile NRT helpers
            nrt_module, _ = nrtdynmod.create_nrt_module(self.context)
            library.add_ir_module(nrt_module)

        for entry in self.export_entries:
            cres = compile_extra(
                self.typing_context,
                self.context,
                entry.function,
                entry.signature.args,
                entry.signature.return_type,
                flags,
                locals={},
                library=library,
            )

            func_name = cres.fndesc.llvm_func_name
            llvm_func = cres.library.get_function(func_name)
            # Explictly rename the symbol to the one used in @cc.export
            # so that one can call it from Numba @jit code
            llvm_func.name = entry.symbol
            llvm_func.linkage = "external"

            if self.export_python_wrap:
                wrappername = cres.fndesc.llvm_cpython_wrapper_name
                wrapper = cres.library.get_function(wrappername)
                wrapper.name = self._mangle_method_symbol(entry.symbol)
                wrapper.linkage = "external"
                fnty = cres.target_context.call_conv.get_function_type(
                    cres.fndesc.restype, cres.fndesc.argtypes
                )
                self.exported_function_types[entry] = fnty
                self.function_environments[entry] = cres.environment
                self.environment_gvs[entry] = cres.fndesc.env_name
            else:
                llvm_func.name = entry.symbol
                self.dll_exports.append(entry.symbol)

        if self.export_python_wrap:
            wrapper_module = library.create_ir_module("wrapper")
            self._emit_python_wrapper(wrapper_module)
            library.add_ir_module(wrapper_module)

        # Hide all functions in the DLL except those explicitly exported
        # including cc.export symbols
        # Note: this part of the code differs from Numba AOT compiler. Numba
        # hides @cc.export symbols in the shared library
        library.finalize()
        for fn in library.get_defined_functions():
            if fn.name not in self.dll_exports:
                if fn.linkage in {Linkage.private, Linkage.internal}:
                    # Private/Internal linkage must have "default" visibility
                    fn.visibility = "default"
        return library


class CustomCC(CC):
    @global_compiler_lock
    def _compile_object_files(self, build_dir):
        compiler = CustomModuleCompiler(
            self._export_entries,
            self._basename,
            self._use_nrt,
            cpu_name=self._target_cpu,
        )
        compiler.external_init_function = self._init_function
        temp_obj = os.path.join(
            build_dir, os.path.splitext(self._output_file)[-1] + ".o"
        )
        log.info("generating LLVM code for '%s' into %s", self._basename, temp_obj)
        compiler.write_native_object(temp_obj, wrap=True)
        return [temp_obj], compiler.dll_exports


cc = CustomCC("my_module")
# Uncomment the following line to print out the compilation steps
# cc.verbose = True


@cc.export("multf", "f8(f8, f8)")
@cc.export("multi", "i4(i4, i4)")
def mult(a, b):
    return a * b


@cc.export("square", "f8(f8)")
def square(a):
    return a**2


@cc.export("array_sum", "f8(f8[:])")
def array_sum(a):
    return np.sum(a)


@cc.export("np_arange", "i8[:](i4)")
def np_arange(sz):
    return np.arange(sz, dtype=np.int64)


@cc.export("np_ones_like", "i4[:](i4[:])")
def np_ones_like(arr):
    return np.ones_like(arr)


if __name__ == "__main__":
    cc.compile()
