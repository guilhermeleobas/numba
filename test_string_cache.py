import unittest
import warnings
import os
import numpy as np
import shutil
import subprocess
import contextlib
import sys
from pathlib import Path
from numba import config, jit
from numba.typed import List, Dict
from numba.core.types import int64
from numba.core.errors import NumbaWarning
from numba.tests.test_caching import DispatcherCacheUsecasesTest
from string_cache import STRING_CACHE_DIR
from custom_dispatcher import ZFunctionCache


class CacheStringSourceBaseTest(DispatcherCacheUsecasesTest):
    def setUp(self):
        super().setUp()
        self.old_cache_dir = os.environ.get("NUMBA_CACHE_DIR", None)
        #     os.environ['NUMBA_CACHE_DIR'] = self.cache_dir
        os.environ["NUMBA_CACHE_DIR"] = ""
        config.reload_config()
        self.cache_dir = STRING_CACHE_DIR
        p = Path(STRING_CACHE_DIR)
        if p.exists():
            shutil.rmtree(p)

    def tearDown(self):
        super().tearDown()
        if self.old_cache_dir:
            os.environ["NUMBA_CACHE_DIR"] = self.old_cache_dir
        config.reload_config()


class TestCacheStringSource(CacheStringSourceBaseTest):
    """Test cache for String source functions

    It mirrors the tests done on caching of normal functions, except:
     - it does not test object mode
     - it does not test generated jit

    """

    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "cache_usecases.py")
    modname = "dispatcher_caching_test_folder"

    def run_in_separate_process(self, *, envvars={}):
        # Cached functions can be run from a distinct process.
        # Also stresses issue #1603: uncached function calling cached function
        # shouldn't fail compiling.
        code = """if 1:
            import sys
            from string_cache import *

            sys.path.insert(0, %(tempdir)r)
            mod = __import__(%(modname)r)
            mod.self_test()
            """ % dict(tempdir=self.tempdir, modname=self.modname)

        subp_env = os.environ.copy()
        subp_env.update(envvars)
        popen = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=subp_env,
        )
        out, err = popen.communicate()
        if popen.returncode != 0:
            raise AssertionError(
                "process failed with code %s: \n"
                "stdout follows\n%s\n"
                "stderr follows\n%s\n" % (popen.returncode, out.decode(), err.decode()),
            )

    def test_string_source(self):
        self.check_pycache(0)
        mod = self.import_module()
        self.check_pycache(0)

        mod.load(mod.add_usecase)
        f = mod.str_add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(2)  # 1 index, 1 data
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        self.check_pycache(3)  # 1 index, 2 data
        self.check_hits(f, 0, 2)

        # Check the code runs ok from another process
        self.run_in_separate_process()

    def test_inner_then_outer(self):
        # Caching inner then outer function is ok
        mod = self.import_module()
        mod.load([mod.inner, mod.outer_uncached])
        self.assertPreciseEqual(mod.str_inner(3, 2), 6)
        self.check_pycache(2)  # 1 index, 1 data
        # "str_outer_uncached" calls "inner" and not "str_inner"
        self.assertPreciseEqual(mod.inner(3, 2), 6)
        self.check_pycache(2)  # 1 index, 1 data
        # Uncached outer function shouldn't fail (issue #1603)
        f = mod.str_outer_uncached
        self.assertPreciseEqual(f(3, 2), 2)
        self.check_pycache(2)  # 1 index, 1 data

        mod = self.import_module()
        mod.load([mod.outer_uncached, mod.inner])
        f = mod.str_outer_uncached
        self.assertPreciseEqual(f(3, 2), 2)
        self.check_pycache(2)  # 1 index, 1 data
        # Cached outer will create new cache entries
        f = mod.str_outer
        self.assertPreciseEqual(f(3, 2), 2)
        self.check_pycache(4)  # 2 index, 2 data
        self.assertPreciseEqual(f(3.5, 2), 2.5)
        self.check_pycache(6)  # 2 index, 4 data

    def test_outer_then_inner(self):
        # Caching outer then inner function is ok
        mod = self.import_module()
        mod.load([mod.outer_uncached, mod.inner])
        self.assertPreciseEqual(mod.str_outer(3, 2), 2)
        self.check_pycache(4)  # 2 index, 2 data
        self.assertPreciseEqual(mod.str_outer_uncached(3, 2), 2)
        # str_outer_uncached calls inner and not str_inner
        self.check_pycache(4)  # same
        mod = self.import_module()
        mod.load(mod.inner)
        f = mod.str_inner
        self.assertPreciseEqual(f(3, 2), 6)
        self.check_pycache(4)  # same
        self.assertPreciseEqual(f(3.5, 2), 6.5)
        self.check_pycache(5)  # 2 index, 3 data

    @unittest.expectedFailure
    def test_ctypes(self):
        # Functions using a ctypes pointer can't be cached and raise
        # a warning.
        mod = self.import_module()

        mod.load([mod.use_c_sin, mod.use_c_sin_nest1, mod.use_c_sin_nest2])
        for f in [mod.str_use_c_sin, mod.str_use_c_sin_nest1, mod.str_use_c_sin_nest2]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", NumbaWarning)

                self.assertPreciseEqual(f(0.0), 0.0)
                self.check_pycache(0)

            self.assertGreater(len(w), 0)
            self.assertTrue(
                any(
                    f'Cannot cache compiled function "{f.__name__}' in str(warn.message)
                    for warn in w
                )
            )

    @unittest.skip  # slow when pickle is used
    def test_big_array(self):
        # Code references big array globals cannot be cached
        mod = self.import_module()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", NumbaWarning)

            mod.load(mod.use_big_array)
            f = mod.str_use_big_array
            np.testing.assert_equal(f(), mod.biggie)
            self.check_pycache(0)

        self.assertEqual(len(w), 1)
        self.assertIn(
            'Cannot cache compiled function "str_use_big_array" '
            "as it uses dynamic globals",
            str(w[0].message),
        )

    def test_closure(self):
        mod = self.import_module()

        with warnings.catch_warnings():
            warnings.simplefilter("error", NumbaWarning)

            f = mod.str_closure1
            self.assertPreciseEqual(f(3), 6)  # 3 + 3 = 6
            f = mod.str_closure2
            self.assertPreciseEqual(f(3), 8)  # 3 + 5 = 8
            f = mod.str_closure3
            self.assertPreciseEqual(f(3), 10)  # 3 + 7 = 8
            f = mod.str_closure4
            self.assertPreciseEqual(f(3), 12)  # 3 + 9 = 12
            # contrary to TestCache.test_closure, each call to f(...) will
            # generate a nbi/nbc file
            self.check_pycache(8)  # 4 nbi, 4 nbc

    def test_cache_reuse(self):
        mod = self.import_module()
        mod.load([mod.add_usecase, mod.outer_uncached, mod.record_return, mod.inner])
        mod.str_add_usecase(2, 3)
        mod.str_add_usecase(2.5, 3.5)
        mod.str_outer_uncached(2, 3)
        mod.str_outer(2, 3)
        mod.str_record_return(mod.packed_arr, 0)
        mod.str_record_return(mod.aligned_arr, 1)
        # mod.generated_usecase(2, 3)
        mtimes = self.get_cache_mtimes()
        # Two signatures compiled
        self.check_hits(mod.str_add_usecase, 0, 2)

        mod2 = self.import_module()
        self.assertIsNot(mod, mod2)
        mod2.load(mod.add_usecase)
        f = mod2.str_add_usecase
        f(2, 3)
        self.check_hits(f, 1, 0)
        f(2.5, 3.5)
        self.check_hits(f, 2, 0)

        # The files haven't changed
        self.assertEqual(self.get_cache_mtimes(), mtimes)

        self.run_in_separate_process()
        self.assertEqual(self.get_cache_mtimes(), mtimes)

    def replace_in_file(self, old, new):
        with open(self.modfile, "r") as f:
            lines = f.readlines()

        with open(self.modfile, "w") as f:
            for line in lines:
                f.write(line if line.strip() != old else new)

    def test_cache_invalidate(self):
        mod = self.import_module()
        mod.load(mod.add_usecase)
        f = mod.str_add_usecase
        # self.assertPreciseEqual(f(2, 3), 6)

        # This should change the functions' results
        self.replace_in_file("Z = 1", "Z = 10")

        mod = self.import_module()
        mod.load(mod.add_usecase)
        f = mod.str_add_usecase
        self.assertPreciseEqual(f(2, 3), 15)

    def test_same_names(self):
        # Function with the same names should still disambiguate
        mod = self.import_module()
        f = mod.str_renamed_function1
        self.assertPreciseEqual(f(2), 4)
        f = mod.str_renamed_function2
        self.assertPreciseEqual(f(2), 8)

    def test_primitive(self):
        mod = self.import_module()
        mod.load(mod.primitive_add)
        f = mod.str_primitive_add
        self.check_pycache(0)

        self.assertPreciseEqual(f(2, 3), 5)
        self.check_pycache(2)  # 1 nbi + 1 nbc
        self.assertPreciseEqual(f(2, 3), 5)

        self.assertPreciseEqual(f(2.2, 3.3), 5.5)
        self.check_pycache(3)  # 1 nbi + 2 nbc
        self.check_hits(f, 0, 2)

        mod = self.import_module()
        mod.load(mod.primitive_add)
        f = mod.str_primitive_add
        self.assertPreciseEqual(f(4, 4), 8)
        self.check_pycache(3)  # caching won't change by reloading the module
        self.check_hits(f, 1, 0)

        self.assertPreciseEqual(f("hello", ", world"), "hello, world")
        self.check_pycache(4)  # 1 nbi + 3 nbc
        self.check_hits(f, 1, 1)

    def test_tuple_append(self):
        mod = self.import_module()
        mod.load(mod.tuple_append)
        f = mod.str_tuple_append

        t, e = (1, 2), 3

        self.assertPreciseEqual(f(t, e), t + (e,))
        self.check_pycache(2)
        self.check_hits(f, 0, 1)

        mod = self.import_module()
        mod.load(mod.tuple_append)
        f = mod.str_tuple_append
        self.assertPreciseEqual(f(t, -e), t + (-e,))
        self.check_pycache(2)
        self.check_hits(f, 1, 0)

    def test_typedlist(self):
        mod = self.import_module()
        mod.load(mod.typed_list_append)
        f = mod.str_typed_list_append
        lst = List([1, 2])
        self.assertPreciseEqual(f(lst, 3), List([1, 2, 3]))
        self.check_pycache(2)
        self.check_hits(f, 0, 1)

        mod = self.import_module()
        mod.load(mod.typed_list_append)
        f = mod.str_typed_list_append
        self.assertPreciseEqual(f(lst, -3), List([1, 2, 3, -3]))
        self.check_pycache(2)
        self.check_hits(f, 1, 0)

    def test_typeddict(self):
        mod = self.import_module()
        mod.load(mod.typed_dict_add)
        f = mod.str_typed_dict_add
        d = Dict.empty(key_type=int64, value_type=int64)
        de = Dict.empty(key_type=int64, value_type=int64)
        de[3] = 4
        self.assertPreciseEqual(f(d, 3, 4), de)
        self.check_pycache(2)
        self.check_hits(f, 0, 1)

        mod = self.import_module()
        mod.load(mod.typed_dict_add)
        f = mod.str_typed_dict_add
        de[5] = 6
        self.assertPreciseEqual(f(d, 5, 6), de)
        self.check_pycache(2)
        self.check_hits(f, 1, 0)

    def test_ndarray(self):
        mod = self.import_module()
        mod.load(mod.ndarray)
        f = mod.str_ndarray
        arr = np.arange(5)
        self.assertPreciseEqual(f(arr), np.zeros_like(arr))
        self.check_pycache(2)
        self.check_hits(f, 0, 1)

        mod = self.import_module()
        mod.load(mod.ndarray)
        f = mod.str_ndarray
        arr += 20
        self.assertPreciseEqual(f(arr), np.ones_like(arr))
        self.check_pycache(2)
        self.check_hits(f, 1, 0)

    def test_fn_incr_or_decr_simple(self):
        mod = self.import_module()
        mod.load(mod.incr_or_decr)
        f = mod.str_incr_or_decr

        self.assertPreciseEqual(f("incr", 3), 4)
        self.check_pycache(2)
        self.check_hits(f, 0, 1)

        mod = self.import_module()
        mod.load(mod.incr_or_decr)
        f = mod.str_incr_or_decr
        self.assertPreciseEqual(f("incr", 4), 5)
        self.check_pycache(2)
        self.check_hits(f, 1, 0)

    def test_fn_incr_or_decr(self):
        mod = self.import_module()
        mod.load(mod.incr_or_decr)
        f = mod.str_incr_or_decr

        self.assertPreciseEqual(f("incr", 3), 4)
        self.check_pycache(2)
        self.check_hits(f, 0, 1)

        self.assertPreciseEqual(f("decr", 5.5), 4.5)
        self.check_pycache(3)
        self.check_hits(f, 0, 2)

        mod = self.import_module()
        mod.load(mod.incr_or_decr)
        f = mod.str_incr_or_decr
        self.assertPreciseEqual(f("incr", 4), 5)
        self.check_pycache(3)
        self.check_hits(f, 1, 0)

        self.assertPreciseEqual(f("decr", 4), 3)
        self.check_pycache(3)
        self.check_hits(f, 1, 0)

    def test_fn_incr_or_decr_by_Z_simple(self):
        mod = self.import_module()
        mod.load(mod.incr_or_decr_by_Z)
        f = mod.str_incr_or_decr_by_Z

        self.assertPreciseEqual(f("incr", 3), 4)
        self.check_pycache(2)
        self.check_hits(f, 0, 1)

        mod = self.import_module()
        mod.load(mod.incr_or_decr_by_Z)
        f = mod.str_incr_or_decr_by_Z
        self.assertPreciseEqual(f("incr", 3), 4)
        self.check_pycache(2)
        self.check_hits(f, 1, 0)

        self.replace_in_file("Z = 1", "Z = 10")

        mod = self.import_module()
        mod.load(mod.incr_or_decr_by_Z)
        f = mod.str_incr_or_decr_by_Z
        self.assertPreciseEqual(f("incr", 4), 14)
        self.check_pycache(4)
        self.check_hits(f, 0, 1)

    def test_all_incr_decr_fns(self):
        mod = self.import_module()
        mod.load(mod.all_incr_decr_fns)
        f = mod.str_all_incr_decr_fns

        self.assertPreciseEqual(f(3), 3)
        self.check_pycache(2)
        self.check_hits(f, 0, 1)

        # reloading the module should use what we already have in cache
        mod = self.import_module()
        mod.load(mod.all_incr_decr_fns)
        f = mod.str_all_incr_decr_fns
        self.assertPreciseEqual(f(3), 3)
        self.check_pycache(2)
        self.check_hits(f, 1, 0)

        self.replace_in_file("Z = 1", "Z = 10")

        mod = self.import_module()
        mod.load(mod.all_incr_decr_fns)
        f = mod.str_all_incr_decr_fns
        self.assertPreciseEqual(f(4), 4)
        self.check_pycache(4)
        self.check_hits(f, 0, 1)

    def test_func(self):
        mod = self.import_module()
        incr = jit(nopython=True)(mod.incr)
        decr = jit(nopython=True)(mod.decr)
        identity = jit(nopython=True)(mod.identity)
        mod.load(mod.func)
        f = mod.str_func

        self.assertPreciseEqual(f(incr, 3), 4)
        self.check_pycache(2)
        self.check_hits(f, 0, 1)

        self.assertPreciseEqual(f(decr, 5.5), 4.5)
        self.check_pycache(3)
        self.check_hits(f, 0, 2)

        mod = self.import_module()
        mod.load(mod.func)
        f = mod.str_func
        self.assertPreciseEqual(f(incr, 4), 5)
        self.check_pycache(3)
        self.check_hits(f, 1, 0)

        self.assertPreciseEqual(f(identity, 4), 4)
        self.check_pycache(4)
        self.check_hits(f, 1, 1)

    def test_outer0(self):
        mod = self.import_module()
        mod.load(mod.outer0)
        f = mod.str_outer0
        self.assertNotIsInstance(f._cache, ZFunctionCache)
        self.assertPreciseEqual(f(mod.inner1, 3), 4)
        self.check_pycache(2)
        self.check_hits(f, 0, 1)

        mod = self.import_module()
        mod.load(mod.outer0)
        f = mod.str_outer0
        self.assertPreciseEqual(f(mod.inner1, 3), 4)
        self.check_pycache(3)
        self.check_hits(f, 0, 1)

        self.replace_in_file("Z = 1", "Z = 10")
        mod = self.import_module()
        mod.load(mod.outer0)
        f = mod.str_outer0
        self.assertPreciseEqual(f(mod.inner1, 3), 13)
        self.check_pycache(4)
        self.check_hits(f, 0, 1)


class TestZCacheStringSource(TestCacheStringSource):
    def setUp(self):
        os.environ["USE_Z_CACHE"] = "1"
        return super().setUp()

    def tearDown(self):
        del os.environ["USE_Z_CACHE"]

    def test_custom_target(self):
        @jit(_target="Z", nopython=True)
        def incr(x):
            return x + 1

        @jit(_target="Z", nopython=True, cache=True)
        def foo(f, x):
            return f(x)

        self.assertEqual(foo(incr, 3), 4)
        self.assertIsInstance(foo._cache, ZFunctionCache)

    def test_outer0(self):
        mod = self.import_module()
        mod.load(mod.outer0)
        f = mod.str_outer0
        self.assertIsInstance(f._cache, ZFunctionCache)
        self.assertPreciseEqual(f(mod.inner1, 3), 4)
        self.check_pycache(2)
        self.check_hits(f, 0, 1)

        mod = self.import_module()
        mod.load(mod.outer0)
        f = mod.str_outer0
        self.assertPreciseEqual(f(mod.inner1, 3), 4)
        self.check_pycache(2)
        self.check_hits(f, 1, 0)

        self.replace_in_file("Z = 1", "Z = 10")
        mod = self.import_module()
        mod.load(mod.outer0)
        f = mod.str_outer0
        # this won't retrigger recompilation
        self.assertPreciseEqual(f(mod.inner1, 3), 4)
        self.check_pycache(2)
        self.check_hits(f, 1, 0)


class TestCustomDispatcher(CacheStringSourceBaseTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "cache_usecases.py")
    modname = "dispatcher_caching_test_folder"

    @contextlib.contextmanager
    def enable_z_cache(self):
        try:
            os.environ["USE_Z_CACHE"] = "1"
            yield
        finally:
            del os.environ["USE_Z_CACHE"]

    def import_module_with_z_cache(self):
        with self.enable_z_cache():
            return self.import_module()

    def test_ZFunctionCache_testcases(self):
        mod = self.import_module_with_z_cache()
        mod.load(mod.primitive_add)
        f = mod.str_primitive_add
        self.assertIsInstance(f._cache, ZFunctionCache)
        self.assertEqual(f(1, 3), 4)
        self.check_hits(f, 0, 1)
        self.check_pycache(2)

        mod = self.import_module()
        mod.load(mod.primitive_add)
        f = mod.str_primitive_add
        self.assertNotIsInstance(f._cache, ZFunctionCache)
        self.check_pycache(2)
        self.assertEqual(f(1, 3), 4)
        self.check_pycache(2)
        self.check_hits(f, 1, 0)

        mod = self.import_module_with_z_cache()
        mod.load(mod.primitive_add)
        f = mod.str_primitive_add
        self.assertIsInstance(f._cache, ZFunctionCache)
        self.assertEqual(f(2, 3), 5)
        self.check_hits(f, 1, 0)
        self.check_pycache(2)

    def test_higher_order_function_no_whitelist(self):
        mod = self.import_module()
        mod.load(mod.outer0)
        f = mod.str_outer0
        self.assertNotIsInstance(f._cache, ZFunctionCache)
        self.assertEqual(f(mod.jit_incr, 1), 2)
        self.check_hits(f, 0, 1)
        self.check_pycache(2)

        # first reload with Z cache enabled + no whitelist
        mod = self.import_module()
        mod.load(mod.outer0)
        f = mod.str_outer0
        self.assertNotIsInstance(f._cache, ZFunctionCache)
        self.assertEqual(f(mod.jit_incr, 1), 2)
        self.check_hits(f, 0, 1)
        self.check_pycache(3)  # jit an extra function

    def test_higher_order_function_whitelist(self):
        mod = self.import_module_with_z_cache()
        mod.load(mod.outer0)
        f = mod.str_outer0
        self.assertIsInstance(f._cache, ZFunctionCache)
        self.assertEqual(f(mod.jit_incr, 1), 2)
        self.check_hits(f, 0, 1)
        self.check_pycache(2)

        # first reload with whitelist enabled
        mod = self.import_module_with_z_cache()
        mod.load(mod.outer0)
        f = mod.str_outer0
        self.assertIsInstance(f._cache, ZFunctionCache)
        self.assertEqual(f(mod.jit_incr, 1), 2)
        self.check_hits(f, 1, 0)
        self.check_pycache(2)

        # reloading the module without z cache should return to the default behavior
        mod = self.import_module()
        mod.load(mod.outer0)
        f = mod.str_outer0
        self.assertNotIsInstance(f._cache, ZFunctionCache)
        self.assertEqual(f(mod.jit_incr, 1), 2)
        self.check_hits(f, 0, 1)
        self.check_pycache(3)  # extra nbc


if __name__ == "__main__":
    unittest.main()
