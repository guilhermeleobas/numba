import functools
import itertools
import pickle
import textwrap
import warnings

import numpy as np

from numba import njit, vectorize
from numba.tests.support import MemoryLeakMixin, TestCase
from numba.core.errors import (TypingError, NumbaNotImplementedError,
                               NumbaExperimentalFeatureWarning)
import unittest
from numba.np.ufunc import dufunc
from numba.np.numpy_support import from_dtype


def pyuadd(a0, a1):
    return a0 + a1


def pysub(a0, a1):
    return a0 - a1


def pymult(a0, a1):
    return a0 * a1


def pydiv(a0, a1):
    return a0 // a1


def pymin(a0, a1):
    return a0 if a0 < a1 else a1


class TestDUFunc(MemoryLeakMixin, unittest.TestCase):

    def nopython_dufunc(self, pyfunc):
        return dufunc.DUFunc(pyfunc, targetoptions=dict(nopython=True))

    def test_frozen(self):
        duadd = self.nopython_dufunc(pyuadd)
        self.assertFalse(duadd._frozen)
        duadd._frozen = True
        self.assertTrue(duadd._frozen)
        with self.assertRaises(ValueError):
            duadd._frozen = False
        with self.assertRaises(TypeError):
            duadd(np.linspace(0,1,10), np.linspace(1,2,10))

    def test_scalar(self):
        duadd = self.nopython_dufunc(pyuadd)
        self.assertEqual(pyuadd(1,2), duadd(1,2))

    def test_npm_call(self):
        duadd = self.nopython_dufunc(pyuadd)

        @njit
        def npmadd(a0, a1, o0):
            duadd(a0, a1, o0)
        X = np.linspace(0,1.9,20)
        X0 = X[:10]
        X1 = X[10:]
        out0 = np.zeros(10)
        npmadd(X0, X1, out0)
        np.testing.assert_array_equal(X0 + X1, out0)
        Y0 = X0.reshape((2,5))
        Y1 = X1.reshape((2,5))
        out1 = np.zeros((2,5))
        npmadd(Y0, Y1, out1)
        np.testing.assert_array_equal(Y0 + Y1, out1)
        Y2 = X1[:5]
        out2 = np.zeros((2,5))
        npmadd(Y0, Y2, out2)
        np.testing.assert_array_equal(Y0 + Y2, out2)

    def test_npm_call_implicit_output(self):
        duadd = self.nopython_dufunc(pyuadd)

        @njit
        def npmadd(a0, a1):
            return duadd(a0, a1)
        X = np.linspace(0,1.9,20)
        X0 = X[:10]
        X1 = X[10:]
        out0 = npmadd(X0, X1)
        np.testing.assert_array_equal(X0 + X1, out0)
        Y0 = X0.reshape((2,5))
        Y1 = X1.reshape((2,5))
        out1 = npmadd(Y0, Y1)
        np.testing.assert_array_equal(Y0 + Y1, out1)
        Y2 = X1[:5]
        out2 = npmadd(Y0, Y2)
        np.testing.assert_array_equal(Y0 + Y2, out2)
        out3 = npmadd(1.,2.)
        self.assertEqual(out3, 3.)

    def test_ufunc_props(self):
        duadd = self.nopython_dufunc(pyuadd)
        self.assertEqual(duadd.nin, 2)
        self.assertEqual(duadd.nout, 1)
        self.assertEqual(duadd.nargs, duadd.nin + duadd.nout)
        self.assertEqual(duadd.ntypes, 0)
        self.assertEqual(duadd.types, [])
        self.assertEqual(duadd.identity, None)
        duadd(1, 2)
        self.assertEqual(duadd.ntypes, 1)
        self.assertEqual(duadd.ntypes, len(duadd.types))
        self.assertIsNone(duadd.signature)

    def test_ufunc_props_jit(self):
        duadd = self.nopython_dufunc(pyuadd)
        duadd(1, 2)  # initialize types attribute

        attributes = {'nin': duadd.nin,
                      'nout': duadd.nout,
                      'nargs': duadd.nargs,
                      #'ntypes': duadd.ntypes,
                      #'types': duadd.types,
                      'identity': duadd.identity,
                      'signature': duadd.signature}

        def get_attr_fn(attr):
            fn = f'''
                def impl():
                    return duadd.{attr}
            '''
            l = {}
            exec(textwrap.dedent(fn), {'duadd': duadd}, l)
            return l['impl']

        for attr, val in attributes.items():
            cfunc = njit(get_attr_fn(attr))
            self.assertEqual(val, cfunc(),
                             f'Attribute differs from original: {attr}')

        # We don't expose [n]types attributes as they are dynamic attributes
        # and can change as the user calls the ufunc
        # cfunc = njit(get_attr_fn('ntypes'))
        # self.assertEqual(cfunc(), 1)
        # duadd(1.1, 2.2)
        # self.assertEqual(cfunc(), 2)


class TestDUFuncMethodsBase(TestCase):

    @functools.cache
    def _generate_jit(self, ufunc, kind, identity=None):
        assert kind in ('reduce', 'reduceat', 'at')
        if kind == 'reduce':
            if ufunc.nin == 2:
                vec = vectorize(identity=identity)(lambda a, b: ufunc(a, b))
            else:
                vec = vectorize(identity=identity)(lambda a: ufunc(a))

            @njit
            def fn(array, axis=0, initial=None):
                return vec.reduce(array, axis=axis, initial=initial)
            return fn
        elif kind == 'reduceat':
            if ufunc.nin != 2:
                raise ValueError('reduceat only supported for binary functions')
            vec = vectorize(identity=identity)(lambda a, b: ufunc(a, b))

            @njit
            def fn(array, indices, axis=0, dtype=None, out=None):
                return vec.reduceat(array, indices, axis, dtype, out)
            return fn
        else:
            if ufunc.nin == 2:
                vec = vectorize(identity=identity)(lambda a, b: ufunc(a, b))
            else:
                vec = vectorize(identity=identity)(lambda a: ufunc(a))

            @njit
            def fn(*args):
                return vec.at(*args)
            return fn

    def _reduce(self, ufunc, identity):
        return self._generate_jit(ufunc, 'reduce', identity=identity)

    def _reduceat(self, ufunc):
        return self._generate_jit(ufunc, 'reduceat')

    def _at(self, ufunc):
        return self._generate_jit(ufunc, 'at')


class TestDUFuncAt(TestDUFuncMethodsBase):
    def _compare_output(self, fn, ufunc, a, *args):
        expected = a.copy()
        got = a.copy()
        ufunc.at(expected, *args)
        fn(got, *args)
        self.assertPreciseEqual(expected, got)

    def test_numpy_ufunc_at_basic(self):
        # tests taken from: https://github.com/numpy/numpy/blob/27d8c43eb958b4ecee59b4d66908750759a9afc2/numpy/core/tests/test_ufunc.py#L1974-L2003  # noqa: E501
        # NumPy also test this function with a Rational array dtype. We skip
        # this test as Numba doesn't support Rational
        a = np.arange(10, dtype=int)

        add_at = self._at(np.add)
        negative_at = self._at(np.negative)

        negative_vec = vectorize()(lambda a: np.negative(a))

        @njit
        def negative_jit_2(a, indices, b):
            return negative_vec.at(a, indices, b)

        # basic testing
        self._compare_output(add_at, np.add, a, [2, 5, 2], 1)

        # missing second operand
        err_msg = 'second operand needed for ufunc'
        with self.assertRaisesRegex(TypingError, err_msg):
            add_at(a.copy(), [2, 5, 3], None)

        self._compare_output(negative_at, np.negative, a.copy(), [2, 5, 3])

        b = np.array([100, 100, 100])
        self._compare_output(add_at, np.add, a.copy(), [2, 5, 2], b)

        # extraneous second operand
        err_msg = 'second operand provided when ufunc is unary'
        with self.assertRaisesRegex(TypingError, err_msg):
            negative_jit_2(a.copy(), [2, 5, 3], [1, 2, 3])

        with self.assertRaises(TypingError):
            add_at(a.copy(), [2, 5, 3], [[1, 2], 1])

    def test_ufunc_at_inner_loop(self):
        typecodes = np.typecodes['Complex']
        ufuncs = (np.add, np.subtract, np.multiply)
        for typecode in typecodes:

            try:
                from_dtype(np.dtype(typecode))
            except NumbaNotImplementedError:
                continue

            for ufunc in ufuncs:
                a = np.ones(10, dtype=typecode)
                indx = np.concatenate([np.ones(6, dtype=np.intp),
                                       np.full(18, 4, dtype=np.intp)])
                value = a.dtype.type(1j)
                ufunc_at = self._at(ufunc)
                ufunc_at(a, indx, value)
                expected = np.ones_like(a)
                if ufunc is np.multiply:
                    expected[1] = expected[4] = -1
                else:
                    expected[1] += 6 * (value if ufunc is np.add else -value)
                    expected[4] += 18 * (value if ufunc is np.add else -value)

        self.assertPreciseEqual(a, expected)

    def test_ufunc_at_ellipsis(self):
        # Make sure the indexed loop check does not choke on iters
        # with subspaces
        arr = np.zeros(5, dtype=int)
        add_at = self._at(np.add)
        self._compare_output(add_at, np.add, arr, slice(None),
                             np.ones(5, dtype=int))

    def test_ufunc_at_negative(self):
        arr = np.ones(5, dtype=np.int32)
        indx = np.arange(5)
        at = self._at(np.negative)
        at(arr, indx)
        assert np.all(arr == [-1, -1, -1, -1, -1])

    def test_ufunc_at_large(self):
        # NumPy issue gh-23457
        indices = np.zeros(8195, dtype=np.int16)
        b = np.zeros(8195, dtype=float)
        b[0] = 10
        b[1] = 5
        b[8192:] = 100
        a = np.zeros(1, dtype=float)
        add_at = self._at(np.add)
        add_at(a, indices, b)
        assert a[0] == b.sum()

    def test_cast_index_fastpath(self):
        arr = np.zeros(10)
        values = np.ones(100000)
        add_at = self._at(np.add)
        # index must be cast, which may be buffered in chunks:
        index = np.zeros(len(values), dtype=np.uint8)
        add_at(arr, index, values)
        assert arr[0] == len(values)

    def test_ufunc_at_scalar_value_fastpath(self):
        values = (np.ones(1), np.ones(()), np.float64(1.), 1.)
        for value in values:
            arr = np.zeros(1000)
            # index must be cast, which may be buffered in chunks:
            index = np.repeat(np.arange(1000), 2)
            add_at = self._at(np.add)
            add_at(arr, index, value)
            np.testing.assert_array_equal(arr, np.full_like(arr, 2 * value))

    def test_ufunc_at_multiD(self):
        a = np.arange(9).reshape(3, 3)
        b = np.array([[100, 100, 100], [200, 200, 200], [300, 300, 300]])
        add_at = self._at(np.add)
        add_at(a, (slice(None), np.asarray([1, 2, 1])), b)
        self.assertPreciseEqual(a, np.array(
            [[0, 201, 102], [3, 404, 205], [6, 607, 308]]))

        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        add_at(a, (slice(None), slice(None), np.asarray([1, 2, 1])), b)
        self.assertPreciseEqual(a, np.array(
                                [[[0, 401, 202],
                                  [3, 404, 205],
                                  [6, 407, 208]],

                                 [[9, 410, 211],
                                  [12, 413, 214],
                                  [15, 416, 217]],

                                 [[18, 419, 220],
                                  [21, 422, 223],
                                  [24, 425, 226]]]))

        a = np.arange(9).reshape(3, 3)
        b = np.array([[100, 100, 100], [200, 200, 200], [300, 300, 300]])
        add_at(a, (np.asarray([1, 2, 1]), slice(None)), b)
        self.assertPreciseEqual(a, np.asarray(
            [[0, 1, 2], [403, 404, 405], [206, 207, 208]]))

        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        add_at(a, (slice(None), np.asarray([1, 2, 1]), slice(None)), b)
        self.assertPreciseEqual(a, np.asarray(
                                [[[0,  1,  2],
                                  [203, 404, 605],
                                  [106, 207, 308]],

                                 [[9,  10, 11],
                                  [212, 413, 614],
                                  [115, 216, 317]],

                                 [[18, 19, 20],
                                  [221, 422, 623],
                                  [124, 225, 326]]]))

        a = np.arange(9).reshape(3, 3)
        b = np.array([100, 200, 300])
        add_at(a, (0, np.asarray([1, 2, 1])), b)
        self.assertPreciseEqual(a, np.asarray(
            [[0, 401, 202], [3, 4, 5], [6, 7, 8]]))

        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        add_at(a, (np.asarray([1, 2, 1]), 0, slice(None)), b)
        self.assertPreciseEqual(a, np.asarray(
                                [[[0,  1,  2],
                                  [3,  4,  5],
                                  [6,  7,  8]],

                                 [[209, 410, 611],
                                  [12,  13, 14],
                                  [15,  16, 17]],

                                 [[118, 219, 320],
                                  [21,  22, 23],
                                  [24,  25, 26]]]))

        a = np.arange(27).reshape(3, 3, 3)
        b = np.array([100, 200, 300])
        add_at = self._at(np.add)
        add_at(a, (slice(None), slice(None), slice(None)), b)
        self.assertPreciseEqual(a, np.asarray(
                                [[[100, 201, 302],
                                  [103, 204, 305],
                                  [106, 207, 308]],

                                 [[109, 210, 311],
                                  [112, 213, 314],
                                  [115, 216, 317]],

                                 [[118, 219, 320],
                                  [121, 222, 323],
                                  [124, 225, 326]]]))

    def test_ufunc_at_0D(self):
        a = np.array(0)
        add_at = self._at(np.add)
        add_at(a, (), 1)
        self.assertPreciseEqual(a, np.array(1))

        with self.assertRaises(TypingError):
            add_at(a, 0, 1)

        b = np.arange(3)
        add_at(b, 0, 1)
        self.assertPreciseEqual(b, np.array([1, 1, 2]))

        # NumPy checks for IndexError but we can't call a jit function with an
        # empty list as Numba raises "can't compute fingerprint of empty list"
        with self.assertRaises(ValueError):
            add_at(a, [], 1)

    def test_ufunc_at_dtypes(self):
        # Test mixed dtypes
        a = np.arange(10)
        power_at = self._at(np.power)
        power_at(a, [1, 2, 3, 2], 3.5)
        self.assertPreciseEqual(a, np.array([0, 1, 4414, 46, 4, 5, 6, 7, 8, 9]))

    def test_ufunc_at_boolean(self):
        # Test boolean indexing and boolean ufuncs
        a = np.arange(10)
        index = a % 2 == 0
        equal_at = self._at(np.equal)
        # boolean indexing not supported
        equal_at(a, index, [0, 2, 4, 6, 8])
        self.assertPreciseEqual(a, np.array([1, 1, 1, 3, 1, 5, 1, 7, 1, 9]))

    def test_ufunc_at_boolean2(self):
        # Test unary operator
        a = np.arange(10, dtype='u4')
        invert_at = self._at(np.invert)
        invert_at(a, [2, 5, 2])
        self.assertPreciseEqual(a, np.array([0, 1, 2, 3, 4, 5 ^ 0xffffffff, 6,
                                             7, 8, 9], dtype=np.uint32))

    def test_ufunc_at_advanced(self):
        # Test empty subspace
        orig = np.arange(4)
        a = orig[:, None][:, 0:0]
        add_at = self._at(np.add)
        add_at(a, [0, 1], 3)
        self.assertPreciseEqual(orig, np.arange(4))

    @unittest.expectedFailure
    def test_ufunc_at_advanced_2(self):
        # Test with swapped byte order
        index = np.array([1, 2, 1], np.dtype('i').newbyteorder())
        values = np.array([1, 2, 3, 4], np.dtype('f').newbyteorder())
        add_at = self._at(np.add)
        add_at(values, index, 3)
        self.assertPreciseEqual(values, [1, 8, 6, 4])

    def test_ufunc_at_advanced_3(self):
        # Test exception thrown
        values = np.array(['a', 1], dtype=object)
        add_at = self._at(np.add)
        with self.assertRaises(TypingError):
            add_at(values, [0, 1], 1)
        self.assertPreciseEqual(values, np.array(['a', 1], dtype=object))

    def test_ufunc_at_advanced_4(self):
        # Test multiple output ufuncs raise error, NumPy gh-5665
        modf_at = self._at(np.modf)
        # NumPy raises ValueError as modf returns multiple outputs
        with self.assertRaises(TypingError):
            modf_at(np.arange(10), [1])

    def test_ufunc_at_advanced_5(self):
        # Test maximum
        maximum_at = self._at(np.maximum)
        a = np.array([1, 2, 3])
        maximum_at(a, [0], 0)
        self.assertPreciseEqual(a, np.array([1, 2, 3]))

    def test_ufunc_at_negative_indexes(self):
        dtypes = np.typecodes['AllInteger'] + np.typecodes['Float']
        ufuncs = (np.add, np.subtract, np.divide, np.minimum, np.maximum)

        for dtype in dtypes:

            if dtype in ('e',):  # skip float16 as we don't have an impl. for it
                continue

            try:
                from_dtype(np.dtype(dtype))
            except NumbaNotImplementedError:
                continue

            for ufunc in ufuncs:
                a = np.arange(0, 10).astype(dtype)
                indxs = np.array([-1, 1, -1, 2]).astype(np.intp)
                vals = np.array([1, 5, 2, 10], dtype=a.dtype)

                expected = a.copy()
                for i, v in zip(indxs, vals):
                    expected[i] = ufunc(expected[i], v)

                ufunc_at = self._at(ufunc)
                ufunc_at(a, indxs, vals)
                np.testing.assert_array_equal(a, expected)
                assert np.all(indxs == [-1, 1, -1, 2])

    @unittest.expectedFailure
    def test_ufunc_at_not_none_signature(self):
        # Test ufuncs with non-trivial signature raise a TypeError
        a = np.ones((2, 2, 2))
        b = np.ones((1, 2, 2))
        # matmul is a gufunc, thus, this will fail atm
        matmul_at = self._at(np.matmul)
        err_msg = 'does not support ufunc with non-trivial signature'
        with self.assertRaisesRegex(TypingError, err_msg):
            matmul_at(a, [0], b)

        # a = np.array([[[1, 2], [3, 4]]])
        # assert_raises(TypeError, np.linalg._umath_linalg.det.at, a, [0])

    def test_ufunc_at_no_loop_for_op(self):
        # str dtype does not have a ufunc loop for np.add
        arr = np.ones(10, dtype=str)
        add_at = self._at(np.add)
        # NumPy raises `np.core._exceptions._UFuncNoLoopError`
        with self.assertRaises(TypingError):
            add_at(arr, [0, 1], [0, 1])

    def test_ufunc_at_output_casting(self):
        arr = np.array([-1])
        equal_at = self._at(np.equal)
        equal_at(arr, [0], [0])
        assert arr[0] == 0

    def test_ufunc_at_broadcast_failure(self):
        arr = np.arange(5)
        add_at = self._at(np.add)

        # NumPy raises ValueError('array is not broadcastable to correct shape')
        msg = 'operands could not be broadcast together with remapped shapes'
        with self.assertRaisesRegex(ValueError, msg):
            add_at(arr, [0, 1], [1, 2, 3])

    def test_ufunc_at_dynamic(self):
        arr = np.arange(5)

        @vectorize
        def inc(x):
            return x + 1

        self.assertEqual(len(inc.types), 0)

        # trying to call inc.at should trigger compilation
        inc.at(arr, [1, 3])

        self.assertGreater(len(inc.types), 0)

    def test_ufunc_at_experimental_warning(self):
        arr = np.arange(5)
        add_at = self._at(np.add)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaExperimentalFeatureWarning)

            add_at(arr, [0, 3], 10)

        self.assertGreater(len(w), 0)
        self.assertIn('ufunc.at feature is experimental', str(w[0].message))


class TestDUFuncReduce(TestDUFuncMethodsBase):
    def _check_reduce(self, ufunc, dtype=None, initial=None):

        @njit
        def foo(a, axis, dtype, initial):
            return ufunc.reduce(a,
                                axis=axis,
                                dtype=dtype,
                                initial=initial)

        inputs = [
            np.arange(5),
            np.arange(4).reshape(2, 2),
            np.arange(40).reshape(5, 4, 2),
        ]
        for array in inputs:
            for axis in range(array.ndim):
                expected = foo.py_func(array, axis, dtype, initial)
                got = foo(array, axis, dtype, initial)
                self.assertPreciseEqual(expected, got)

    def _check_reduce_axis(self, ufunc, dtype, initial=None):

        @njit
        def foo(a, axis):
            return ufunc.reduce(a, axis=axis, initial=initial)

        def _check(*args):
            try:
                expected = foo.py_func(array, axis)
            except ValueError as e:
                self.assertEqual(e.args[0], exc_msg)
                with self.assertRaisesRegex(TypingError, exc_msg):
                    got = foo(array, axis)
            else:
                got = foo(array, axis)
                self.assertPreciseEqual(expected, got)

        exc_msg = (f"reduction operation '{ufunc.__name__}' is not "
                   "reorderable, so at most one axis may be specified")
        inputs = [
            np.arange(40, dtype=dtype).reshape(5, 4, 2),
            np.arange(10, dtype=dtype),
        ]
        for array in inputs:
            for i in range(1, array.ndim + 1):
                for axis in itertools.combinations(range(array.ndim), r=i):
                    _check(array, axis)

            # corner cases: Reduce over axis=() and axis=None
            for axis in ((), None):
                _check(array, axis)

    def test_add_reduce(self):
        duadd = vectorize('int64(int64, int64)', identity=0)(pyuadd)
        self._check_reduce(duadd)
        self._check_reduce_axis(duadd, dtype=np.int64)

    def test_mul_reduce(self):
        dumul = vectorize('int64(int64, int64)', identity=1)(pymult)
        self._check_reduce(dumul)

    def test_non_associative_reduce(self):
        dusub = vectorize('int64(int64, int64)')(pysub)
        dudiv = vectorize('int64(int64, int64)')(pydiv)
        self._check_reduce(dusub)
        self._check_reduce_axis(dusub, dtype=np.int64)
        self._check_reduce(dudiv)
        self._check_reduce_axis(dudiv, dtype=np.int64)

    def test_reduce_dtype(self):
        duadd = vectorize('float64(float64, int64)', identity=0)(pyuadd)
        self._check_reduce(duadd, dtype=np.float64)

    def test_min_reduce(self):
        dumin = vectorize('int64(int64, int64)')(pymin)
        self._check_reduce(dumin, initial=10)
        self._check_reduce_axis(dumin, dtype=np.int64)

    def test_add_reduce_initial(self):
        # Initial should be used as a start
        duadd = vectorize('int64(int64, int64)', identity=0)(pyuadd)
        self._check_reduce(duadd, dtype=np.int64, initial=100)

    def test_add_reduce_no_initial_or_identity(self):
        # don't provide an initial or identity value
        duadd = vectorize('int64(int64, int64)')(pyuadd)
        self._check_reduce(duadd, dtype=np.int64)

    def test_invalid_input(self):
        duadd = vectorize('float64(float64, int64)', identity=0)(pyuadd)

        @njit
        def foo(a):
            return duadd.reduce(a)

        exc_msg = 'The first argument "array" must be array-like'
        with self.assertRaisesRegex(TypingError, exc_msg):
            foo('a')

    def test_dufunc_negative_axis(self):
        duadd = vectorize('int64(int64, int64)', identity=0)(pyuadd)

        @njit
        def foo(a, axis):
            return duadd.reduce(a, axis=axis)

        a = np.arange(40).reshape(5, 4, 2)
        cases = (0, -1, (0, -1), (-1, -2), (1, -1), -3)
        for axis in cases:
            expected = duadd.reduce(a, axis)
            got = foo(a, axis)
            self.assertPreciseEqual(expected, got)

    def test_dufunc_invalid_axis(self):
        duadd = vectorize('int64(int64, int64)', identity=0)(pyuadd)

        @njit
        def foo(a, axis):
            return duadd.reduce(a, axis=axis)

        a = np.arange(40).reshape(5, 4, 2)
        cases = ((0, 0), (0, 1, 0), (0, -3), (-1, -1), (-1, 2))
        for axis in cases:
            msg = "duplicate value in 'axis'"
            with self.assertRaisesRegex(ValueError, msg):
                foo(a, axis)

        cases = (-4, 3, (0, -4),)
        for axis in cases:
            with self.assertRaisesRegex(ValueError, "Invalid axis"):
                foo(a, axis)


class TestDUFuncReduceAt(TestDUFuncMethodsBase):
    def _compare_output(self, ufunc, a, idx, **kwargs):
        fn = self._reduceat(ufunc)
        expected = a.copy()
        got = a.copy()
        ufunc.reduceat(expected, idx, **kwargs)
        fn(got, idx, **kwargs)
        self.assertPreciseEqual(expected, got)

    def test_reduceat_out_kw(self):
        arr = np.arange(4)
        idx = np.asarray([0, 3, 1, 2])
        add_reduce = self._reduceat(np.add)
        add_reduce(arr, idx, out=arr)
        self.assertPreciseEqual(np.asarray([3, 3, 3, 6]), arr)
        add_reduce(arr, idx, out=arr)
        self.assertPreciseEqual(np.asarray([9, 6, 6, 12]), arr)

    @unittest.expectedFailure
    def test_reduceat_axis_kw(self):
        arrays = (
            np.arange(16).reshape(4, 4),
            np.arange(40).reshape(4, 5, 2),
            np.ones((4, 4))
        )
        indices = (
            np.asarray([0, 3, 1, 2, 0]),
            np.asarray([0, 3, 1, 2]),
        )
        axis = (1, 0, -1)  # needs gh-9296
        for array in arrays:
            for idx in indices:
                for ax in axis:
                    self._compare_output(np.add, array, idx, axis=ax)

    @unittest.expectedFailure
    def test_reduceat_invalid_axis(self):
        arr = np.ones((4, 4))
        idx = np.asarray([0, 3, 1, 2])
        add_reduceat = self._reduceat(np.add)

        for ax in (2, -2):  # needs gh-9296
            msg = (f'axis {ax} is out of bounds for array of dimension '
                   f'{arr.ndim}')
            with self.assertRaisesRegex(ValueError, msg):
                add_reduceat(arr, idx, ax)

    def test_reduceat_cast_args_to_array(self):
        add_reduceat = self._reduceat(np.add)

        # cast array and indices
        a = [1, 2, 3, 4]
        idx = [1, 2, 3]
        expected = np.add.reduceat(a, idx)
        got = add_reduceat(a, idx)
        self.assertPreciseEqual(expected, got)

        # array and indices as tuples
        a = (1, 2, 3, 4)
        idx = (1, 2, 3)
        expected = np.add.reduceat(a, idx)
        got = add_reduceat(a, idx)
        self.assertPreciseEqual(expected, got)

    # tests below this line were copied from NumPy
    # https://github.com/numpy/numpy/blob/7f8dc13b9bcaa26cb378b9c8246110ca1dc9ce75/numpy/_core/tests/test_ufunc.py#L200C1-L200C1  # noqa: E501
    def test_reduceat_basic(self):
        x = np.arange(8)
        idx = [0,4, 1,5, 2,6, 3,7]
        self._compare_output(np.add, x, idx)

    def test_reduceat_basic_2d(self):
        x = np.linspace(0, 15, 16).reshape(4, 4)
        idx = [0, 3, 1, 2, 0]
        self._compare_output(np.add, x, idx)

    def test_reduceat_shifting_sum(self):
        L = 6
        x = np.arange(L)
        idx = np.array(list(zip(np.arange(L - 2), np.arange(L - 2) + 2))).ravel()  # noqa: E501
        self._compare_output(np.add, x, idx)

    def test_reduceat_int_array_reduceat_inplace(self):
        # Checks that in-place reduceats work, see also gh-7465
        arr = np.ones(4, dtype=np.int64)
        out = np.ones(4, dtype=np.int64)
        self._compare_output(np.add, arr, np.arange(4), out=arr)
        self._compare_output(np.add, arr, np.arange(4), out=arr)
        self.assertPreciseEqual(arr, out)

        # needs gh-9296
        # And the same if the axis argument is used
        arr = np.ones((2, 4), dtype=np.int64)
        arr[0, :] = [2 for i in range(4)]
        out = np.ones((2, 4), dtype=np.int64)
        out[0, :] = [2 for i in range(4)]
        self._compare_output(np.add, arr, np.arange(4), out=arr, axis=-1)
        self._compare_output(np.add, arr, np.arange(4), out=arr, axis=-1)
        self.assertPreciseEqual(arr, out)

    def test_reduceat_out_shape_mismatch(self):
        # Should raise an error mentioning "shape" or "size"
        # original test has an extra step for accumulate but we don't support
        # it yet
        add_reduceat = self._reduceat(np.add)

        for with_cast in (True, False):
            arr = np.arange(5)
            out = np.arange(3)  # definitely wrong shape
            if with_cast:
                # If a cast is necessary on the output, we can be sure to use
                # the generic NpyIter (non-fast) path.
                out = out.astype(np.float64)

            with self.assertRaises(ValueError):
                add_reduceat(arr, [0, 3], out=out)

    def test_reduceat_empty(self):
        """Reduceat should work with empty arrays"""
        indices = np.array([], 'i4')
        x = np.array([], 'f8')
        add_reduceat = self._reduceat(np.add)
        expected = np.add.reduceat(x, indices)
        got = add_reduceat(x, indices)
        self.assertPreciseEqual(expected, got)
        self.assertEqual(expected.dtype, got.dtype)

        # Another case with a slightly different zero-sized shape
        x = np.ones((5, 2))
        idx = np.asarray([], dtype=np.intp)
        self._compare_output(np.add, x, idx, axis=0)
        self._compare_output(np.add, x, idx, axis=1)

    def test_reduceat_error_ndim_2(self):
        add_reduceat = self._reduceat(np.add)

        # indices ndim > 1
        a = np.arange(5)
        idx = np.arange(10).reshape(5, 2)
        with self.assertRaisesRegex(TypingError, 'have at most 1 dimension'):
            add_reduceat(a, idx)

    def test_reduceat_error_non_binary_function(self):
        # non binary functions
        @vectorize
        def neg(a):
            return np.negative(a)

        @njit
        def neg_reduceat(a, idx):
            return neg.reduceat(a, idx)

        a = np.arange(5)
        msg = 'reduceat only supported for binary functions'
        with self.assertRaisesRegex(TypingError, msg):
            neg_reduceat(a, [0, 1, 2])

    def test_reduceat_error_argument_types(self):
        # first argument must be array-like
        add_reduceat = self._reduceat(np.add)
        with self.assertRaisesRegex(TypingError, '"array" must be array-like'):
            add_reduceat('abc', [1, 2, 3])

        with self.assertRaisesRegex(TypingError, '"indices" must be array-like'):  # noqa: E501
            add_reduceat(np.arange(5), 'abcd')

        with self.assertRaisesRegex(TypingError, 'output must be an array'):
            add_reduceat(np.arange(5), [1, 2, 3], out=())

        with self.assertRaisesRegex(TypingError, '"axis" must be an integer'):
            add_reduceat(np.arange(5), [1, 2, 3], axis=(1,))


class TestDUFuncPickling(MemoryLeakMixin, unittest.TestCase):
    def check(self, ident, result_type):
        buf = pickle.dumps(ident)
        rebuilt = pickle.loads(buf)

        # Check reconstructed dufunc
        r = rebuilt(123)
        self.assertEqual(123, r)
        self.assertIsInstance(r, result_type)

        # Try to use reconstructed dufunc in @jit
        @njit
        def foo(x):
            return rebuilt(x)

        r = foo(321)
        self.assertEqual(321, r)
        self.assertIsInstance(r, result_type)

    def test_unrestricted(self):
        @vectorize
        def ident(x1):
            return x1

        self.check(ident, result_type=(int, np.integer))

    def test_restricted(self):
        @vectorize(["float64(float64)"])
        def ident(x1):
            return x1

        self.check(ident, result_type=float)


if __name__ == "__main__":
    unittest.main()
