from numba import njit
import numpy as np

# Based on https://github.com/numpy/numpy/blob/f702b26fff3271ba6a6ba29a021fc19051d1f007/numpy/core/src/multiarray/iterators.c#L1129-L1212  # noqa
@njit(no_cfunc_wrapper=True)
def broadcast_shapes(*args):
    # discover the number of dimensions
    return len(args)
    m = 0
    # for arg in args:
    #     m = max(m, len(arg))

    # propagate args
    r = [1] * m
    # for arg in args:
    #     sz = len(arg)
    #     for i, tmp in enumerate(arg):
    #         k = m - sz + i
    #         if tmp == 1:
    #             continue
    #         if r[k] == 1:
    #             r[k] = tmp
    #         elif r[k] != tmp:
    #             raise ValueError("shape mismatch: objects"
    #                              " cannot be broadcast"
    #                              " to a single shape")

    return r

@njit
def foo(*args):
    return broadcast_shapes(*args)

def test_broadcast_shapes_succeeds():
    # tests public broadcast_shapes
    data = [
        # [[], ()],
        # [[()], ()],
        [[(7,)], (7,)],
        # [[(1, 2), (2,)], (1, 2)],
        # [[(1, 1)], (1, 1)],
        # [[(1, 1), (3, 4)], (3, 4)],
        # [[(6, 7), (5, 6, 1), (7,), (5, 1, 7)], (5, 6, 7)],
        # [[(5, 6, 1)], (5, 6, 1)],
        # [[(1, 3), (3, 1)], (3, 3)],
        # [[(1, 0), (0, 0)], (0, 0)],
        # [[(0, 1), (0, 0)], (0, 0)],
        # [[(1, 0), (0, 1)], (0, 0)],
        # [[(1, 1), (0, 0)], (0, 0)],
        # [[(1, 1), (1, 0)], (1, 0)],
        # [[(1, 1), (0, 1)], (0, 1)],
        # [[(), (0,)], (0,)],
        # [[(0,), (0, 0)], (0, 0)],
        # [[(0,), (0, 1)], (0, 0)],
        # [[(1,), (0, 0)], (0, 0)],
        # [[(), (0, 0)], (0, 0)],
        # [[(1, 1), (0,)], (1, 0)],
        # [[(1,), (0, 1)], (0, 1)],
        # [[(1,), (1, 0)], (1, 0)],
        # [[(), (1, 0)], (1, 0)],
        # [[(), (0, 1)], (0, 1)],
        # [[(1,), (3,)], (3,)],
        # [[2, (3, 2)], (3, 2)],
    ]
    for input_shapes, target_shape in data:
        expected = target_shape
        result = broadcast_shapes(*input_shapes)
        print()
        print(input_shapes)
        print(expected, result, list(expected) == result)

test_broadcast_shapes_succeeds()
