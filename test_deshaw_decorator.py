import pytest
import warnings
import numpy as np
from numba.core.event import install_listener, Listener, EventStatus
from deshaw_jit import deshaw_jit, NumbaInterpreterModeWarning, de_events


class CustomListener(Listener):
    def __init__(self) -> None:
        self.triggered = False
        super().__init__()


class CompilerListener(CustomListener):
    def on_start(self, event):
        self.triggered = True
        assert event.status == EventStatus.START
        assert event.kind == de_events["jit"]

    def on_end(self, event):
        pass


class InterpreterListener(CustomListener):
    def on_start(self, event):
        self.triggered = True
        assert event.status == EventStatus.START
        assert event.kind == de_events["interpreter"]

    def on_end(self, event):
        pass


class ListenerNotTriggeredException(Exception):
    ...


def add(a, b):
    return a + b


def sum_fast(A):
    acc = 0.0
    # with fastmath, the reduction can be vectorized as floating point
    # reassociation is permitted.
    for x in A:
        acc += np.sqrt(x)
    return acc


def use_jit_sum_fast(A):
    # for small arrays, just interpret
    return len(A) > 1_000


def enable_jit(*args, **kwargs):
    return True


def disable_jit(*args, **kwargs):
    return False


def check_listener(kind, listener, cfunc, args):
    with install_listener(kind, listener):
        cfunc(*args)
    # ensure listener was triggered
    if listener.triggered == False:
        raise ListenerNotTriggeredException()


def check_interpreter_listener(cfunc, args):
    listener = InterpreterListener()
    return check_listener(de_events["interpreter"], listener, cfunc, args)


def check_compiler_listener(cfunc, args):
    listener = CompilerListener()
    return check_listener(de_events["jit"], listener, cfunc, args)


@pytest.mark.parametrize("use_jit", [True, False, enable_jit, disable_jit])
def test_use_jit(use_jit):
    cfunc = deshaw_jit(use_jit=use_jit)(add)
    if use_jit in (False, disable_jit):
        check_interpreter_listener(cfunc, (2, 3))
    else:
        check_compiler_listener(cfunc, (2, 3))


@pytest.mark.parametrize("A", [np.arange(10), np.arange(100_000)])
def test_use_jit_sum_fast(A):
    cfunc = deshaw_jit(use_jit=use_jit_sum_fast)(sum_fast)
    if len(A) == 10:
        check_interpreter_listener(cfunc, (A,))
    else:
        check_compiler_listener(cfunc, (A,))


def test_disable_jit():
    _jit = deshaw_jit("int64(int64, int64)", use_jit=disable_jit)
    cfunc = _jit(add)
    check_interpreter_listener(cfunc, ("Hello, ", "World"))

    # Compiling should fail
    with pytest.raises(ListenerNotTriggeredException):
        check_compiler_listener(cfunc, ("Hello, ", "World"))


@pytest.mark.parametrize("inp", [(2, 3), (2.2, 4.4)])
def test_disable_jit_with_signature(inp):
    _jit = deshaw_jit("float64(float64, float64)", use_jit=disable_jit)
    cfunc = _jit(add)
    check_compiler_listener(cfunc, inp)

    # Interpreter Listener should fail
    with pytest.raises(ListenerNotTriggeredException):
        check_interpreter_listener(cfunc, inp)


def test_invalid_call():
    cfunc = deshaw_jit("int64(int64, int64)", warn_on_fallback=True)(add)
    with pytest.raises(TypeError):
        cfunc("Hello", "world")


def test_no_warn_on_fallback():
    cfunc = deshaw_jit("float64(float64, float64)", warn_on_fallback=True)(add)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfunc(2, 3)
        cfunc(2.2, 3.3)


_options = [
    dict(nopython=True),
    dict(nopython=True, fastmath=True),
    dict(nopython=True, fastmath=True, use_jit=True),
    dict(nopython=True, fastmath=True, use_jit=False),
]


@pytest.mark.parametrize("options", _options)
def test_deshaw_jit_with_options(options):
    _jit = deshaw_jit(**options)
    cfunc = _jit(add)
    assert cfunc(2, 3) == 5
    assert cfunc(2.2, 3.3) == 5.5
    assert cfunc("hello", ", world") == "hello, world"


def test_invalid_use_jit_type():
    with pytest.raises(TypeError):
        deshaw_jit(use_jit="always")(add)
