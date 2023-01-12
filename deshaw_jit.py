from typing import Callable
from numba.core.target_extension import CPU, dispatcher_registry, target_registry
from numba.core.registry import CPUDispatcher
from numba.core import errors, event
from numba import jit


# 1. For jitted functions with a cache we can use a jitted function
# if it's in the cache and an interpreted function otherwise.
#
# 2. We could add a "dispatching" logic, an optional function to pass to the
# numba.jit decorator which will decide whether to use the jit or not.
# We currently do this manually in a number of places.

de_events = dict(jit="jit_execution", interpreter="interpreter_execution")


class NumbaInterpreterModeWarning(errors.NumbaWarning):
    """
    Emit a warning in case jit falls back into interpreter mode
    """


class DEShawDispatcher(CPUDispatcher):
    def _default_checker(*args, **kwargs):
        return True

    def __init__(self, *args, targetoptions, **kwargs):
        """
        use_jit: Callable or bool
            Custom logic passed to the dispatcher object which will decide
            whether to use the jit or not. Default is to always use jit.

        warn_on_fallback: bool
            Set to True to warn when jit compilation/execution falls back
            to interpreter mode. Default value is False
        """
        self.use_jit = targetoptions.pop("use_jit", DEShawDispatcher._default_checker)
        self.warn_on_fallback = targetoptions.pop("warn_on_fallback", False)
        super().__init__(*args, targetoptions=targetoptions, **kwargs)

    def _emit_fallback_warning(self):
        import warnings

        func_name = self.py_func.__name__
        msg = f"{func_name} not using JIT"
        if self.warn_on_fallback:
            warnings.warn(msg, NumbaInterpreterModeWarning)

    def _function_in_cache(self, *args, **kwargs):
        # partially copied from dispatcher.py::explain_ambiguous
        args = tuple([self.typeof_pyval(a) for a in args])
        sigs = self.nopython_signatures
        assert not kwargs, "kwargs not handled"
        func = self.typingctx.resolve_overload(
            self.py_func, sigs, args, kwargs, unsafe_casting=False
        )
        return True if func else False

    def can_compile(self):
        return self._can_compile

    def _fallback_interpreter(self, *args, **kwargs):
        # fallback to interpreter if cannot use jit or compilation is
        # disabled
        event.start_event(
            de_events["interpreter"], data={"py_func": self.py_func, "args": args}
        )
        self._emit_fallback_warning()
        ret = self.py_func(*args, **kwargs)
        event.end_event(
            de_events["interpreter"], data={"py_func": self.py_func, "args": args}
        )
        return ret

    def _run_jit_func(self, *args, **kwargs):
        # function in cache
        event.start_event(
            de_events["jit"], data={"py_func": self.py_func, "args": args}
        )
        ret = super().__call__(*args, **kwargs)
        event.end_event(de_events["jit"], data={"py_func": self.py_func, "args": args})
        return ret

    def __call__(self, *args, **kwargs):
        if self._function_in_cache(*args, **kwargs):
            return self._run_jit_func(*args, **kwargs)

        # Run use_jit function
        use_jit = self.use_jit(*args, **kwargs)

        if use_jit and not self.can_compile():
            self._explain_matching_error(*args, **kwargs)

        if not use_jit:
            return self._fallback_interpreter(*args, **kwargs)

        return self._run_jit_func(*args, **kwargs)


class DEShawJIT(CPU):
    ...


target_registry["DEShawJIT"] = DEShawJIT


dispatcher_registry[target_registry["DEShawJIT"]] = DEShawDispatcher


def deshaw_jit(
    *args, use_jit=DEShawDispatcher._default_checker, warn_on_fallback=False, **kws
):
    """
    This decorator is used to compile a Python function into native code.
    custom options:
        use_jit: Callable
            Custom logic passed to the dispatcher object which will decide
            whether to use the jit or not. Default is to always use jit.

        warn_on_fallback: Callable or bool
            Set to True to warn when jit compilation/execution falls back
            to interpreter mode. Default value is False

    """
    if not isinstance(use_jit, (bool, Callable)):
        msg = f"'use_jit' must be a boolean or a Callable. Got {type(use_jit)}"
        raise TypeError(msg)

    if isinstance(use_jit, bool):
        kws["use_jit"] = lambda *a, **kw: use_jit
    else:
        kws["use_jit"] = use_jit

    kws["warn_on_fallback"] = warn_on_fallback
    return jit(
        *args,
        **kws,
        _target="DEShawJIT",
    )


# from numba.core import event


# class CustomListener(event.Listener):
#     def on_start(self, event):
#         print(f'[START] {event.data["py_func"].__name__} {event.kind}...')

#     def on_end(self, event):
#         print(f'[END] {event.data["py_func"].__name__} {event.kind}')


# def int_jit(a):
#     return isinstance(a, int)


# @deshaw_jit(use_jit=int_jit)
# def incr(a):
#     return a + 1


# listener = CustomListener()
# with event.install_listener("jit_execution", listener):
#     incr(1)

# listener = CustomListener()
# with event.install_listener("interpreter_execution", listener):
#     incr(1.23)
