# Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Explicitly import the test submodule
from . import test
# Version setup.
from ._version import __version__

import os as _os
import cloudpickle as _cloudpickle
from threading import Lock as _Lock

if _os.name == 'posix':
    # NOTE: on some platforms Python by default opens extensions
    # with the RTLD_LOCAL flag, which creates problems because
    # public symbols used by heyoka (e.g., sleef functions, quad
    # precision math) are then not found by the LLVM jit machinery.
    # Thus, before importing core, we temporarily flip on the
    # RTLD_GLOBAL flag, which makes the symbols visible and
    # solves these issues. Another possible approach suggested
    # in the llvm discord is to manually and explicitly add
    # libheyoka.so to the DL search path:
    # DynamicLibrarySearchGenerator::Load(“/path/to/libheyoka.so”)
    # See:
    # https://docs.python.org/3/library/ctypes.html
    import ctypes as _ctypes
    import sys as _sys
    _orig_dlopen_flags = _sys.getdlopenflags()
    _sys.setdlopenflags(_orig_dlopen_flags | _ctypes.RTLD_GLOBAL)

    try:
        # We import the sub-modules into the root namespace.
        from .core import *
    finally:
        # Restore the original dlopen flags whatever
        # happens.
        _sys.setdlopenflags(_orig_dlopen_flags)

        del _ctypes
        del _sys
        del _orig_dlopen_flags
else:
    # We import the sub-modules into the root namespace.
    from .core import *

del _os

# Small helper to check if real128 is available.


def _with_real128():
    from . import core

    return hasattr(core, "real128")


def taylor_adaptive(sys, state, **kwargs):
    from .core import _taylor_adaptive_dbl, _taylor_adaptive_ldbl

    fp_type = kwargs.pop("fp_type", "double")

    if fp_type == "double":
        return _taylor_adaptive_dbl(sys, state, **kwargs)

    if fp_type == "long double":
        return _taylor_adaptive_ldbl(sys, state, **kwargs)

    if _with_real128() and fp_type == "real128":
        from .core import _taylor_adaptive_f128
        return _taylor_adaptive_f128(sys, state, **kwargs)

    raise TypeError(
        "the floating-point type \"{}\" is not recognized/supported".format(fp_type))


def eval(e, map, pars=[], **kwargs):
    from .core import _eval_dbl, _eval_ldbl

    fp_type = kwargs.pop("fp_type", "double")

    if fp_type == "double":
        return _eval_dbl(e, map, pars, **kwargs)

    if fp_type == "long double":
        return _eval_ldbl(e, map, pars, **kwargs)

    if _with_real128() and fp_type == "real128":
        from .core import _eval_f128
        return _eval_f128(e, map, pars, **kwargs)

    raise TypeError(
        "the floating-point type \"{}\" is not recognized/supported".format(fp_type))


def recommended_simd_size(fp_type="double"):
    from .core import _recommended_simd_size_dbl

    if fp_type == "double":
        return _recommended_simd_size_dbl()

    raise TypeError(
        "the floating-point type \"{}\" is not recognized/supported".format(fp_type))


def taylor_adaptive_batch(sys, state, **kwargs):
    from .core import _taylor_adaptive_batch_dbl

    fp_type = kwargs.pop("fp_type", "double")

    if fp_type == "double":
        return _taylor_adaptive_batch_dbl(sys, state, **kwargs)

    raise TypeError(
        "the floating-point type \"{}\" is not recognized/supported".format(fp_type))


def taylor_add_jet(sys, order, **kwargs):
    from .core import _taylor_add_jet_dbl, _taylor_add_jet_ldbl

    fp_type = kwargs.pop("fp_type", "double")

    if fp_type == "double":
        return _taylor_add_jet_dbl(sys, order, **kwargs)

    if fp_type == "long double":
        return _taylor_add_jet_ldbl(sys, order, **kwargs)

    if _with_real128() and fp_type == "real128":
        from .core import _taylor_add_jet_f128
        return _taylor_add_jet_f128(sys, order, **kwargs)

    raise TypeError(
        "the floating-point type \"{}\" is not recognized/supported".format(fp_type))


def add_cfunc(fn, **kwargs):
    from .core import _add_cfunc_dbl, _add_cfunc_ldbl

    fp_type = kwargs.pop("fp_type", "double")

    if fp_type == "double":
        return _add_cfunc_dbl(fn, **kwargs)

    if fp_type == "long double":
        return _add_cfunc_ldbl(fn, **kwargs)

    if _with_real128() and fp_type == "real128":
        from .core import _add_cfunc_f128
        return _add_cfunc_f128(fn, **kwargs)

    raise TypeError(
        "the floating-point type \"{}\" is not recognized/supported".format(fp_type))


def nt_event(ex, callback, **kwargs):
    from .core import _nt_event_dbl, _nt_event_ldbl

    fp_type = kwargs.pop("fp_type", "double")

    if fp_type == "double":
        return _nt_event_dbl(ex, callback, **kwargs)

    if fp_type == "long double":
        return _nt_event_ldbl(ex, callback, **kwargs)

    if _with_real128() and fp_type == "real128":
        from .core import _nt_event_f128
        return _nt_event_f128(ex, callback, **kwargs)

    raise TypeError(
        "the floating-point type \"{}\" is not recognized/supported".format(fp_type))


def t_event(ex, **kwargs):
    from .core import _t_event_dbl, _t_event_ldbl

    fp_type = kwargs.pop("fp_type", "double")

    if fp_type == "double":
        return _t_event_dbl(ex, **kwargs)

    if fp_type == "long double":
        return _t_event_ldbl(ex, **kwargs)

    if _with_real128() and fp_type == "real128":
        from .core import _t_event_f128
        return _t_event_f128(ex, **kwargs)

    raise TypeError(
        "the floating-point type \"{}\" is not recognized/supported".format(fp_type))


def nt_event_batch(ex, callback, **kwargs):
    from .core import _nt_event_batch_dbl

    fp_type = kwargs.pop("fp_type", "double")

    if fp_type == "double":
        return _nt_event_batch_dbl(ex, callback, **kwargs)

    raise TypeError(
        "the floating-point type \"{}\" is not recognized/supported".format(fp_type))


def t_event_batch(ex, **kwargs):
    from .core import _t_event_batch_dbl

    fp_type = kwargs.pop("fp_type", "double")

    if fp_type == "double":
        return _t_event_batch_dbl(ex, **kwargs)

    raise TypeError(
        "the floating-point type \"{}\" is not recognized/supported".format(fp_type))


def from_sympy(ex, s_dict={}):
    from ._sympy_utils import _with_sympy, _from_sympy_impl

    if not _with_sympy:
        raise ImportError(
            "The 'from_sympy()' function is not available because sympy is not installed")

    from sympy import Basic
    from .core import expression

    if not isinstance(ex, Basic):
        raise TypeError(
            "The 'ex' parameter must be a sympy expression but it is of type {} instead".format(type(ex)))

    if not isinstance(s_dict, dict):
        raise TypeError(
            "The 's_dict' parameter must be a dict but it is of type {} instead".format(type(s_dict)))

    if any(not isinstance(_, Basic) for _ in s_dict):
        raise TypeError("The keys in 's_dict' must all be sympy expressions")

    if any(not isinstance(s_dict[_], expression) for _ in s_dict):
        raise TypeError(
            "The values in 's_dict' must all be heyoka expressions")

    return _from_sympy_impl(ex, s_dict, {})


# Machinery for the setup of the serialization backend.

# Helper to create dicts mapping a name to a serialization backend
# and vice-versa.
def _make_s11n_backend_maps():
    import pickle

    ret = {"cloudpickle": _cloudpickle, "pickle": pickle}

    try:
        import dill
        ret["dill"] = dill
    except ImportError:
        pass

    inv = dict([(ret[_], _) for _ in ret])

    return ret, inv


_s11n_backend_map, _s11n_backend_inv_map = _make_s11n_backend_maps()

# The currently active s11n backend.
_s11n_backend = _cloudpickle

# Lock to protect access to _s11n_backend.
_s11n_backend_mutex = _Lock()


def set_serialization_backend(name):
    global _s11n_backend

    if not isinstance(name, str):
        raise TypeError(
            "The serialization backend must be specified as a string, but an object of type {} was provided instead".format(type(name)))

    if not name in _s11n_backend_map:
        raise ValueError(
            "The serialization backend '{}' is not valid. The valid backends are: {}".format(name, list(_s11n_backend_map.keys())))

    new_backend = _s11n_backend_map[name]

    with _s11n_backend_mutex:
        _s11n_backend = new_backend


def get_serialization_backend():
    with _s11n_backend_mutex:
        return _s11n_backend


# Ensemble propagations.
def _ensemble_propagate_generic(tp, ta, arg, n_iter, gen, **kwargs):
    import numpy as np

    if not isinstance(n_iter, int):
        raise TypeError(
            "The n_iter parameter must be an integer, but an object of type {} was provided instead".format(type(n_iter)))

    if n_iter < 0:
        raise ValueError(
            "The n_iter parameter must be non-negative, but it is {} instead".format(n_iter))

    # Validate arg and max_delta_t, if present.
    def is_iterable(x):
        from collections.abc import Iterable

        return isinstance(x, Iterable)

    if tp == "until" or tp == "for":
        if is_iterable(arg):
            raise TypeError(
                "Cannot perform an ensemble propagate_until/for(): the final epoch/time interval must be a scalar, not an iterable object")
    else:
        arg = np.array(arg)

        if arg.ndim != 1:
            raise ValueError(
                "Cannot perform an ensemble propagate_grid(): the input time grid must be one-dimensional, but instead it has {} dimensions".format(arg.ndim))

    if "max_delta_t" in kwargs and is_iterable(kwargs["max_delta_t"]):
        raise TypeError(
            "Cannot perform an ensemble propagate_until/for/grid(): the \"max_delta_t\" argument must be a scalar, not an iterable object")

    # Parallelisation algorithm.
    algo = kwargs.pop("algorithm", "thread")
    allowed_algos = ["thread", "process"]

    if algo == "thread":
        from ._ensemble_impl import _ensemble_propagate_thread
        return _ensemble_propagate_thread(tp, ta, arg, n_iter, gen, **kwargs)

    if algo == "process":
        from ._ensemble_impl import _ensemble_propagate_process
        return _ensemble_propagate_process(tp, ta, arg, n_iter, gen, **kwargs)

    raise ValueError("The parallelisation algorithm must be one of {}, but '{}' was provided instead".format(
        allowed_algos, algo))


def ensemble_propagate_until(ta, t, n_iter, gen, **kwargs):
    return _ensemble_propagate_generic("until", ta, t, n_iter, gen, **kwargs)


def ensemble_propagate_for(ta, delta_t, n_iter, gen, **kwargs):
    return _ensemble_propagate_generic("for", ta, delta_t, n_iter, gen, **kwargs)


def ensemble_propagate_grid(ta, grid, n_iter, gen, **kwargs):
    return _ensemble_propagate_generic("grid", ta, grid, n_iter, gen, **kwargs)


def ensemble_propagate_until_batch(ta, t, n_iter, gen, **kwargs):
    return _ensemble_propagate_generic("until", ta, t, n_iter, gen, **kwargs)


def ensemble_propagate_for_batch(ta, delta_t, n_iter, gen, **kwargs):
    return _ensemble_propagate_generic("for", ta, delta_t, n_iter, gen, **kwargs)


def ensemble_propagate_grid_batch(ta, grid, n_iter, gen, **kwargs):
    return _ensemble_propagate_generic("grid", ta, grid, n_iter, gen, **kwargs)
