# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Version setup.
from ._version import __version__

import cloudpickle as _cloudpickle
from threading import Lock as _Lock

# We import the sub-modules into the root namespace.
from .core import *

# Explicitly import the submodules
# NOTE: it is *important* that the import is performed
# here, *after* the initial import of core. Otherwise,
# we would get missing symbols on POSIX platforms.
from . import test, model, callback


def _with_real128():
    # Small helper to check if real128 is available.
    from . import core

    return hasattr(core, "real128")


def _with_real():
    # Small helper to check if real is available.
    from . import core

    return hasattr(core, "real")


from numpy import float32 as _f32, float64 as _f64, longdouble as _ld

_fp_to_suffix_dict = {_f32: "_flt", _f64: "_dbl", float: "_dbl", _ld: "_ldbl"}

del _f32
del _f64
del _ld

if _with_real128():
    _fp_to_suffix_dict[real128] = "_f128"

if _with_real():
    _fp_to_suffix_dict[real] = "_real"


def _fp_to_suffix(fp_t):
    if not isinstance(fp_t, type):
        raise TypeError(
            'A Python type was expected in input, but an object of type "{}" was'
            " provided instead".format(type(fp_t))
        )

    if fp_t in _fp_to_suffix_dict:
        return _fp_to_suffix_dict[fp_t]

    raise TypeError(
        'The floating-point type "{}" is not recognized/supported'.format(fp_t)
    )


def taylor_adaptive(sys, state=[], **kwargs):
    from . import core

    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(core, "taylor_adaptive{}".format(fp_suffix))(sys, state, **kwargs)


def taylor_adaptive_batch(sys, state, **kwargs):
    from . import core

    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(core, "taylor_adaptive_batch{}".format(fp_suffix))(
        sys, state, **kwargs
    )


def recommended_simd_size(fp_type=float):
    from . import core

    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(core, "_recommended_simd_size{}".format(fp_suffix))()


def cfunc(fn, vars, **kwargs):
    from . import core

    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(core, "cfunc{}".format(fp_suffix))(fn, vars, **kwargs)


def nt_event(ex, callback, **kwargs):
    from . import core

    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(core, "nt_event{}".format(fp_suffix))(ex, callback, **kwargs)


def t_event(ex, **kwargs):
    from . import core

    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(core, "t_event{}".format(fp_suffix))(ex, **kwargs)


def nt_event_batch(ex, callback, **kwargs):
    from . import core

    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(core, "nt_event_batch{}".format(fp_suffix))(ex, callback, **kwargs)


def t_event_batch(ex, **kwargs):
    from . import core

    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(core, "t_event_batch{}".format(fp_suffix))(ex, **kwargs)


def from_sympy(ex, s_dict={}):
    from ._sympy_utils import _with_sympy, _from_sympy_impl

    if not _with_sympy:
        raise ImportError(
            "The 'from_sympy()' function is not available because sympy is not"
            " installed"
        )

    from sympy import Basic
    from .core import expression

    if not isinstance(ex, Basic):
        raise TypeError(
            "The 'ex' parameter must be a sympy expression but it is of type {} instead"
            .format(type(ex))
        )

    if not isinstance(s_dict, dict):
        raise TypeError(
            "The 's_dict' parameter must be a dict but it is of type {} instead".format(
                type(s_dict)
            )
        )

    if any(not isinstance(_, Basic) for _ in s_dict):
        raise TypeError("The keys in 's_dict' must all be sympy expressions")

    if any(not isinstance(s_dict[_], expression) for _ in s_dict):
        raise TypeError("The values in 's_dict' must all be heyoka expressions")

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
            "The serialization backend must be specified as a string, but an object of"
            " type {} was provided instead".format(type(name))
        )

    if not name in _s11n_backend_map:
        raise ValueError(
            "The serialization backend '{}' is not valid. The valid backends are: {}"
            .format(name, list(_s11n_backend_map.keys()))
        )

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
            "The n_iter parameter must be an integer, but an object of type {} was"
            " provided instead".format(type(n_iter))
        )

    if n_iter < 0:
        raise ValueError(
            "The n_iter parameter must be non-negative, but it is {} instead".format(
                n_iter
            )
        )

    # Validate arg and max_delta_t, if present.
    def is_iterable(x):
        from collections.abc import Iterable

        return isinstance(x, Iterable)

    if tp == "until" or tp == "for":
        if is_iterable(arg):
            raise TypeError(
                "Cannot perform an ensemble propagate_until/for(): the final epoch/time"
                " interval must be a scalar, not an iterable object"
            )
    else:
        arg = np.array(arg)

        if arg.ndim != 1:
            raise ValueError(
                "Cannot perform an ensemble propagate_grid(): the input time grid must"
                " be one-dimensional, but instead it has {} dimensions".format(arg.ndim)
            )

    if "max_delta_t" in kwargs and is_iterable(kwargs["max_delta_t"]):
        raise TypeError(
            'Cannot perform an ensemble propagate_until/for/grid(): the "max_delta_t"'
            " argument must be a scalar, not an iterable object"
        )

    # Parallelisation algorithm.
    algo = kwargs.pop("algorithm", "thread")
    allowed_algos = ["thread", "process"]

    if algo == "thread":
        from ._ensemble_impl import _ensemble_propagate_thread

        return _ensemble_propagate_thread(tp, ta, arg, n_iter, gen, **kwargs)

    if algo == "process":
        from ._ensemble_impl import _ensemble_propagate_process

        return _ensemble_propagate_process(tp, ta, arg, n_iter, gen, **kwargs)

    raise ValueError(
        "The parallelisation algorithm must be one of {}, but '{}' was provided instead"
        .format(allowed_algos, algo)
    )


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


def _real_reduce_factory():
    # Internal factory function used in the implementation
    # of the pickle protocol for real.
    return real()


# Machinery for the par generator.
def _create_par():
    from . import core

    return core._par_generator()


par = _create_par()
"""
Parameter factory.

This global object is used to create :py:class:`~heyoka.expression` objects
representing :ref:`runtime parameters <runtime_param>`. The parameter index
must be passed to the index operator of the factory object.

Examples:
  >>> from heyoka import par
  >>> p0 = par[0] # p0 will represent the parameter value at index 0

"""


# Machinery for the time attribute.
def _create_time():
    from . import core

    return core._time


time = _create_time()
"""
Time placeholder.

This global object is an :py:class:`~heyoka.expression` which is used to represent
time (i.e., the independent variable) in differential equations.

"""
