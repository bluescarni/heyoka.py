# Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Version setup.
from ._version import __version__

# The top-level imports from the core module.
from ._core import (
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    cfunc_dbl,
    cfunc_f128,
    cfunc_flt,
    cfunc_ldbl,
    cfunc_real,
    code_model,
    continuous_output_batch_dbl,
    continuous_output_batch_flt,
    continuous_output_dbl,
    continuous_output_f128,
    continuous_output_flt,
    continuous_output_ldbl,
    continuous_output_real,
    cos,
    cosh,
    dfun,
    diff,
    diff_args,
    diff_tensors,
    dtens,
    eop_data,
    eop_data_row,
    eq,
    erf,
    event_direction,
    exp,
    expression,
    func_args,
    get_nthreads,
    get_params,
    get_variables,
    gt,
    gte,
    hamiltonian,
    install_custom_numpy_mem_handler,
    kepDE,
    kepE,
    kepF,
    lagrangian,
    leaky_relu,
    leaky_relup,
    llvm_multi_state,
    llvm_state,
    log,
    logical_and,
    logical_or,
    lt,
    lte,
    make_vars,
    neq,
    nt_event_batch_dbl,
    nt_event_batch_flt,
    nt_event_dbl,
    nt_event_f128,
    nt_event_flt,
    nt_event_ldbl,
    nt_event_real,
    pi,
    prod,
    real,
    real128,
    real_prec_max,
    real_prec_min,
    relu,
    relup,
    remove_custom_numpy_mem_handler,
    rename_variables,
    select,
    set_logger_level_critical,
    set_logger_level_debug,
    set_logger_level_error,
    set_logger_level_info,
    set_logger_level_trace,
    set_logger_level_warning,
    set_nthreads,
    sigmoid,
    sin,
    sinh,
    sqrt,
    subs,
    sum,
    sw_data,
    sw_data_row,
    t_event_batch_dbl,
    t_event_batch_flt,
    t_event_dbl,
    t_event_f128,
    t_event_flt,
    t_event_ldbl,
    t_event_real,
    tan,
    tanh,
    taylor_adaptive_batch_dbl,
    taylor_adaptive_batch_flt,
    taylor_adaptive_dbl,
    taylor_adaptive_f128,
    taylor_adaptive_flt,
    taylor_adaptive_ldbl,
    taylor_adaptive_real,
    taylor_outcome,
    to_sympy,
    var_args,
    var_ode_sys,
)

# Explicitly import the sub-packages
#
# NOTE: it is *important* that the import is performed here, *after* the initial import of core. Otherwise,
# we would get missing symbols on POSIX platforms.
from . import test, model, callback

from ._fp_suffixes import _fp_to_suffix
from ._sympy_utils import _with_sympy, _from_sympy_impl


def taylor_adaptive(sys, state=[], **kwargs):
    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return globals()[f"taylor_adaptive{fp_suffix}"](sys, state, **kwargs)


def taylor_adaptive_batch(sys, state, **kwargs):
    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return globals()[f"taylor_adaptive_batch{fp_suffix}"](sys, state, **kwargs)


def recommended_simd_size(fp_type=float):
    fp_suffix = _fp_to_suffix(fp_type)

    return globals()[f"_recommended_simd_size{fp_suffix}"]()


def cfunc(fn, vars, **kwargs):
    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(core, "cfunc{}".format(fp_suffix))(fn, vars, **kwargs)


def nt_event(ex, callback, **kwargs):
    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(core, "nt_event{}".format(fp_suffix))(ex, callback, **kwargs)


def t_event(ex, **kwargs):
    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(core, "t_event{}".format(fp_suffix))(ex, **kwargs)


def nt_event_batch(ex, callback, **kwargs):
    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(core, "nt_event_batch{}".format(fp_suffix))(ex, callback, **kwargs)


def t_event_batch(ex, **kwargs):
    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(core, "t_event_batch{}".format(fp_suffix))(ex, **kwargs)


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
            "The serialization backend '{}' is not valid. The valid backends are: {}".format(
                name, list(_s11n_backend_map.keys())
            )
        )

    new_backend = _s11n_backend_map[name]

    with _s11n_backend_mutex:
        _s11n_backend = new_backend


def get_serialization_backend():
    with _s11n_backend_mutex:
        return _s11n_backend


# Machinery to setup the custom SSL verify file.
def _setup_custom_verify_file():
    try:
        import certifi
    except ImportError:
        return

    from .core import _set_ssl_verify_file

    _set_ssl_verify_file(certifi.where())


_setup_custom_verify_file()


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
        "The parallelisation algorithm must be one of {}, but '{}' was provided instead".format(
            allowed_algos, algo
        )
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


time: expression = _create_time()
"""
Time expression.

This global object is an :py:class:`~heyoka.expression` which is used to represent
time (i.e., the independent variable) in righ-hand side of differential equations.

"""

eop_data_row: _dtype = eop_data_row
"""
EOP data row.

.. versionadded:: 7.3.0

This is a :ref:`structured NumPy datatype<numpy:defining-structured-types>` used to represent
a row of EOP data in the :py:class:`~heyoka.eop_data` class. The fields in the datatype are:

- the UTC MJD,
- the UT1-UTC difference (in seconds),
- the :math:`x` component of the polar motion (in arcsecs),
- the :math:`y` component of the polar motion (in arcsecs),
- the :math:`x` component of the correction to the IAU 2000/2006
  precession/nutation model (in milliarcsecs),
- the :math:`y` component of the correction to the IAU 2000/2006
  precession/nutation model (in milliarcsecs).

"""

sw_data_row: _dtype = sw_data_row
"""
Space weather data row.

.. versionadded:: 7.3.0

This is a :ref:`structured NumPy datatype<numpy:defining-structured-types>` used to represent
a row of space weather (SW) data in the :py:class:`~heyoka.sw_data` class. The fields in the datatype are:

- the UTC MJD,
- the arithmetic average of the 8 Ap indices for the day,
- the observed 10.7-cm solar radio flux (F10.7),
- the 81-day arithmetic average of the observed F10.7 centred on the day.

"""

__all__ = [
    # Imports from the core module.
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "cfunc_dbl",
    "cfunc_f128",
    "cfunc_flt",
    "cfunc_ldbl",
    "cfunc_real",
    "code_model",
    "continuous_output_batch_dbl",
    "continuous_output_batch_flt",
    "continuous_output_dbl",
    "continuous_output_f128",
    "continuous_output_flt",
    "continuous_output_ldbl",
    "continuous_output_real",
    "cos",
    "cosh",
    "dfun",
    "diff",
    "diff_args",
    "diff_tensors",
    "dtens",
    "eop_data",
    "eop_data_row",
    "eq",
    "erf",
    "event_direction",
    "exp",
    "expression",
    "func_args",
    "get_nthreads",
    "get_params",
    "get_variables",
    "gt",
    "gte",
    "hamiltonian",
    "install_custom_numpy_mem_handler",
    "kepDE",
    "kepE",
    "kepF",
    "lagrangian",
    "leaky_relu",
    "leaky_relup",
    "llvm_multi_state",
    "llvm_state",
    "log",
    "logical_and",
    "logical_or",
    "lt",
    "lte",
    "make_vars",
    "neq",
    "nt_event_batch_dbl",
    "nt_event_batch_flt",
    "nt_event_dbl",
    "nt_event_f128",
    "nt_event_flt",
    "nt_event_ldbl",
    "nt_event_real",
    "pi",
    "prod",
    "real",
    "real128",
    "real_prec_max",
    "real_prec_min",
    "relu",
    "relup",
    "remove_custom_numpy_mem_handler",
    "rename_variables",
    "select",
    "set_logger_level_critical",
    "set_logger_level_debug",
    "set_logger_level_error",
    "set_logger_level_info",
    "set_logger_level_trace",
    "set_logger_level_warning",
    "set_nthreads",
    "sigmoid",
    "sin",
    "sinh",
    "sqrt",
    "subs",
    "sum",
    "sw_data",
    "sw_data_row",
    "t_event_batch_dbl",
    "t_event_batch_flt",
    "t_event_dbl",
    "t_event_f128",
    "t_event_flt",
    "t_event_ldbl",
    "t_event_real",
    "tan",
    "tanh",
    "taylor_adaptive_batch_dbl",
    "taylor_adaptive_batch_flt",
    "taylor_adaptive_dbl",
    "taylor_adaptive_f128",
    "taylor_adaptive_flt",
    "taylor_adaptive_ldbl",
    "taylor_adaptive_real",
    "taylor_outcome",
    "to_sympy",
    "var_args",
    "var_ode_sys",
    # Sub-packages.
    "test",
    "model",
    "callback",
]
