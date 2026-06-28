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
from ._sympy_utils import from_sympy
from ._generic_wrappers import (
    taylor_adaptive,
    taylor_adaptive_batch,
    recommended_simd_size,
    cfunc,
    nt_event,
    t_event,
    nt_event_batch,
    t_event_batch,
)
from ._s11n import get_serialization_backend, set_serialization_backend
from ._ensemble_impl import (
    ensemble_propagate_until,
    ensemble_propagate_for,
    ensemble_propagate_grid,
    ensemble_propagate_until_batch,
    ensemble_propagate_for_batch,
    ensemble_propagate_grid_batch,
)
from . import _core
from numpy import dtype as _dtype


# Machinery to setup the custom SSL verify file.
def _setup_custom_verify_file():
    try:
        import certifi
    except ImportError:
        return

    from ._core import _set_ssl_verify_file

    _set_ssl_verify_file(certifi.where())


_setup_custom_verify_file()

# NOTE: these global attributes need to be defined directly in this file - if we
# define them in a separate file and then import them, sphinx documentation is not
# properly built.

par = _core._par
"""
Parameter factory.

This global object is used to create :py:class:`~heyoka.expression` objects
representing :ref:`runtime parameters <runtime_param>`. The parameter index
must be passed to the index operator of the factory object.

Examples:
  >>> from heyoka import par
  >>> p0 = par[0] # p0 will represent the parameter value at index 0

"""

time: _core.expression = _core._time
"""
Time expression.

This global object is an :py:class:`~heyoka.expression` which is used to represent
time (i.e., the independent variable) in right-hand side of differential equations.

"""

eop_data_row: _dtype = _core.eop_data_row
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

sw_data_row: _dtype = _core.sw_data_row
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
    "__version__",
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
    # Globals.
    "par",
    "time",
    "eop_data_row",
    "sw_data_row",
    # Sympy utils.
    "from_sympy",
    # Generic wrappers.
    "taylor_adaptive",
    "taylor_adaptive_batch",
    "recommended_simd_size",
    "cfunc",
    "nt_event",
    "t_event",
    "nt_event_batch",
    "t_event_batch",
    # Serialization.
    "get_serialization_backend",
    "set_serialization_backend",
    # Ensemble propagations.
    "ensemble_propagate_until",
    "ensemble_propagate_for",
    "ensemble_propagate_grid",
    "ensemble_propagate_until_batch",
    "ensemble_propagate_for_batch",
    "ensemble_propagate_grid_batch",
]
