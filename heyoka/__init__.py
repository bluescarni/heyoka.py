# Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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


def taylor_adaptive(sys, state, **kwargs):
    from .core import _taylor_adaptive_dbl, _taylor_adaptive_ldbl

    fp_type = kwargs.pop("fp_type", "double")

    if fp_type == "double":
        return _taylor_adaptive_dbl(sys, state, **kwargs)

    if fp_type == "long double":
        return _taylor_adaptive_ldbl(sys, state, **kwargs)

    if with_real128 and fp_type == "real128":
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

    if with_real128 and fp_type == "real128":
        from .core import _eval_f128
        return _eval_f128(e, map, pars, **kwargs)

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

    if with_real128 and fp_type == "real128":
        from .core import _taylor_add_jet_f128
        return _taylor_add_jet_f128(sys, order, **kwargs)

    raise TypeError(
        "the floating-point type \"{}\" is not recognized/supported".format(fp_type))


def nt_event(ex, callback, **kwargs):
    from .core import _nt_event_dbl, _nt_event_ldbl

    fp_type = kwargs.pop("fp_type", "double")

    if fp_type == "double":
        return _nt_event_dbl(ex, callback, **kwargs)

    if fp_type == "long double":
        return _nt_event_ldbl(ex, callback, **kwargs)

    if with_real128 and fp_type == "real128":
        from .core import _nt_event_f128
        return _nt_event_f128(ex, callback, **kwargs)


def t_event(ex, **kwargs):
    from .core import _t_event_dbl, _t_event_ldbl

    fp_type = kwargs.pop("fp_type", "double")

    if fp_type == "double":
        return _t_event_dbl(ex, **kwargs)

    if fp_type == "long double":
        return _t_event_ldbl(ex, **kwargs)

    if with_real128 and fp_type == "real128":
        from .core import _t_event_f128
        return _t_event_f128(ex, **kwargs)


def from_sympy(ex):
    from ._sympy_utils import _with_sympy, _from_sympy_impl

    if not _with_sympy:
        raise ImportError(
            "The 'from_sympy()' function is not available because sympy is not installed")

    return _from_sympy_impl(ex)


# Machinery for the setup of the serialization backend.
_serialization_backend = _cloudpickle


def set_serialization_backend(name):
    if not isinstance(name, str):
        raise TypeError(
            "The serialization backend must be specified as a string, but an object of type {} was provided instead".format(type(name)))
    global _serialization_backend
    if name == "pickle":
        import pickle
        _serialization_backend = pickle
    elif name == "cloudpickle":
        _serialization_backend = _cloudpickle
    elif name == "dill":
        try:
            import dill
            _serialization_backend = dill
        except ImportError:
            raise ImportError(
                "The 'dill' serialization backend was specified, but the dill module is not installed.")
    else:
        raise ValueError(
            "The serialization backend '{}' is not valid. The valid backends are: ['pickle', 'cloudpickle', 'dill']".format(name))


def get_serialization_backend():
    return _serialization_backend
