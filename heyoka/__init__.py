# Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Version setup.
from ._version import __version__

import os as _os

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
    import ctypes as _ctypes, sys as _sys
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

# Explicitly import the test submodule
from . import test

def taylor_adaptive(sys, state, **kwargs):
    fp_type = kwargs.pop("fp_type", "double")

    if fp_type == "double":
        return taylor_adaptive_dbl(sys, state, **kwargs)

    if fp_type == "long double":
        return taylor_adaptive_ldbl(sys, state, **kwargs)

    if with_real128 and fp_type == "real128":
        return taylor_adaptive_f128(sys, state, **kwargs)

    raise TypeError("the floating-point type \"{}\" is not recognized/supported".format(fp_type))

def taylor_adaptive_batch(sys, state, **kwargs):
    fp_type = kwargs.pop("fp_type", "double")

    if fp_type == "double":
        return taylor_adaptive_batch_double(sys, state, **kwargs)

    raise TypeError("the floating-point type \"{}\" is not recognized/supported".format(fp_type))

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

    raise TypeError("the floating-point type \"{}\" is not recognized/supported".format(fp_type))
