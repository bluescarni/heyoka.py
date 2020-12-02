# Copyright 2020 Francesco Biscani (bluescarni@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Version setup.
from ._version import __version__
# We import the sub-modules into the root namespace
from .core import *

def taylor_adaptive(sys, state, **kwargs):
    fp_type = kwargs.pop("fp_type", "double")

    if fp_type == "double":
        return taylor_adaptive_double(sys, state, **kwargs)

    if fp_type == "long double":
        return taylor_adaptive_long_double(sys, state, **kwargs)

    if with_real128 and fp_type == "real128":
        return taylor_adaptive_real128(sys, state, **kwargs)

    raise TypeError("the floating-point type \"{}\" is not recognized/supported".format(fp_type))
