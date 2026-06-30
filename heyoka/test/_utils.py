# Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from .. import _core


# Small helper to get the epsilon of a floating-point type.
def _get_eps(fp_t):
    if fp_t == float or fp_t == np.longdouble or fp_t == np.float32:
        return np.finfo(fp_t).eps

    if hasattr(_core, "real128"):
        return _core._get_real128_eps()

    raise TypeError(
        'Cannot compute the epsilon of the floating-point type "{}"'.format(fp_t)
    )


def _isclose(a, b, rtol, atol):
    def within_tol(x, y, atol, rtol):
        with np.errstate(invalid="ignore"):
            return np.less_equal(abs(x - y), atol + rtol * abs(y))

    x = np.asanyarray(a)
    y = np.asanyarray(b)

    xfin = np.isfinite(x)
    yfin = np.isfinite(y)
    if np.all(xfin) and np.all(yfin):
        return within_tol(x, y, atol, rtol)
    else:
        finite = xfin & yfin
        cond = np.zeros_like(finite, subok=True)
        # Because we're using boolean indexing, x & y must be the same shape.
        # Ideally, we'd just do x, y = broadcast_arrays(x, y). It's in
        # lib.stride_tricks, though, so we can't import it here.
        x = x * np.ones_like(cond)
        y = y * np.ones_like(cond)
        # Avoid subtraction with infinite/nan values...
        cond[finite] = within_tol(x[finite], y[finite], atol, rtol)
        # Check for equality of infinite values...
        cond[~finite] = x[~finite] == y[~finite]

        return cond[()]  # Flatten 0d arrays to scalars


def _allclose(a, b, rtol, atol):
    res = np.all(_isclose(a, b, rtol=rtol, atol=atol))
    return bool(res)
