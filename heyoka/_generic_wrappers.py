# Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from numpy import float32, float64, longdouble
from . import _core


def _with_real128():
    # Small helper to check if real128 is available.
    return hasattr(_core, "real128")


def _with_real():
    # Small helper to check if real is available.
    return hasattr(_core, "real")


_fp_to_suffix_dict = {
    float32: "_flt",
    float64: "_dbl",
    float: "_dbl",
    longdouble: "_ldbl",
}

if _with_real128():
    _fp_to_suffix_dict[_core.real128] = "_f128"

if _with_real():
    _fp_to_suffix_dict[_core.real] = "_real"


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
    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(_core, f"taylor_adaptive{fp_suffix}")(sys, state, **kwargs)


def taylor_adaptive_batch(sys, state, **kwargs):
    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(_core, f"taylor_adaptive_batch{fp_suffix}")(sys, state, **kwargs)


def recommended_simd_size(fp_type=float):
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(_core, f"_recommended_simd_size{fp_suffix}")()


def cfunc(fn, vars, **kwargs):
    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(_core, f"cfunc{fp_suffix}")(fn, vars, **kwargs)


def nt_event(ex, callback, **kwargs):
    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(_core, f"nt_event{fp_suffix}")(ex, callback, **kwargs)


def t_event(ex, **kwargs):
    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(_core, f"t_event{fp_suffix}")(ex, **kwargs)


def nt_event_batch(ex, callback, **kwargs):
    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(_core, f"nt_event_batch{fp_suffix}")(ex, callback, **kwargs)


def t_event_batch(ex, **kwargs):
    fp_type = kwargs.pop("fp_type", float)
    fp_suffix = _fp_to_suffix(fp_type)

    return getattr(_core, f"t_event_batch{fp_suffix}")(ex, **kwargs)
