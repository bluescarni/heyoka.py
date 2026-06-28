# Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import importlib.util
import re
from math import isfinite
import numpy as np
from . import _core
from ._globals import time as htime, par

_with_sympy = importlib.util.find_spec("sympy") is not None


def _from_sympy_symbol(sym):
    # Check if it is a parameter.
    m = re.match(r"par\[((?:[1-9][0-9]*|0))\]", sym.name)

    if m:
        return par[int(m.groups()[0])]
    else:
        return _core.expression(sym.name)


def _from_sympy_number(ex):
    import sympy

    is_rational = isinstance(ex, sympy.Rational)

    if (
        not isinstance(ex, sympy.Float)
        and not isinstance(ex, sympy.Integer)
        and not is_rational
    ):
        raise TypeError(
            "Only floating-point, integer and (some) rational numbers can be converted"
            " from sympy"
        )

    # Extract the needed precision in bits.
    # NOTE: the bit size returned by mpmath accounts
    # for the implicit bit and it is thus consistent
    # with the value returned by bit_length().
    if is_rational:
        # NOTE: for rationals we allow conversion only
        # if den is a power of 2.
        den = ex.q
        if not (den & (den - 1) == 0):
            raise ValueError(
                "Cannot convert from sympy a rational number whose denominator is not a"
                " power of 2"
            )

        # The needed precision is given by the bit size of the
        # numerator.
        prec = ex.p.bit_length()
    else:
        prec = (
            ex.num.context.prec if isinstance(ex, sympy.Float) else int(ex).bit_length()
        )

    nf_err_msg = "A non-finite value was produced when converting from a sympy number"

    if prec <= 53:
        # Double precision is sufficient to represent
        # exactly the number.
        retval = float(ex)

        # NOTE: a non-finite value could be produced if the original
        # number is non-finite or if its exponent is too large.
        if not isfinite(retval):
            raise ValueError(nf_err_msg)

        return _core.expression(retval)

    # NOTE: the number returned by finfo does not account for
    # the implicit bit.
    if prec <= np.finfo(np.longdouble).nmant + 1:
        # Long double precision is sufficient to represent
        # exactly the number.
        retval = (
            np.longdouble(ex.p) / np.longdouble(ex.q)
            if is_rational
            else np.longdouble(str(ex))
        )
        if not np.isfinite(retval):
            raise ValueError(nf_err_msg)

        return _core.expression(retval)

    if hasattr(_core, "real128") and prec <= 113:
        # We have real128, and quadmath precision
        # is enough to represent exactly the number.
        real128 = _core.real128

        retval = real128(ex.p) / real128(ex.q) if is_rational else real128(str(ex))

        if not np.isfinite(retval):
            raise ValueError(nf_err_msg)

        return _core.expression(retval)

    if hasattr(_core, "real"):
        # We have real, we can in principle represent
        # any number.
        real = _core.real

        # Ensure we are not going to employ
        # a too-low precision.
        prec = max(prec, _core.real_prec_min())

        retval = (
            real(ex.p, prec) / real(ex.q, prec) if is_rational else real(str(ex), prec)
        )

        if not np.isfinite(retval):
            raise ValueError(nf_err_msg)

        return _core.expression(retval)

    raise ValueError(
        "Cannot convert the number {} from sympy exactly: the required precision ({})"
        " is too high".format(ex, prec)
    )


def _build_fmap():
    if not _with_sympy:
        return None

    import sympy

    retval = {}

    retval[sympy.acos] = _core.acos
    retval[sympy.acosh] = _core.acosh
    retval[sympy.asin] = _core.asin
    retval[sympy.asinh] = _core.asinh
    retval[sympy.atan] = _core.atan
    retval[sympy.atan2] = _core.atan2
    retval[sympy.atanh] = _core.atanh
    retval[sympy.cos] = _core.cos
    retval[sympy.cosh] = _core.cosh
    retval[sympy.erf] = _core.erf
    retval[sympy.exp] = _core.exp
    retval[sympy.log] = _core.log
    retval[sympy.sin] = _core.sin
    retval[sympy.sinh] = _core.sinh
    retval[sympy.tan] = _core.tan
    retval[sympy.tanh] = _core.tanh
    retval[sympy.Pow] = lambda x, y: x**y
    # NOTE: sympy.pi is an instance of this type.
    retval[sympy.core.numbers.Pi] = lambda: _core.pi

    def add_wrapper(*args):
        return _core.sum(args)

    retval[sympy.Add] = add_wrapper

    def mul_wrapper(*args):
        return _core.prod(args)

    retval[sympy.Mul] = mul_wrapper

    retval[sympy.Function("heyoka_kepE")] = _core.kepE
    retval[sympy.Function("heyoka_kepF")] = _core.kepF
    retval[sympy.Function("heyoka_kepDE")] = _core.kepDE
    retval[sympy.Function("heyoka_time")] = lambda: htime

    return retval


_fmap = _build_fmap()


def _from_sympy_function(func, s_dict, c_dict):
    args = [_from_sympy_impl(arg, s_dict, c_dict) for arg in func.args]

    tp = type(func)

    if tp not in _fmap:
        raise TypeError("Unable to convert the sympy object {}".format(func))

    return _fmap[tp](*args)


def _from_sympy_impl(ex, s_dict, c_dict):
    import sympy

    # Check s_dict first.
    if ex in s_dict:
        return s_dict[ex]

    # Check if we already converted this expression.
    if id(ex) in c_dict:
        return c_dict[id(ex)]

    if isinstance(ex, sympy.Number):
        ret = _from_sympy_number(ex)
        c_dict[id(ex)] = ret
        return ret

    if isinstance(ex, sympy.Symbol):
        ret = _from_sympy_symbol(ex)
        c_dict[id(ex)] = ret
        return ret

    ret = _from_sympy_function(ex, s_dict, c_dict)
    c_dict[id(ex)] = ret

    return ret


def from_sympy(ex, s_dict={}):
    from sympy import Basic

    if not _with_sympy:
        raise ImportError(
            "The 'from_sympy()' function is not available because sympy is not"
            " installed"
        )

    if not isinstance(ex, Basic):
        raise TypeError(
            "The 'ex' parameter must be a sympy expression but it is of type {} instead".format(
                type(ex)
            )
        )

    if not isinstance(s_dict, dict):
        raise TypeError(
            "The 's_dict' parameter must be a dict but it is of type {} instead".format(
                type(s_dict)
            )
        )

    if any(not isinstance(_, Basic) for _ in s_dict):
        raise TypeError("The keys in 's_dict' must all be sympy expressions")

    if any(not isinstance(s_dict[_], _core.expression) for _ in s_dict):
        raise TypeError("The values in 's_dict' must all be heyoka expressions")

    return _from_sympy_impl(ex, s_dict, {})
