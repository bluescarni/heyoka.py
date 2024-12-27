# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

_with_sympy = True

try:
    import sympy as _spy
except ImportError:
    _with_sympy = False


def _from_sympy_symbol(sym):
    import re

    # Check if it is a parameter.
    m = re.match(r"par\[((?:[1-9][0-9]*|0))\]", sym.name)

    if m:
        from . import par

        return par[int(m.groups()[0])]
    else:
        from . import expression

        return expression(sym.name)


def _from_sympy_number(ex):
    is_rational = isinstance(ex, _spy.Rational)

    if (
        not isinstance(ex, _spy.Float)
        and not isinstance(ex, _spy.Integer)
        and not is_rational
    ):
        raise TypeError(
            "Only floating-point, integer and (some) rational numbers can be converted"
            " from sympy"
        )

    from . import expression

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
            ex.num.context.prec if isinstance(ex, _spy.Float) else int(ex).bit_length()
        )

    nf_err_msg = "A non-finite value was produced when converting from a sympy number"

    if prec <= 53:
        # Double precision is sufficient to represent
        # exactly the number.

        from math import isfinite

        retval = float(ex)

        # NOTE: a non-finite value could be produced if the original
        # number is non-finite or if its exponent is too large.
        if not isfinite(retval):
            raise ValueError(nf_err_msg)

        return expression(retval)

    import numpy as np

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

        return expression(retval)

    from . import core

    if hasattr(core, "real128") and prec <= 113:
        # We have real128, and quadmath precision
        # is enough to represent exactly the number.
        real128 = core.real128

        retval = real128(ex.p) / real128(ex.q) if is_rational else real128(str(ex))

        if not np.isfinite(retval):
            raise ValueError(nf_err_msg)

        return expression(retval)

    if hasattr(core, "real"):
        # We have real, we can in principle represent
        # any number.
        real = core.real

        # Ensure we are not going to employ
        # a too-low precision.
        prec = max(prec, core.real_prec_min())

        retval = (
            real(ex.p, prec) / real(ex.q, prec) if is_rational else real(str(ex), prec)
        )

        if not np.isfinite(retval):
            raise ValueError(nf_err_msg)

        return expression(retval)

    raise ValueError(
        "Cannot convert the number {} from sympy exactly: the required precision ({})"
        " is too high".format(ex, prec)
    )


def _build_fmap():
    if not _with_sympy:
        return None

    from . import core, pi, time as htime

    retval = {}

    retval[_spy.acos] = core.acos
    retval[_spy.acosh] = core.acosh
    retval[_spy.asin] = core.asin
    retval[_spy.asinh] = core.asinh
    retval[_spy.atan] = core.atan
    retval[_spy.atan2] = core.atan2
    retval[_spy.atanh] = core.atanh
    retval[_spy.cos] = core.cos
    retval[_spy.cosh] = core.cosh
    retval[_spy.erf] = core.erf
    retval[_spy.exp] = core.exp
    retval[_spy.log] = core.log
    retval[_spy.sin] = core.sin
    retval[_spy.sinh] = core.sinh
    retval[_spy.tan] = core.tan
    retval[_spy.tanh] = core.tanh
    retval[_spy.Pow] = lambda x, y: x**y
    # NOTE: sympy.pi is an instance of this type.
    retval[_spy.core.numbers.Pi] = lambda: pi

    def add_wrapper(*args):
        return core.sum(args)

    retval[_spy.Add] = add_wrapper

    def mul_wrapper(*args):
        return core.prod(args)

    retval[_spy.Mul] = mul_wrapper

    retval[_spy.Function("heyoka_kepE")] = core.kepE
    retval[_spy.Function("heyoka_kepF")] = core.kepF
    retval[_spy.Function("heyoka_kepDE")] = core.kepDE
    retval[_spy.Function("heyoka_time")] = lambda: htime

    return retval


_fmap = _build_fmap()


def _from_sympy_function(func, s_dict, c_dict):
    args = [_from_sympy_impl(arg, s_dict, c_dict) for arg in func.args]

    tp = type(func)

    if not tp in _fmap:
        raise TypeError("Unable to convert the sympy object {}".format(func))

    return _fmap[tp](*args)


def _from_sympy_impl(ex, s_dict, c_dict):
    # Check s_dict first.
    if ex in s_dict:
        return s_dict[ex]

    # Check if we already converted this expression.
    if id(ex) in c_dict:
        return c_dict[id(ex)]

    if isinstance(ex, _spy.Number):
        ret = _from_sympy_number(ex)
        c_dict[id(ex)] = ret
        return ret

    if isinstance(ex, _spy.Symbol):
        ret = _from_sympy_symbol(ex)
        c_dict[id(ex)] = ret
        return ret

    ret = _from_sympy_function(ex, s_dict, c_dict)
    c_dict[id(ex)] = ret

    return ret
