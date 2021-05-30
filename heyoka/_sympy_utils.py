# Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
    m = re.match(r'par\[((?:[1-9][0-9]*|0))\]', sym.name)

    if m:
        from . import par
        return par[int(m.groups()[0])]
    else:
        from . import expression
        return expression(sym.name)

def _from_sympy_number(ex):
    is_rational = isinstance(ex, _spy.Rational)

    if not isinstance(ex, _spy.Float) and not isinstance(ex, _spy.Integer) and not is_rational:
        raise TypeError("Only floating-point, integer and (some) rational numbers can be converted from sympy")
    
    from . import expression

    # Extract the needed precision in bits.
    # NOTE: the bit size returned by mpmath accounts
    # for the implicit bit and it is thus consistent
    # with the value returned by bit_length().
    if is_rational:
        # NOTE: for rationals we allow conversion only
        # if den is a power of 2.
        den = int(ex.denominator())
        if not (den & (den-1) == 0):
            raise ValueError("Cannot convert from sympy a rational number whose denominator is not a power of 2")

        # The needed precision is given by the bit size of the
        # numerator.
        prec = int(ex.numerator()).bit_length()
    else:
        prec = ex.num.context.prec if isinstance(ex, _spy.Float) else int(ex).bit_length()

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
        retval = np.longdouble(str(ex.numerator())) / np.longdouble(str(ex.denominator())) if is_rational else np.longdouble(str(ex))
        if not np.isfinite(retval):
            raise ValueError(nf_err_msg)

        return expression(retval)

    from . import with_real128

    if with_real128 and prec <= 113:
        # We have mpmath and real128, and quadmath precision
        # is enough to represent exactly the number. Temporarily
        # set the working precision to 113 and create an expression
        # from a quadmath number.
        from mpmath import mpf, workprec, isfinite
        with workprec(113):
            retval = mpf(str(ex.numerator())) / mpf(str(ex.denominator())) if is_rational else mpf(str(ex))
            if not isfinite(retval):
                raise ValueError(nf_err_msg)

            return expression(retval)            
    else:
        raise ValueError("Cannot convert the number {} from sympy exactly: the required precision ({}) is too high".format(ex, prec))

def _build_fmap():
    if not _with_sympy:
        return None

    from . import core

    retval = {}

    retval[_spy.acos] = core.acos
    retval[_spy.acosh] = core.acosh
    retval[_spy.asin] = core.asin
    retval[_spy.asinh] = core.asinh
    retval[_spy.atan] = core.atan
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

    def add_wrapper(*args):
        return core.pairwise_sum(args)

    retval[_spy.Add] = add_wrapper

    def mul_wrapper(*args):
        return core.pairwise_prod(args)

    retval[_spy.Mul] = mul_wrapper

    retval[_spy.Function("heyoka_kepE")] = core.kepE
    retval[_spy.Function("heyoka_time")] = core._time_func
    retval[_spy.Function("heyoka_tpoly")] = core.tpoly

    return retval

_fmap = _build_fmap()

def _from_sympy_function(func):
    args = [_from_sympy_impl(arg) for arg in func.args]

    tp = type(func)

    if not tp in _fmap:
        raise TypeError("Unable to convert the sympy function {}".format(tp))

    return _fmap[tp](*args)

def _from_sympy_impl(ex):
    if isinstance(ex, _spy.Number):
        return _from_sympy_number(ex)

    if isinstance(ex, _spy.Symbol):
        return _from_sympy_symbol(ex)

    return _from_sympy_function(ex)
