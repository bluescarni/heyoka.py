# Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


import unittest
from importlib.util import find_spec

import numpy as np
from .. import (
    to_sympy,
    from_sympy,
    make_vars,
    expression,
    par,
    pi,
    prod,
    sum as hsum,
    time as htime,
    _core,
)
from ..model import nbody

# NOTE: real and real128 are available only in some builds. Reference them through
# these module globals so the optional sub-tests can guard with "... is not None".
real = getattr(_core, "real", None)
real128 = getattr(_core, "real128", None)

# sympy (and mpmath) are optional dependencies: skip the whole test case if absent.
_has_sympy = find_spec("sympy") is not None


@unittest.skipUnless(_has_sympy, "sympy is not available")
class sympy_test_case(unittest.TestCase):
    def test_basic(self):
        import sympy

        with self.assertRaises(TypeError) as cm:
            from_sympy(3.5)
        self.assertTrue(
            "The 'ex' parameter must be a sympy expression but it is of type"
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            from_sympy(sympy.Symbol("x"), [])
        self.assertTrue(
            "The 's_dict' parameter must be a dict but it is of type"
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            from_sympy(sympy.Symbol("x"), {3.5: 3.5})
        self.assertTrue(
            "The keys in 's_dict' must all be sympy expressions" in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            from_sympy(sympy.Symbol("x"), {sympy.Symbol("x"): 3.5})
        self.assertTrue(
            "The values in 's_dict' must all be heyoka expressions" in str(cm.exception)
        )

        # Test the s_dict functionality of from_sympy().
        x, y = sympy.symbols("x y", real=True)
        hx, hy, hz = make_vars("x", "y", "z")

        self.assertEqual(
            from_sympy((x - y) * (x + y), s_dict={x - y: hz}), hsum([hx, hy]) * hz
        )

    def test_number_conversion(self):
        from sympy import Float, Rational, Integer
        from mpmath import workprec

        with self.assertRaises(ValueError) as cm:
            from_sympy(Rational(3, 5))
        self.assertTrue(
            "Cannot convert from sympy a rational number whose denominator is not a"
            " power of 2" in str(cm.exception)
        )

        # From integer.
        self.assertEqual(from_sympy(Integer(-42)), expression(-42.0))

        # From rational.
        self.assertEqual(from_sympy(Rational(42, -2)), expression(-21.0))

        # Single precision.
        with workprec(24):
            self.assertEqual(
                to_sympy(expression(np.float32("1.1"))),
                Float("1.1", precision=np.finfo(np.float32).nmant + 1),
            )

        # Double precision.
        with workprec(53):
            self.assertEqual(to_sympy(expression(1.1)), Float(1.1))
            self.assertEqual(from_sympy(Float(1.1)), expression(1.1))

            self.assertEqual(
                to_sympy(expression((2**40 + 1) / (2**128))),
                float(Rational(2**40 + 1, 2**128)),
            )
            self.assertEqual(
                from_sympy(Rational(2**40 + 1, 2**128)),
                expression((2**40 + 1) / (2**128)),
            )

        # Long double precision.
        if not _core._ppc_arch:
            with workprec(np.finfo(np.longdouble).nmant + 1):
                self.assertEqual(
                    to_sympy(expression(np.longdouble("1.1"))),
                    Float("1.1", precision=np.finfo(np.longdouble).nmant + 1),
                )

                # NOTE: on platforms where long double is not wider than
                # double (e.g., MSVC), conversion from sympy will produce a double
                # and these tests will fail.
                if np.finfo(np.longdouble).nmant > np.finfo(float).nmant:
                    self.assertEqual(
                        from_sympy(Float("1.1")), expression(np.longdouble("1.1"))
                    )

                    expo = np.finfo(np.longdouble).nmant - 10
                    self.assertEqual(
                        to_sympy(
                            expression(
                                np.longdouble(2**expo + 1) / np.longdouble(2**128)
                            )
                        ),
                        Float(
                            Rational(2**expo + 1, 2**128),
                            precision=np.finfo(np.longdouble).nmant + 1,
                        ),
                    )
                    self.assertEqual(
                        from_sympy(Rational(2**expo + 1, 2**128)),
                        expression(np.longdouble(2**expo + 1) / np.longdouble(2**128)),
                    )

        # Too high precision.
        if real is None:
            with self.assertRaises(ValueError) as cm:
                from_sympy(Integer(2**500 + 1))
            self.assertTrue("the required precision" in str(cm.exception))

        if real128 is None or _core._ppc_arch:
            return

        # Quad precision.
        with workprec(113):
            self.assertEqual(
                to_sympy(expression(real128("1.1"))), Float("1.1", precision=113)
            )
            self.assertEqual(from_sympy(Float("1.1")), expression(real128("1.1")))

            expo = 100
            self.assertEqual(
                to_sympy(expression(real128(2**expo + 1) / real128(2**128))),
                Float(Rational(2**expo + 1, 2**128), precision=113),
            )
            self.assertEqual(
                from_sympy(Rational(2**expo + 1, 2**128)),
                expression(real128(2**expo + 1) / real128(2**128)),
            )

    def test_sympar_conversion(self):
        from sympy import Symbol

        self.assertEqual(Symbol("x", real=True), to_sympy(expression("x")))
        self.assertEqual(Symbol("par[0]", real=True), to_sympy(par[0]))
        self.assertEqual(Symbol("par[9]", real=True), to_sympy(par[9]))
        self.assertEqual(Symbol("par[123]", real=True), to_sympy(par[123]))
        self.assertEqual(
            Symbol("par[-123]", real=True), to_sympy(expression("par[-123]"))
        )
        self.assertEqual(Symbol("par[]", real=True), to_sympy(expression("par[]")))

        self.assertEqual(from_sympy(Symbol("x")), expression("x"))
        self.assertEqual(from_sympy(Symbol("par[0]")), par[0])
        self.assertEqual(from_sympy(Symbol("par[9]")), par[9])
        self.assertEqual(from_sympy(Symbol("par[123]")), par[123])
        self.assertEqual(from_sympy(Symbol("par[-123]")), expression("par[-123]"))
        self.assertEqual(from_sympy(Symbol("par[]")), expression("par[]"))

    def test_func_conversion(self):
        import sympy as spy

        x, y, z, a, b, c = spy.symbols("x y z a b c", real=True)
        hx, hy, hz, ha, hb, hc = make_vars("x", "y", "z", "a", "b", "c")

        self.assertEqual(_core.acos(hx), from_sympy(spy.acos(x)))
        self.assertEqual(to_sympy(_core.acos(hx)), spy.acos(x))

        self.assertEqual(_core.acosh(hx), from_sympy(spy.acosh(x)))
        self.assertEqual(to_sympy(_core.acosh(hx)), spy.acosh(x))

        self.assertEqual(_core.asin(hx), from_sympy(spy.asin(x)))
        self.assertEqual(to_sympy(_core.asin(hx)), spy.asin(x))

        self.assertEqual(_core.asinh(hx), from_sympy(spy.asinh(x)))
        self.assertEqual(to_sympy(_core.asinh(hx)), spy.asinh(x))

        self.assertEqual(_core.atan(hx), from_sympy(spy.atan(x)))
        self.assertEqual(to_sympy(_core.atan(hx)), spy.atan(x))

        self.assertEqual(_core.atan2(hy, hx), from_sympy(spy.atan2(y, x)))
        self.assertEqual(to_sympy(_core.atan2(hy, hx)), spy.atan2(y, x))

        self.assertEqual(_core.atanh(hx), from_sympy(spy.atanh(x)))
        self.assertEqual(to_sympy(_core.atanh(hx)), spy.atanh(x))

        self.assertEqual(_core.cos(hx), from_sympy(spy.cos(x)))
        self.assertEqual(to_sympy(_core.cos(hx)), spy.cos(x))

        self.assertEqual(_core.cosh(hx), from_sympy(spy.cosh(x)))
        self.assertEqual(to_sympy(_core.cosh(hx)), spy.cosh(x))

        self.assertEqual(_core.erf(hx), from_sympy(spy.erf(x)))
        self.assertEqual(to_sympy(_core.erf(hx)), spy.erf(x))

        self.assertEqual(_core.exp(hx), from_sympy(spy.exp(x)))
        self.assertEqual(to_sympy(_core.exp(hx)), spy.exp(x))

        self.assertEqual(_core.log(hx), from_sympy(spy.log(x)))
        self.assertEqual(to_sympy(_core.log(hx)), spy.log(x))

        self.assertEqual(_core.sin(hx), from_sympy(spy.sin(x)))
        self.assertEqual(to_sympy(_core.sin(hx)), spy.sin(x))

        self.assertEqual(_core.sinh(hx), from_sympy(spy.sinh(x)))
        self.assertEqual(to_sympy(_core.sinh(hx)), spy.sinh(x))

        self.assertEqual(_core.sqrt(hx), from_sympy(spy.sqrt(x)))
        self.assertEqual(to_sympy(_core.sqrt(hx)), spy.sqrt(x))

        self.assertEqual(_core.tan(hx), from_sympy(spy.tan(x)))
        self.assertEqual(to_sympy(_core.tan(hx)), spy.tan(x))

        self.assertEqual(_core.tanh(hx), from_sympy(spy.tanh(x)))
        self.assertEqual(to_sympy(_core.tanh(hx)), spy.tanh(x))

        self.assertEqual(hx**3.5, from_sympy(x**3.5))
        self.assertEqual(to_sympy(hx**3.5), x**3.5)

        self.assertEqual(hsum([hx, hy, hz]), from_sympy(x + y + z))
        self.assertEqual(to_sympy(hx + hy + hz), x + y + z)
        self.assertEqual(to_sympy(hsum([hx, hy, hz])), x + y + z)
        self.assertEqual(to_sympy(hsum([hx])), x)
        self.assertEqual(to_sympy(hsum([])), 0)
        self.assertEqual(
            hsum([ha, hb, hc, hx, hy, hz]), from_sympy(x + y + z + a + b + c)
        )
        self.assertEqual(to_sympy(ha + hb + hc + hx + hy + hz), x + y + z + a + b + c)
        self.assertEqual(
            to_sympy(hsum([ha, hb, hc, hx, hy, hz])), x + y + z + a + b + c
        )

        self.assertEqual(prod([hx, hy, hz]), from_sympy(x * y * z))
        self.assertEqual(to_sympy(hx * hy * hz), x * y * z)
        self.assertEqual(
            prod([ha, hb, hc, hx, hy, hz]), from_sympy(x * y * z * a * b * c)
        )
        self.assertEqual(to_sympy(ha * hb * hc * hx * hy * hz), x * y * z * a * b * c)

        self.assertEqual(hsum([hx, -1.0 * hy, -1.0 * hz]), from_sympy(x - y - z))
        self.assertEqual(to_sympy(hx - hy - hz), x - y - z)

        # Run a test in the vector form as well.
        self.assertEqual(to_sympy([hx - hy - hz, hx * hy * hz]), [x - y - z, x * y * z])

        self.assertEqual(hx * hz**-1.0, from_sympy(x / z))
        self.assertEqual(to_sympy(hx / hz), x / z)

        self.assertEqual(
            _core.kepE(hx, hy), from_sympy(spy.Function("heyoka_kepE")(x, y))
        )
        self.assertEqual(
            to_sympy(_core.kepE(hx, hy)), spy.Function("heyoka_kepE")(x, y)
        )

        self.assertEqual(
            _core.kepF(hx, hy, hz), from_sympy(spy.Function("heyoka_kepF")(x, y, z))
        )
        self.assertEqual(
            to_sympy(_core.kepF(hx, hy, hz)), spy.Function("heyoka_kepF")(x, y, z)
        )

        self.assertEqual(
            _core.kepDE(hx, hy, hz), from_sympy(spy.Function("heyoka_kepDE")(x, y, z))
        )
        self.assertEqual(
            to_sympy(_core.kepDE(hx, hy, hz)), spy.Function("heyoka_kepDE")(x, y, z)
        )

        # relu/relup.
        self.assertEqual(
            to_sympy(_core.relu(hx)), spy.Piecewise((x, x > 0), (0.0, True))
        )
        self.assertEqual(
            to_sympy(_core.relup(hx)), spy.Piecewise((1.0, x > 0), (0.0, True))
        )
        self.assertEqual(
            to_sympy(_core.relu(hx, 0.1)), spy.Piecewise((x, x > 0), (x * 0.1, True))
        )
        self.assertEqual(
            to_sympy(_core.relup(hx, 0.1)), spy.Piecewise((1.0, x > 0), (0.1, True))
        )

        self.assertEqual(-1.0 * hx, from_sympy(-x))
        self.assertEqual(to_sympy(-hx), -x)

        self.assertEqual(
            to_sympy(_core.sigmoid(hx + hy)), 1.0 / (1.0 + spy.exp(-x - y))
        )

        self.assertEqual(htime, from_sympy(spy.Function("heyoka_time")()))
        self.assertEqual(to_sympy(htime), spy.Function("heyoka_time")())

        with self.assertRaises(TypeError) as cm:
            from_sympy(abs(x))
        self.assertTrue("Unable to convert the sympy object" in str(cm.exception))

        # Test caching behaviour.
        foo = hx + hy
        bar = foo / (foo * hz + 1.0)
        bar_spy = to_sympy(bar)
        self.assertEqual(
            id(bar_spy.args[1]), id(bar_spy.args[0].args[0].args[1].args[1])
        )

        # pi constant.
        self.assertEqual(to_sympy(pi), spy.pi)
        self.assertEqual(from_sympy(spy.pi), pi)
        self.assertEqual(to_sympy(from_sympy(spy.pi)), spy.pi)

        # nbody helper.
        [to_sympy(_[1]) for _ in nbody(2)]
        [to_sympy(_[1]) for _ in nbody(4)]
        [to_sympy(_[1]) for _ in nbody(10)]
