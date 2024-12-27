# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


import unittest as _ut


class sympy_test_case(_ut.TestCase):
    def test_basic(self):
        try:
            import sympy
        except ImportError:
            return

        from . import from_sympy, make_vars, sum as hsum

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
        try:
            import sympy
        except ImportError:
            return

        from . import to_sympy, from_sympy, expression, core
        from .core import _ppc_arch
        from sympy import Float, Rational, Integer
        from mpmath import workprec
        import numpy as np

        with self.assertRaises(ValueError) as cm:
            from_sympy(Rational(3, 5))
        self.assertTrue(
            "Cannot convert from sympy a rational number whose denominator is not a"
            " power of 2"
            in str(cm.exception)
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
        if not _ppc_arch:
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
        if not hasattr(core, "real"):
            with self.assertRaises(ValueError) as cm:
                from_sympy(Integer(2**500 + 1))
            self.assertTrue("the required precision" in str(cm.exception))

        if not hasattr(core, "real128") or _ppc_arch:
            return

        from .core import real128

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
        try:
            import sympy
        except ImportError:
            return

        from . import to_sympy, from_sympy, expression, par
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
        try:
            import sympy
        except ImportError:
            return

        import sympy as spy

        from . import (
            core,
            make_vars,
            from_sympy,
            to_sympy,
            pi,
            sum as hsum,
            prod,
            time as htime,
        )

        from .model import nbody

        x, y, z, a, b, c = spy.symbols("x y z a b c", real=True)
        hx, hy, hz, ha, hb, hc = make_vars("x", "y", "z", "a", "b", "c")

        self.assertEqual(core.acos(hx), from_sympy(spy.acos(x)))
        self.assertEqual(to_sympy(core.acos(hx)), spy.acos(x))

        self.assertEqual(core.acosh(hx), from_sympy(spy.acosh(x)))
        self.assertEqual(to_sympy(core.acosh(hx)), spy.acosh(x))

        self.assertEqual(core.asin(hx), from_sympy(spy.asin(x)))
        self.assertEqual(to_sympy(core.asin(hx)), spy.asin(x))

        self.assertEqual(core.asinh(hx), from_sympy(spy.asinh(x)))
        self.assertEqual(to_sympy(core.asinh(hx)), spy.asinh(x))

        self.assertEqual(core.atan(hx), from_sympy(spy.atan(x)))
        self.assertEqual(to_sympy(core.atan(hx)), spy.atan(x))

        self.assertEqual(core.atan2(hy, hx), from_sympy(spy.atan2(y, x)))
        self.assertEqual(to_sympy(core.atan2(hy, hx)), spy.atan2(y, x))

        self.assertEqual(core.atanh(hx), from_sympy(spy.atanh(x)))
        self.assertEqual(to_sympy(core.atanh(hx)), spy.atanh(x))

        self.assertEqual(core.cos(hx), from_sympy(spy.cos(x)))
        self.assertEqual(to_sympy(core.cos(hx)), spy.cos(x))

        self.assertEqual(core.cosh(hx), from_sympy(spy.cosh(x)))
        self.assertEqual(to_sympy(core.cosh(hx)), spy.cosh(x))

        self.assertEqual(core.erf(hx), from_sympy(spy.erf(x)))
        self.assertEqual(to_sympy(core.erf(hx)), spy.erf(x))

        self.assertEqual(core.exp(hx), from_sympy(spy.exp(x)))
        self.assertEqual(to_sympy(core.exp(hx)), spy.exp(x))

        self.assertEqual(core.log(hx), from_sympy(spy.log(x)))
        self.assertEqual(to_sympy(core.log(hx)), spy.log(x))

        self.assertEqual(core.sin(hx), from_sympy(spy.sin(x)))
        self.assertEqual(to_sympy(core.sin(hx)), spy.sin(x))

        self.assertEqual(core.sinh(hx), from_sympy(spy.sinh(x)))
        self.assertEqual(to_sympy(core.sinh(hx)), spy.sinh(x))

        self.assertEqual(core.sqrt(hx), from_sympy(spy.sqrt(x)))
        self.assertEqual(to_sympy(core.sqrt(hx)), spy.sqrt(x))

        self.assertEqual(core.tan(hx), from_sympy(spy.tan(x)))
        self.assertEqual(to_sympy(core.tan(hx)), spy.tan(x))

        self.assertEqual(core.tanh(hx), from_sympy(spy.tanh(x)))
        self.assertEqual(to_sympy(core.tanh(hx)), spy.tanh(x))

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
            core.kepE(hx, hy), from_sympy(spy.Function("heyoka_kepE")(x, y))
        )
        self.assertEqual(to_sympy(core.kepE(hx, hy)), spy.Function("heyoka_kepE")(x, y))

        self.assertEqual(
            core.kepF(hx, hy, hz), from_sympy(spy.Function("heyoka_kepF")(x, y, z))
        )
        self.assertEqual(
            to_sympy(core.kepF(hx, hy, hz)), spy.Function("heyoka_kepF")(x, y, z)
        )

        self.assertEqual(
            core.kepDE(hx, hy, hz), from_sympy(spy.Function("heyoka_kepDE")(x, y, z))
        )
        self.assertEqual(
            to_sympy(core.kepDE(hx, hy, hz)), spy.Function("heyoka_kepDE")(x, y, z)
        )

        # relu/relup.
        self.assertEqual(
            to_sympy(core.relu(hx)), spy.Piecewise((x, x > 0), (0.0, True))
        )
        self.assertEqual(
            to_sympy(core.relup(hx)), spy.Piecewise((1.0, x > 0), (0.0, True))
        )
        self.assertEqual(
            to_sympy(core.relu(hx, 0.1)), spy.Piecewise((x, x > 0), (x * 0.1, True))
        )
        self.assertEqual(
            to_sympy(core.relup(hx, 0.1)), spy.Piecewise((1.0, x > 0), (0.1, True))
        )

        self.assertEqual(-1.0 * hx, from_sympy(-x))
        self.assertEqual(to_sympy(-hx), -x)

        self.assertEqual(to_sympy(core.sigmoid(hx + hy)), 1.0 / (1.0 + spy.exp(-x - y)))

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
