# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class expression_test_case(_ut.TestCase):
    def test_basic(self):
        from . import expression as ex, core
        from . import make_vars
        import numpy as np

        with_real128 = hasattr(core, "real128")
        ld_63bit = np.finfo(np.longdouble).nmant == 63

        if with_real128:
            real128 = core.real128

        # Minimal make_vars() testing.
        with self.assertRaises(ValueError) as cm:
            make_vars()
        self.assertTrue(
            "At least one argument is required when invoking 'make_vars()'"
            in str(cm.exception)
        )

        x = make_vars("x")
        self.assertTrue(isinstance(x, ex))
        self.assertEqual(x, ex("x"))

        l = make_vars("x", "y")
        self.assertTrue(isinstance(l, list))
        x, y = l
        self.assertEqual(x, ex("x"))
        self.assertEqual(y, ex("y"))

        # Constructors.
        self.assertEqual(ex(), ex(0))
        self.assertEqual(ex(123), ex(123.0))
        self.assertEqual(str(ex(123)), "123.00000000000000")
        self.assertEqual(str(ex(np.float32("1.1"))), "1.10000002")

        # Error with large integer.
        with self.assertRaises(TypeError) as cm:
            ex(123 << 56)
        self.assertTrue("incompatible constructor arguments" in str(cm.exception))

        self.assertEqual(str(ex(1.1)), "1.1000000000000001")

        if ld_63bit:
            self.assertEqual(str(ex(np.longdouble("1.1"))), "1.10000000000000000002")

        if with_real128:
            self.assertEqual(
                str(ex(real128("1.1"))), "1.10000000000000000000000000000000008"
            )

        self.assertEqual(str(ex("x")), "x")

        # Unary arithmetics.
        self.assertEqual(ex(42), +ex(42))
        self.assertEqual(ex(-42), -ex(42))

        # Addition.
        self.assertEqual(ex(42) + ex(-1), ex(41))
        self.assertEqual(ex(42) + -1, ex(41))
        self.assertEqual(-1 + ex(42), ex(41))
        with self.assertRaises(TypeError) as cm:
            ex(42) + (2 << 112)
        with self.assertRaises(TypeError) as cm:
            (2 << 112) + ex(42)
        self.assertEqual(ex(42) + -1.1, ex(40.899999999999999))
        self.assertEqual(-1.1 + ex(42), ex(40.899999999999999))
        if ld_63bit:
            self.assertEqual(
                ex(42) + np.longdouble("-1.1"),
                ex(np.longdouble("40.9000000000000000014")),
            )
            self.assertEqual(
                np.longdouble("-1.1") + ex(42),
                ex(np.longdouble("40.9000000000000000014")),
            )
        if with_real128:
            self.assertEqual(
                ex(42) + real128("-1.1"),
                ex(real128("40.8999999999999999999999999999999988")),
            )
            self.assertEqual(
                real128("-1.1") + ex(42),
                ex(real128("40.8999999999999999999999999999999988")),
            )

        # Subtraction.
        self.assertEqual(ex(42) - ex(-1), ex(43))
        self.assertEqual(ex(42) - -1, ex(43))
        self.assertEqual(-1 - ex(42), ex(-43))
        with self.assertRaises(TypeError) as cm:
            ex(42) - (2 << 112)
        with self.assertRaises(TypeError) as cm:
            (2 << 112) - ex(42)
        self.assertEqual(ex(42) - -1.1, ex(43.100000000000001))
        self.assertEqual(-1.1 - ex(42), ex(-43.100000000000001))
        if ld_63bit:
            self.assertEqual(
                ex(42) - np.longdouble("-1.1"),
                ex(np.longdouble("43.0999999999999999986")),
            )
            self.assertEqual(
                np.longdouble("-1.1") - ex(42),
                ex(np.longdouble("-43.0999999999999999986")),
            )
        if with_real128:
            self.assertEqual(
                ex(42) - real128("-1.1"),
                ex(real128("43.1000000000000000000000000000000012")),
            )
            self.assertEqual(
                real128("-1.1") - ex(42),
                ex(real128("-43.1000000000000000000000000000000012")),
            )

        # Multiplication.
        self.assertEqual(ex(42) * ex(-1), ex(-42))
        self.assertEqual(ex(42) * -1, ex(-42))
        self.assertEqual(-1 * ex(42), ex(-42))
        with self.assertRaises(TypeError) as cm:
            ex(42) * (2 << 112)
        with self.assertRaises(TypeError) as cm:
            (2 << 112) * ex(42)
        self.assertEqual(ex(42) * -1.1, ex(-46.200000000000003))
        self.assertEqual(-1.1 * ex(42), ex(-46.200000000000003))
        if ld_63bit:
            self.assertEqual(
                ex(42) * np.longdouble("-1.1"),
                ex(np.longdouble("-46.2000000000000000007")),
            )
            self.assertEqual(
                np.longdouble("-1.1") * ex(42),
                ex(np.longdouble("-46.2000000000000000007")),
            )
        if with_real128:
            self.assertEqual(
                ex(42) * real128("-1.1"),
                ex(real128("-46.2000000000000000000000000000000025")),
            )
            self.assertEqual(
                real128("-1.1") * ex(42),
                ex(real128("-46.2000000000000000000000000000000025")),
            )

        # Division.
        self.assertEqual(ex(42) / ex(-1), ex(-42))
        self.assertEqual(ex(42) / -1, ex(-42))
        self.assertEqual(-42 / ex(1), ex(-42))
        with self.assertRaises(TypeError) as cm:
            ex(42) / (2 << 112)
        with self.assertRaises(TypeError) as cm:
            (2 << 112) / ex(42)
        self.assertEqual(ex(42) / -1.1, ex(-38.181818181818180))
        self.assertEqual(-1.1 / ex(42), ex(-0.02619047619047619))
        if ld_63bit:
            self.assertEqual(
                ex(42) / np.longdouble("-1.1"),
                ex(np.longdouble("-38.1818181818181818163")),
            )
            self.assertEqual(
                np.longdouble("-1.1") / ex(42),
                ex(np.longdouble("-0.0261904761904761904772")),
            )
        if with_real128:
            self.assertEqual(
                ex(42) / real128("-1.1"),
                ex(real128("-38.1818181818181818181818181818181801")),
            )
            self.assertEqual(
                real128("-1.1") / ex(42),
                ex(real128("-0.0261904761904761904761904761904761910")),
            )

        # Comparison.
        self.assertEqual(ex(42), ex(42))
        self.assertEqual(ex("x"), ex("x"))
        self.assertNotEqual(ex(41), ex(42))
        self.assertNotEqual(ex("x"), ex("y"))

        # Exponentiation.
        self.assertEqual(str(ex("x") ** ex("y")), "x**y")
        self.assertEqual(str(ex("x") ** ex(2)), "x**2.0000000000000000")
        self.assertEqual(str(ex("x") ** ex(1.1)), "x**1.1000000000000001")
        with self.assertRaises(TypeError) as cm:
            ex(42) ** (2 << 112)
        if ld_63bit:
            self.assertEqual(
                ex(42) / np.longdouble("-1.1"),
                ex(np.longdouble("-38.1818181818181818163")),
            )
        if ld_63bit:
            self.assertEqual(
                str(ex("x") ** ex(np.longdouble("1.1"))),
                "x**1.10000000000000000002",
            )
        if with_real128:
            self.assertEqual(
                str(ex("x") ** ex(real128("1.1"))),
                "x**1.10000000000000000000000000000000008",
            )

        # Copy and deepcopy.
        from copy import copy, deepcopy

        tmp = ex("x") + ex("y")
        tmp.foo = [1, 2, 3]
        id_foo = id(tmp.foo)
        tmp_copy = copy(tmp)
        self.assertEqual(id_foo, id(tmp_copy.foo))
        tmp_dcopy = deepcopy(tmp)
        self.assertNotEqual(id_foo, id(tmp_dcopy.foo))
        self.assertEqual(tmp.foo, tmp_dcopy.foo)

    def test_copy(self):
        from . import make_vars
        from copy import copy, deepcopy

        x, y = make_vars("x", "y")
        ex = x + y

        class foo:
            pass

        ex.bar = foo()

        self.assertEqual(id(ex.bar), id(copy(ex).bar))
        self.assertNotEqual(id(ex.bar), id(deepcopy(ex).bar))
        self.assertEqual(ex, copy(ex))
        self.assertEqual(ex, deepcopy(ex))

    def test_diff(self):
        from . import make_vars, sin, cos, diff, par

        x, y = make_vars("x", "y")
        self.assertEqual(diff(cos(x * x - y), "x"), -sin(x * x - y) * (x + x))
        self.assertEqual(diff(cos(x * x - y), x), -sin(x * x - y) * (x + x))
        self.assertEqual(
            diff(cos(par[0] * par[0] - y), par[0]),
            -sin(par[0] * par[0] - y) * (par[0] + par[0]),
        )

    def test_s11n(self):
        from . import make_vars, sin, cos, core
        from .core import _ppc_arch
        from numpy import longdouble
        import pickle

        x, y = make_vars("x", "y")

        ex = x + 2.0 * y
        self.assertEqual(ex, pickle.loads(pickle.dumps(ex)))

        # Test dynamic attributes.
        ex.foo = "hello world"
        ex = pickle.loads(pickle.dumps(ex))
        self.assertEqual(ex.foo, "hello world")

        if not _ppc_arch:
            ex = sin(longdouble("1.1") * x) + 2.0 * y
            self.assertEqual(ex, pickle.loads(pickle.dumps(ex)))

        if not hasattr(core, "real128"):
            return

        from .core import real128

        # Quad precision.
        if not _ppc_arch:
            ex = sin(longdouble("1.1") * x) + real128("1.3") * cos(2.0 * y)
            self.assertEqual(ex, pickle.loads(pickle.dumps(ex)))

    def test_len(self):
        from . import make_vars

        x, y, z = make_vars("x", "y", "z")

        self.assertEqual(len(x), 1)
        self.assertEqual(len((x - y - z) + (y * z)), 13)

    def test_hash(self):
        from . import make_vars
        from copy import deepcopy

        x, y, z = make_vars("x", "y", "z")

        ex = x + y
        self.assertEqual(z * ex + ex, z * (x + y) + (x + y))
        self.assertEqual(hash(z * ex + ex), hash(z * (x + y) + (x + y)))
        ex2 = deepcopy(ex)
        self.assertEqual(hash(z * ex2 + ex2), hash(z * (x + y) + (x + y)))

    def test_get_variables(self):
        from . import make_vars, get_variables

        x, y, z = make_vars("x", "y", "z")
        self.assertEqual(get_variables(arg=x + y), ["x", "y"])
        self.assertEqual(get_variables(arg=[z - y, x + y]), ["x", "y", "z"])

    def test_rename_variables(self):
        from . import make_vars, rename_variables

        x, y, a, b = make_vars("x", "y", "a", "b")
        self.assertEqual(rename_variables(arg=x + y, d={"x": "b", "y": "a"}), b + a)
        self.assertEqual(
            rename_variables(arg=[x - y, x + y], d={"x": "b", "y": "a"}),
            [b - a, b + a],
        )

    def test_subs(self):
        from . import make_vars, subs

        x, y, a, b = make_vars("x", "y", "a", "b")
        self.assertEqual(str(subs(arg=x + y, smap={"x": b, "y": a})), "(b + a)")
        self.assertEqual(str(subs(arg=x + y, smap={x: b, y: a})), "(b + a)")
        self.assertEqual(
            str(subs(arg=[x + y, x - y], smap={"x": b, "y": a})[1]), "(b - a)"
        )
        self.assertEqual(str(subs(arg=[x + y, x - y], smap={x: b, y: a})[1]), "(b - a)")

    def test_relu_wrappers(self):
        from . import make_vars, leaky_relu, leaky_relup, relu, relup

        x, y = make_vars("x", "y")

        self.assertEqual(leaky_relu(0.0)(x), relu(x))
        self.assertEqual(leaky_relup(0.0)(x), relup(x))
        self.assertEqual(leaky_relu(0.1)(x + y), relu(x + y, 0.1))
        self.assertEqual(leaky_relup(0.1)(x + y), relup(x + y, 0.1))

        self.assertEqual(leaky_relu(0.0)(x * y), relu(x * y))
        self.assertEqual(leaky_relup(0.0)(x * y), relup(x * y))
        self.assertEqual(leaky_relu(0.1)(x * y + y), relu(x * y + y, 0.1))
        self.assertEqual(leaky_relup(0.1)(x * y + y), relup(x * y + y, 0.1))

    def test_dfun(self):
        from . import make_vars, dfun

        x, y = make_vars("x", "y")

        self.assertEqual(str(dfun("f", [x, y])), "(∂^0 f)")
        self.assertEqual(
            str(dfun(name="f", args=[x, y], didx=[(0, 3), (1, 5)])),
            "(∂^8 f)/(∂a0^3 ∂a1^5)",
        )

    def test_relational(self):
        from . import make_vars, lt, eq

        x, y = make_vars("x", "y")

        self.assertEqual(str(lt(x, y)), "(x < y)")
        self.assertEqual(str(eq(x, y)), "(x == y)")
        self.assertTrue("(x == 1.00" in str(eq(x, 1.0)))
        self.assertTrue("0000 == x)" in str(eq(1.0, x)))

        with self.assertRaises(TypeError) as cm:
            lt(1.0, 1.0)
        self.assertTrue(
            "At least one of the arguments of lt() must be an expression"
            in str(cm.exception)
        )

    def test_logical(self):
        from . import make_vars, logical_and, logical_or

        x, y = make_vars("x", "y")

        self.assertEqual(str(logical_and([x, y])), "logical_and(x, y)")
        self.assertEqual(str(logical_or([x, y])), "logical_or(x, y)")

    def test_select(self):
        from . import make_vars, select
        import numpy as np

        x, y = make_vars("x", "y")

        self.assertEqual(str(select(x, y, x)), "select(x, y, x)")

        self.assertTrue("select(1.0000" in str(select(1.0, y, x)))
        self.assertTrue("000, y, x)" in str(select(1.0, y, x)))

        self.assertTrue("select(1.0000" in str(select(1.0, y, 2.0)))
        self.assertTrue("000, y, 2.000" in str(select(1.0, y, 2.0)))

        with self.assertRaises(TypeError) as cm:
            select(1.0, 1.0, 1.0)
        self.assertTrue(
            "At least one of the arguments of select() must be an expression"
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            select(1.0, y, np.longdouble(2.0))
        self.assertTrue(
            "The numerical arguments of select() must be all of the same type"
            in str(cm.exception)
        )

    def test_get_params(self):
        from . import make_vars, par, get_params

        x, y = make_vars("x", "y")

        self.assertEqual(get_params(x + y), [])
        self.assertEqual(get_params(x + par[42]), [par[42]])

        self.assertEqual(get_params([x + y, x - y]), [])
        self.assertEqual(get_params([x + par[42], par[1] - y]), [par[1], par[42]])
