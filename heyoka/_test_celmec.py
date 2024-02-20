# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


import unittest as _ut


class kepE_test_case(_ut.TestCase):
    def test_expr(self):
        from . import kepE, diff, make_vars, sin, cos, core
        from .core import _ppc_arch
        import numpy as np

        x, y = make_vars("x", "y")

        # Try a few overloads.
        kepE(x, y)
        kepE(0.1, y)
        kepE(e=x, M=0.2)

        self.assertEqual(
            diff(kepE(x, y), x), sin(kepE(x, y)) / (1.0 - x * cos(kepE(x, y)))
        )
        self.assertEqual(diff(kepE(x, y), y), 1.0 / (1.0 - x * cos(kepE(x, y))))

        # noconvert() behaviour.
        with self.assertRaises(TypeError) as cm:
            kepE(23, x)

        with self.assertRaises(TypeError) as cm:
            kepE(x, 23)

        if not _ppc_arch:
            self.assertEqual(
                diff(kepE(x, np.longdouble("1.1")), x),
                sin(kepE(x, np.longdouble("1.1")))
                / (1.0 - x * cos(kepE(x, np.longdouble("1.1")))),
            )
            self.assertEqual(
                diff(kepE(np.longdouble("1.1"), y), y),
                1.0 / (1.0 - np.longdouble("1.1") * cos(kepE(np.longdouble("1.1"), y))),
            )

            kepE(np.longdouble(0.1), y)
            kepE(x, np.longdouble(0.2))

            with self.assertRaises(TypeError) as cm:
                kepE(0.1, np.longdouble("1.1"))
            self.assertTrue(
                "At least one of the arguments of kepE() must be an expression"
                in str(cm.exception)
            )

        with self.assertRaises(TypeError) as cm:
            kepE(0.1, 0.2)
        self.assertTrue(
            "At least one of the arguments of kepE() must be an expression"
            in str(cm.exception)
        )

        if not hasattr(core, "real128"):
            return

        from .core import real128

        self.assertEqual(
            diff(kepE(x, real128("1.1")), x),
            sin(kepE(x, real128("1.1"))) / (1.0 - x * cos(kepE(x, real128("1.1")))),
        )
        self.assertEqual(
            diff(kepE(real128("1.1"), y), y),
            1.0 / (1.0 - real128("1.1") * cos(kepE(real128("1.1"), y))),
        )


class kepF_test_case(_ut.TestCase):
    def test_expr(self):
        from . import kepF, make_vars, core
        from .core import _ppc_arch
        import numpy as np

        x, y, z = make_vars("x", "y", "z")

        # Try a few overloads.
        kepF(x, y, z)
        kepF(0.1, y, z)
        kepF(h=0.1, k=0.2, lam=z)

        # noconvert() behaviour.
        with self.assertRaises(TypeError) as cm:
            kepF(23, x, y)

        with self.assertRaises(TypeError) as cm:
            kepF(x, 23, y)

        with self.assertRaises(TypeError) as cm:
            kepF(x, y, 23)

        if not _ppc_arch:
            kepF(x, y, np.longdouble("1.1"))
            kepF(x, np.longdouble(".1"), np.longdouble("1.1"))

            with self.assertRaises(TypeError) as cm:
                kepF(x, 0.1, np.longdouble("1.1"))
            self.assertTrue(
                "The numerical arguments of kepF() must be all of the same type"
                in str(cm.exception)
            )

        with self.assertRaises(TypeError) as cm:
            kepF(0.1, 0.2, 0.3)
        self.assertTrue(
            "At least one of the arguments of kepF() must be an expression"
            in str(cm.exception)
        )

        if not hasattr(core, "real128"):
            return

        from .core import real128

        kepF(real128(0.1), y, z)
        kepF(real128(0.1), real128(0.2), z)

        with self.assertRaises(TypeError) as cm:
            kepF(x, 0.1, real128("1.1"))
        self.assertTrue(
            "The numerical arguments of kepF() must be all of the same type"
            in str(cm.exception)
        )


class kepDE_test_case(_ut.TestCase):
    def test_expr(self):
        from . import kepDE, make_vars, core
        from .core import _ppc_arch
        import numpy as np

        x, y, z = make_vars("x", "y", "z")

        # Try a few overloads.
        kepDE(x, y, z)
        kepDE(0.1, y, z)
        kepDE(s0=0.1, c0=0.2, DM=z)

        # noconvert() behaviour.
        with self.assertRaises(TypeError) as cm:
            kepDE(23, x, y)

        with self.assertRaises(TypeError) as cm:
            kepDE(x, 23, y)

        with self.assertRaises(TypeError) as cm:
            kepDE(x, y, 23)

        if not _ppc_arch:
            kepDE(x, y, np.longdouble("1.1"))
            kepDE(x, np.longdouble(".1"), np.longdouble("1.1"))

            with self.assertRaises(TypeError) as cm:
                kepDE(x, 0.1, np.longdouble("1.1"))
            self.assertTrue(
                "The numerical arguments of kepDE() must be all of the same type"
                in str(cm.exception)
            )

        with self.assertRaises(TypeError) as cm:
            kepDE(0.1, 0.2, 0.3)
        self.assertTrue(
            "At least one of the arguments of kepDE() must be an expression"
            in str(cm.exception)
        )

        if not hasattr(core, "real128"):
            return

        from .core import real128

        kepDE(real128(0.1), y, z)
        kepDE(real128(0.1), real128(0.2), z)

        with self.assertRaises(TypeError) as cm:
            kepDE(x, 0.1, real128("1.1"))
        self.assertTrue(
            "The numerical arguments of kepDE() must be all of the same type"
            in str(cm.exception)
        )
