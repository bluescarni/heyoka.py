# Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class mp_test_case(_ut.TestCase):
    def runTest(self):
        from . import core

        if not hasattr(core, "real"):
            return

        self.test_basic()

    def test_basic(self):
        from . import make_vars, taylor_adaptive, sin, real
        import numpy as np

        x, v = make_vars("x", "v")

        prec = 237

        # Small test with automatic inference of precision.
        ta = taylor_adaptive(
            [(x, v), (v, -9.8 * sin(x))],
            [real(-1, prec), real(0, prec)],
            fp_type=real,
        )

        self.assertEqual(ta.prec, 237)

        def compute_energy(sv):
            return (sv[1] * sv[1]) / 2 + 9.8 * (1 - np.cos(sv[0]))

        orig_E = compute_energy(ta.state)

        ta.propagate_until(real(10))

        self.assertLess(abs((orig_E - compute_energy(ta.state)) / orig_E), 1e-71)

        orig_sv = ta.state

        # Precision explicitly given.
        ta = taylor_adaptive(
            [(x, v), (v, -9.8 * sin(x))],
            [real(-1, prec - 1), real(0, prec + 1)],
            fp_type=real,
            prec=prec,
        )

        self.assertEqual(ta.prec, 237)

        ta.propagate_until(real(10))
        self.assertTrue(np.all(orig_sv == ta.state))
