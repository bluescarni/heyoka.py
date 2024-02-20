# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


import unittest as _ut


class vsop2013_test_case(_ut.TestCase):
    # Just a small basic test.
    def test_basic(self):
        from . import cfunc
        from .model import vsop2013_elliptic, vsop2013_cartesian

        sol = vsop2013_elliptic(1, 1)
        cf = cfunc([sol], [])

        date = 2411545.0
        self.assertAlmostEqual(
            cf([], time=(date - 2451545.0) / 365250)[0], 0.3870979635
        )

        sol = vsop2013_cartesian(1, thresh=1e-8)
        cf = cfunc([sol[0]], [])

        self.assertAlmostEqual(
            cf([], time=(date - 2451545.0) / 365250)[0], 0.3493879042
        )
