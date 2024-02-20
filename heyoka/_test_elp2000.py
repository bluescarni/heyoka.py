# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


import unittest as _ut


class elp2000_test_case(_ut.TestCase):
    # Just a small basic test.
    def test_basic(self):
        from . import cfunc
        from .model import elp2000_cartesian_e2000, elp2000_cartesian_fk5

        sol = elp2000_cartesian_e2000(thresh=1e-5)[0]
        cf = cfunc([sol], [])

        date = 2469000.5
        self.assertAlmostEqual(
            cf([], time=(date - 2451545.0) / 36525)[0], -361605.79234692274
        )

        sol = elp2000_cartesian_fk5(thresh=1e-5)[0]
        cf = cfunc([sol], [])

        self.assertAlmostEqual(
            cf([], time=(date - 2451545.0) / 36525)[0], -361605.7668217605
        )
