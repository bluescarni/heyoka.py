# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class lagham_test_case(_ut.TestCase):
    def test_basic(self):
        # Just some basic testing for the keyword arguments.
        from . import lagrangian, hamiltonian, make_vars, cos, sin, expression

        x, v, p = make_vars("x", "v", "p")
        L = 0.5 * v**2 - (1.0 - cos(x))
        sys = lagrangian(L=L, qs=[x], qdots=[v], D=expression(0.0))
        self.assertEqual(sys, [(x, v), (v, -sin(x))])

        H = p * p / 2.0 + (1.0 - cos(x))
        sys = hamiltonian(H=H, qs=[x], ps=[p])
        self.assertEqual(
            sys,
            [
                (x, (0.50000000000000000 * (p + p))),
                (p, -(-(-sin(x)))),
            ],
        )
