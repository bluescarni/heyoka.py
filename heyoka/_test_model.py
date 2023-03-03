# Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class model_test_case(_ut.TestCase):
    def test_nbody(self):
        from . import model, expression

        dyn = model.nbody(2, masses=[0.0, 0.0])

        self.assertEqual(len(dyn), 12)

        self.assertEqual(dyn[3][1], expression(0.0))
        self.assertEqual(dyn[4][1], expression(0.0))
        self.assertEqual(dyn[5][1], expression(0.0))

        self.assertEqual(dyn[9][1], expression(0.0))
        self.assertEqual(dyn[10][1], expression(0.0))
        self.assertEqual(dyn[11][1], expression(0.0))

        dyn = model.nbody(2, Gconst=5.0)

        self.assertTrue("5.0000000000000" in str(dyn[3][1]))
        self.assertTrue("5.0000000000000" in str(dyn[4][1]))
        self.assertTrue("5.0000000000000" in str(dyn[5][1]))

        self.assertTrue("5.0000000000000" in str(dyn[9][1]))
        self.assertTrue("5.0000000000000" in str(dyn[10][1]))
        self.assertTrue("5.0000000000000" in str(dyn[11][1]))

        en = model.nbody_energy(2, masses=[0.0, 0.0])

        self.assertEqual(en, expression(0.0))

        en = model.nbody_energy(2, Gconst=5.0)

        self.assertTrue("5.0000000000000" in str(en))

        dyn = model.np1body(2, masses=[0.0, 0.0])

        self.assertEqual(len(dyn), 6)

        self.assertEqual(dyn[3][1], expression(0.0))
        self.assertEqual(dyn[4][1], expression(0.0))
        self.assertEqual(dyn[5][1], expression(0.0))

        dyn = model.np1body(2, Gconst=5.0)

        self.assertTrue("10.0000000000000" in str(dyn[3][1]))
        self.assertTrue("10.0000000000000" in str(dyn[4][1]))
        self.assertTrue("10.0000000000000" in str(dyn[5][1]))

        en = model.np1body_energy(2, masses=[])

        self.assertEqual(en, expression(0.0))

        en = model.np1body_energy(2, Gconst=5.0)

        self.assertTrue("5.0000000000000" in str(en))

    def test_pendulum(self):
        from . import model, expression, make_vars, sin, cos

        x, v = make_vars("x", "v")

        dyn = model.pendulum()

        self.assertEqual(dyn[0][0], x)
        self.assertEqual(dyn[0][1], v)
        self.assertEqual(dyn[1][0], v)
        self.assertEqual(dyn[1][1], -sin(x))

        dyn = model.pendulum(gconst=2.0)

        self.assertEqual(dyn[0][0], x)
        self.assertEqual(dyn[0][1], v)
        self.assertEqual(dyn[1][0], v)
        self.assertEqual(dyn[1][1], -2.0 * sin(x))

        dyn = model.pendulum(gconst=4.0, l=2.0)

        self.assertEqual(dyn[0][0], x)
        self.assertEqual(dyn[0][1], v)
        self.assertEqual(dyn[1][0], v)
        self.assertEqual(dyn[1][1], -2.0 * sin(x))

        en = model.pendulum_energy()

        self.assertEqual(
            en, ((0.50000000000000000 * v**2) + (1.0000000000000000 - cos(x)))
        )

        en = model.pendulum_energy(gconst=2.0)

        self.assertEqual(
            en,
            (
                (0.50000000000000000 * v**2)
                + (2.0000000000000000 * (1.0000000000000000 - cos(x)))
            ),
        )

        en = model.pendulum_energy(l=2.0, gconst=4.0)

        self.assertEqual(
            en,
            (
                (
                    (2.0000000000000000 * v**2)
                    + (8.0000000000000000 * (1.0000000000000000 - cos(x)))
                )
            ),
        )
