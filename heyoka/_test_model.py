# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class model_test_case(_ut.TestCase):
    def test_mascon(self):
        from . import (
            model,
            make_vars,
            expression as ex,
            sqrt,
            sum as hysum,
            par,
        )

        x, y, z, vx, vy, vz = make_vars("x", "y", "z", "vx", "vy", "vz")

        dyn = model.mascon(
            Gconst=1.5, masses=[1.1], positions=[[1.0, 2.0, 3.0]], omega=[0.0, 0.0, 3.0]
        )

        self.assertEqual(dyn[0][0], x)
        self.assertEqual(dyn[0][1], ex("vx"))

        pot = model.mascon_potential(
            Gconst=1.5, masses=[1.1], positions=[[1.0, 2.0, 3.0]], omega=[0.0, 0.0, 3.0]
        )

        en = model.mascon_energy(
            Gconst=1.5, masses=[1.1], positions=[[1.0, 2.0, 3.0]], omega=[0.0, 0.0, 3.0]
        )

        self.assertGreater(len(en), len(pot))

        with self.assertRaises(ValueError) as cm:
            model.mascon(
                Gconst=1.5,
                masses=[1.1],
                positions=[1.0, 2.0, 3.0],
                omega=[0.0, 0.0, 3.0],
            )
        self.assertTrue(
            "Invalid positions array in a mascon model: the number of dimensions must be 2, but it is 1 instead"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            model.mascon(
                Gconst=1.5,
                masses=[1.1],
                positions=[[1.0, 2.0, 3.0, 4.0]],
                omega=[0.0, 0.0, 3.0],
            )
        self.assertTrue(
            "Invalid positions array in a mascon model: the number of columns must be 3, but it is 4 instead"
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            model.mascon(
                Gconst=1.5,
                masses=[1.1],
                positions=[[{}, {}, {}]],
                omega=[0.0, 0.0, 3.0],
            )
        self.assertTrue(
            "The positions array in a mascon model could not be converted into an array of expressions - please make sure that the array's values can be converted into heyoka expressions"
            in str(cm.exception)
        )

        # Run also a test with parametric mass.
        dyn = model.mascon(
            Gconst=1.5,
            masses=[par[0]],
            positions=[[1.0, 2.0, 3.0]],
            omega=[0.0, 0.0, 3.0],
        )

    def test_rotating(self):
        from . import model, make_vars, expression as ex, sum as hysum

        x, y, z, vx, vy, vz = make_vars("x", "y", "z", "vx", "vy", "vz")

        dyn = model.rotating(omega=[0.0, 0.0, 3.0])

        self.assertEqual(dyn[0][0], x)
        self.assertEqual(dyn[0][1], ex("vx"))

        self.assertEqual(
            dyn[3][1], hysum([9.0000000000000000 * x, 6.0000000000000000 * vy])
        )

        pot = model.rotating_potential([0.0, 0.0, 3.0])

        en = model.rotating_energy([0.0, 0.0, 3.0])

    def test_fixed_centres(self):
        from . import model, make_vars, expression as ex, sqrt

        x, y, z, vx, vy, vz = make_vars("x", "y", "z", "vx", "vy", "vz")

        dyn = model.fixed_centres(Gconst=1.5, masses=[1.1], positions=[[1.0, 2.0, 3.0]])

        self.assertEqual(dyn[0][0], x)
        self.assertEqual(dyn[0][1], ex("vx"))

        en = model.fixed_centres_energy(
            Gconst=1.5, masses=[1.1], positions=[[1.0, 2.0, 3.0]]
        )

        pot = model.fixed_centres_potential(
            Gconst=1.5, masses=[1.1], positions=[[1.0, 2.0, 3.0]]
        )

        with self.assertRaises(ValueError) as cm:
            model.fixed_centres(Gconst=1.5, masses=[1.1], positions=[1.0, 2.0, 3.0])
        self.assertTrue(
            "Invalid positions array in a fixed centres model: the number of dimensions must be 2, but it is 1 instead"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            model.fixed_centres(
                Gconst=1.5, masses=[1.1], positions=[[1.0, 2.0, 3.0, 4.0]]
            )
        self.assertTrue(
            "Invalid positions array in a fixed centres model: the number of columns must be 3, but it is 4 instead"
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            model.fixed_centres(Gconst=1.5, masses=[1.1], positions=[[{}, {}, {}]])
        self.assertTrue(
            "The positions array in a fixed centres model could not be converted into an array of expressions - please make sure that the array's values can be converted into heyoka expressions"
            in str(cm.exception)
        )

    def test_nbody(self):
        from . import model, expression, sqrt, make_vars

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

        x0, y0, z0, x1, y1, z1 = make_vars("x_0", "y_0", "z_0", "x_1", "y_1", "z_1")

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

        dyn = model.pendulum(gconst=4.0, length=2.0)

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

        en = model.pendulum_energy(length=2.0, gconst=4.0)

        self.assertEqual(
            en,
            (
                (
                    (2.0000000000000000 * v**2)
                    + (8.0000000000000000 * (1.0000000000000000 - cos(x)))
                )
            ),
        )

    def test_cr3bp(self):
        from . import model, make_vars

        x, px, y = make_vars("x", "px", "y")

        dyn = model.cr3bp()
        self.assertEqual(dyn[0][0], x)
        self.assertEqual(dyn[0][1], px + y)

        dyn = model.cr3bp(mu=1.0 / 2**4)
        self.assertTrue("0.06250000000" in str(dyn[3][1]))

        jac = model.cr3bp_jacobi()
        self.assertTrue("0.00100000" in str(jac))

        jac = model.cr3bp_jacobi(mu=1.0 / 2**4)
        self.assertTrue("0.06250000000" in str(jac))

    def test_ffnn(self):
        from . import model, make_vars, tanh, expression, par

        x, y = make_vars("x", "y")

        linear = lambda x: x
        my_ffnn1 = model.ffnn([x], [], 1, [linear])
        my_ffnn2 = model.ffnn([x, y], [], 1, [linear])
        self.assertTrue(my_ffnn1[0] == par[1] + (par[0] * x))
        self.assertTrue(my_ffnn2[0] == par[2] + (par[0] * x) + (par[1] * y))

        my_ffnn3 = model.ffnn([x], [], 1, [linear], [expression(1.2), expression(1.3)])
        self.assertTrue(my_ffnn3[0] == expression(1.3) + (expression(1.2) * x))

        my_ffnn4 = model.ffnn([x], [], 1, [linear], [1.2, 1.3])
        self.assertTrue(my_ffnn4[0] == expression(1.3) + (expression(1.2) * x))
