# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


import unittest as _ut


class var_ode_sys_test_case(_ut.TestCase):
    def test_enum(self):
        from . import var_args

        self.assertEqual(var_args.vars | var_args.time | var_args.params, var_args.all)
        self.assertTrue((var_args.vars | var_args.time) & var_args.time)

    def test_basic(self):
        from . import make_vars, var_ode_sys, var_args, sin, par, time
        from copy import copy, deepcopy
        from pickle import dumps, loads

        x, v = make_vars("x", "v")

        orig_sys = [(x, v), (v, -par[0] * sin(x) + time)]

        vsys = var_ode_sys(orig_sys, var_args.vars)

        self.assertEqual(orig_sys, vsys.sys[:2])
        self.assertEqual(vsys.vargs, [x, v])
        self.assertEqual(vsys.n_orig_sv, 2)
        self.assertEqual(vsys.order, 1)

        vsys = var_ode_sys(sys=orig_sys, args=[v, time, x], order=2)

        self.assertEqual(orig_sys, vsys.sys[:2])
        self.assertEqual(vsys.vargs, [v, time, x])
        self.assertEqual(vsys.n_orig_sv, 2)
        self.assertEqual(vsys.order, 2)

        vsys = var_ode_sys(orig_sys, var_args.vars | var_args.params)

        self.assertEqual(orig_sys, vsys.sys[:2])
        self.assertEqual(vsys.vargs, [x, v, par[0]])
        self.assertEqual(vsys.n_orig_sv, 2)
        self.assertEqual(vsys.order, 1)

        vsys2 = copy(vsys)

        self.assertEqual(orig_sys, vsys2.sys[:2])
        self.assertEqual(vsys2.vargs, [x, v, par[0]])
        self.assertEqual(vsys2.n_orig_sv, 2)
        self.assertEqual(vsys2.order, 1)

        vsys2 = deepcopy(vsys)

        self.assertEqual(orig_sys, vsys2.sys[:2])
        self.assertEqual(vsys2.vargs, [x, v, par[0]])
        self.assertEqual(vsys2.n_orig_sv, 2)
        self.assertEqual(vsys2.order, 1)

        vsys2 = loads(dumps(vsys))

        self.assertEqual(orig_sys, vsys2.sys[:2])
        self.assertEqual(vsys2.vargs, [x, v, par[0]])
        self.assertEqual(vsys2.n_orig_sv, 2)
        self.assertEqual(vsys2.order, 1)
