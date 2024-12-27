# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


import unittest as _ut


class var_integrator_test_case(_ut.TestCase):
    def test_scalar(self):
        from . import (
            make_vars,
            var_ode_sys,
            var_args,
            cos,
            sin,
            par,
            time,
            taylor_adaptive,
            core,
        )
        from sys import getrefcount
        import numpy as np

        x, v = make_vars("x", "v")

        orig_sys = [(x, v), (v, cos(time) - par[0] * v - sin(x))]

        vsys = var_ode_sys(orig_sys, var_args.vars, order=2)

        ta = taylor_adaptive(vsys, [0.2, 0.3], pars=[0.4], time=0.5, compact_mode=True)

        self.assertTrue(ta.dim > 2)
        self.assertEqual(ta.n_orig_sv, 2)
        self.assertTrue(ta.is_variational)
        self.assertEqual(ta.vorder, 2)
        self.assertEqual(ta.vargs, [x, v])

        # Check that the refcount increases by when accessing tstate.
        rc = getrefcount(ta)
        ts = ta.tstate
        self.assertEqual(getrefcount(ta), rc + 1)
        self.assertEqual(ts.shape, (2,))
        self.assertTrue(np.all(ts == [0.0, 0.0]))

        # Test that tstate cannot be written to.
        with self.assertRaises(ValueError) as cm:
            ta.tstate[0] = 0.5

        self.assertEqual(ta.get_vslice(order=0), slice(0, 2, None))
        self.assertEqual(ta.get_vslice(order=0, component=1), slice(1, 2, None))

        self.assertEqual(ta.get_vslice(order=1), slice(2, 6, None))
        self.assertEqual(ta.get_vslice(order=1, component=1), slice(4, 6, None))

        self.assertEqual(ta.get_mindex(0), [0, 0, 0])
        self.assertEqual(ta.get_mindex(1), [1, 0, 0])
        self.assertEqual(ta.get_mindex(2), [0, 1, 0])
        self.assertEqual(ta.get_mindex(3), [0, 0, 1])
        self.assertEqual(ta.get_mindex(4), [1, 1, 0])
        self.assertEqual(ta.get_mindex(i=5), [1, 0, 1])

        ta.propagate_until(3.0)

        ts2 = ta.eval_taylor_map([0.0, 0.0])
        self.assertEqual(getrefcount(ta), rc + 2)
        self.assertTrue(np.shares_memory(ts, ts2))
        self.assertTrue(np.all(ts2 == ta.state[:2]))

        ts2 = ta.eval_taylor_map(np.array([0.0, 0.0]))
        self.assertTrue(np.shares_memory(ts, ts2))
        self.assertTrue(np.all(ts2 == ta.state[:2]))

        with self.assertRaises(TypeError) as cm:
            ta.eval_taylor_map(np.array([0.0, 0.0], dtype=np.int32))
        self.assertTrue(
            "Invalid dtype detected for the inputs of a Taylor map evaluation:"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            ta.eval_taylor_map(np.array([0.0, 0.0, 0.0, 0.0])[::2])
        self.assertTrue(
            "Invalid inputs array detected in a Taylor map evaluation: the array is not"
            " C-style contiguous, please "
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            ta.eval_taylor_map(np.array([[0.0, 0.0], [0.0, 0.0]]))
        self.assertTrue(
            "The array of inputs provided for the evaluation of a Taylor map has 2"
            " dimensions, "
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            ta.eval_taylor_map(np.array([0.0, 0.0, 0.0, 0.0]))
        self.assertTrue(
            "The array of inputs provided for the evaluation of a Taylor map has 4"
            " elements, "
            in str(cm.exception)
        )

        if not hasattr(core, "real"):
            return

        from . import real

        prec = 14

        ta = taylor_adaptive(
            vsys,
            [real(0.2, prec), real(0.3, prec)],
            pars=[real(0.4, prec)],
            time=real(0.5, prec),
            compact_mode=True,
            fp_type=real,
        )

        ta.propagate_until(real(3.0, prec))

        with self.assertRaises(ValueError):
            ta.eval_taylor_map(np.empty((2,), dtype=real))

        with self.assertRaises(ValueError):
            ta.eval_taylor_map(np.array([real(0.0, prec), real(0.0, prec - 1)]))

        ts2 = ta.eval_taylor_map(np.array([real(0.0, prec), real(0.0, prec)]))
        self.assertTrue(np.all(ts2 == ta.state[:2]))

    def test_batch(self):
        from . import (
            make_vars,
            var_ode_sys,
            var_args,
            cos,
            sin,
            par,
            time,
            taylor_adaptive_batch,
            core,
        )
        from sys import getrefcount
        import numpy as np

        x, v = make_vars("x", "v")

        orig_sys = [(x, v), (v, cos(time) - par[0] * v - sin(x))]

        vsys = var_ode_sys(orig_sys, var_args.vars, order=2)

        ta = taylor_adaptive_batch(
            vsys,
            [[0.2, 0.21], [0.3, 0.31]],
            pars=[[0.4, 0.41]],
            time=[0.5, 0.51],
            compact_mode=True,
        )

        self.assertTrue(ta.dim > 2)
        self.assertEqual(ta.n_orig_sv, 2)
        self.assertTrue(ta.is_variational)
        self.assertEqual(ta.vorder, 2)
        self.assertEqual(ta.vargs, [x, v])

        # Check that the refcount increases by when accessing tstate.
        rc = getrefcount(ta)
        ts = ta.tstate
        self.assertEqual(getrefcount(ta), rc + 1)
        self.assertEqual(ts.shape, (2, 2))
        self.assertTrue(np.all(ts == [[0.0, 0.0], [0.0, 0.0]]))

        # Test that tstate cannot be written to.
        with self.assertRaises(ValueError) as cm:
            ta.tstate[0] = 0.5

        self.assertEqual(ta.get_vslice(order=0), slice(0, 2, None))
        self.assertEqual(ta.get_vslice(order=0, component=1), slice(1, 2, None))

        self.assertEqual(ta.get_vslice(order=1), slice(2, 6, None))
        self.assertEqual(ta.get_vslice(order=1, component=1), slice(4, 6, None))

        self.assertEqual(ta.get_mindex(0), [0, 0, 0])
        self.assertEqual(ta.get_mindex(1), [1, 0, 0])
        self.assertEqual(ta.get_mindex(2), [0, 1, 0])
        self.assertEqual(ta.get_mindex(3), [0, 0, 1])
        self.assertEqual(ta.get_mindex(4), [1, 1, 0])
        self.assertEqual(ta.get_mindex(i=5), [1, 0, 1])

        ta.propagate_until(3.0)

        ts2 = ta.eval_taylor_map([[0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(getrefcount(ta), rc + 2)
        self.assertTrue(np.shares_memory(ts, ts2))
        self.assertTrue(np.all(ts2 == ta.state[:2]))

        ts2 = ta.eval_taylor_map(np.array([[0.0, 0.0], [0.0, 0.0]]))
        self.assertTrue(np.shares_memory(ts, ts2))
        self.assertTrue(np.all(ts2 == ta.state[:2]))

        with self.assertRaises(TypeError) as cm:
            ta.eval_taylor_map(np.array([0.0, 0.0], dtype=np.int32))
        self.assertTrue(
            "Invalid dtype detected for the inputs of a Taylor map evaluation:"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            ta.eval_taylor_map(np.array([0.0, 0.0, 0.0, 0.0])[::2])
        self.assertTrue(
            "Invalid inputs array detected in a Taylor map evaluation: the array is not"
            " C-style contiguous, please "
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            ta.eval_taylor_map(np.array([0.0, 0.0]))
        self.assertTrue(
            "The array of inputs provided for the evaluation of a Taylor map has 1"
            " dimension(s), "
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            ta.eval_taylor_map(np.array([[0.0, 0.0]]))
        self.assertTrue(
            "The array of inputs provided for the evaluation of a Taylor map has 1"
            " row(s), but it must have 2 row(s) instead"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            ta.eval_taylor_map(np.array([[0.0], [0.0]]))
        self.assertTrue(
            "The array of inputs provided for the evaluation of a Taylor map has 1"
            " column(s), but it must have 2 column(s) instead"
            in str(cm.exception)
        )

    def test_size_check_bug(self):
        # BUG: wrong size check on the input for a Taylor map (the size is checked against
        # the number of original state variables instead of the number of variational arguments).

        from . import (
            make_vars,
            var_ode_sys,
            var_args,
            cos,
            sin,
            par,
            time,
            taylor_adaptive_batch,
            taylor_adaptive,
            core,
        )
        import numpy as np

        x, v = make_vars("x", "v")

        orig_sys = [(x, v), (v, cos(time) - par[0] * v - sin(x))]

        vsys = var_ode_sys(orig_sys, var_args.vars | var_args.params, order=2)

        ta = taylor_adaptive(
            vsys,
            [0.2, 0.3],
            pars=[0.4],
            time=0.5,
            compact_mode=True,
        )

        # This would throw.
        self.assertTrue(np.all(ta.state[:2] == ta.eval_taylor_map([0.0, 0.0, 0.0])))

        # Test the batch case too.
        ta = taylor_adaptive_batch(
            vsys,
            [[0.2, 0.21], [0.3, 0.31]],
            pars=[[0.4, 0.41]],
            time=[0.5, 0.51],
            compact_mode=True,
        )

        # This would throw.
        self.assertTrue(
            np.all(
                ta.state[:2] == ta.eval_taylor_map([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            )
        )
