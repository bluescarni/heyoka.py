# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class dtens_test_case(_ut.TestCase):
    def test_gradient(self):
        from . import diff_tensors, make_vars, expression as ex

        x, y = make_vars("x", "y")

        dt = diff_tensors([x - y], [x, y])
        self.assertEqual(dt.gradient, [ex(1.0), ex(-1.0)])

    def test_hessian(self):
        from . import diff_tensors, make_vars, expression as ex, cfunc
        import numpy as np

        x, y, z = make_vars("x", "y", "z")

        dt = diff_tensors([x * y**2 * z**3], [x, y, z], diff_order=2)

        h = dt.hessian(component=0)

        cf = cfunc(h[np.triu_indices(3)].flatten(), [x, y, z])

        xval, yval, zval = 1.0, 2.0, 3.0

        out = cf([xval, yval, zval])

        self.assertEqual(len(out), 6)
        self.assertEqual(out[0], 0.0)
        self.assertEqual(out[1], 2 * yval * zval * zval * zval)
        self.assertEqual(out[2], 3 * yval * yval * zval * zval)
        self.assertEqual(out[3], 2 * xval * zval * zval * zval)
        self.assertEqual(out[4], 6 * xval * yval * zval * zval)
        self.assertEqual(out[5], 6 * xval * yval * yval * zval)

    def test_jacobian(self):
        from . import diff_tensors, make_vars, expression as ex, diff_args
        import numpy as np

        x, y = make_vars("x", "y")

        dt = diff_tensors([x - y, -x + y], [x, y])
        self.assertTrue(
            np.all(dt.jacobian == np.array([[ex(1.0), ex(-1.0)], [ex(-1.0), ex(1.0)]]))
        )

        dt = diff_tensors([x - y, -x + y, -x - y], diff_args.vars)
        self.assertTrue(
            np.all(
                dt.jacobian
                == np.array(
                    [[ex(1.0), ex(-1.0)], [ex(-1.0), ex(1.0)], [ex(-1.0), ex(-1.0)]]
                )
            )
        )

    def test_basic(self):
        from . import dtens
        from sys import getrefcount
        import gc
        import pickle
        from copy import copy, deepcopy

        # Default construction.
        dt = dtens()
        self.assertEqual(len(dt), 0)
        self.assertEqual(dt.args, [])
        self.assertEqual(dt.nouts, 0)
        self.assertEqual(dt.nargs, 0)
        self.assertEqual(dt.order, 0)
        with self.assertRaises(KeyError) as cm:
            dt[1, 2, 3]
        self.assertTrue(
            "Cannot locate the derivative corresponding the the vector of indices"
            in str(cm.exception)
        )
        with self.assertRaises(KeyError) as cm:
            dt[1, [(0, 2), (1, 3)]]
        self.assertTrue(
            "Cannot locate the derivative corresponding the the vector of indices"
            in str(cm.exception)
        )
        with self.assertRaises(IndexError) as cm:
            dt[1]
        self.assertTrue(
            "The derivative at index 1 was requested, but the total number of"
            " derivatives is 0"
            in str(cm.exception)
        )
        self.assertFalse([1, 2, 3] in dt)
        self.assertFalse((1, [(0, 2), (1, 3)]) in dt)

        rc = getrefcount(dt)
        it = dt.__iter__()
        self.assertEqual(rc + 1, getrefcount(dt))
        del it
        gc.collect()
        self.assertEqual(rc, getrefcount(dt))

        self.assertEqual(list(dt), [])

        self.assertEqual(dt.index_of(vidx=[1, 2, 3]), 0)
        self.assertEqual(dt.index_of(vidx=(1, [(0, 2), (1, 3)])), 0)

        self.assertEqual(dt.get_derivatives(0), [])
        self.assertEqual(dt.get_derivatives(1), [])
        self.assertEqual(dt.get_derivatives(0, 0), [])
        self.assertEqual(dt.get_derivatives(0, 1), [])
        self.assertEqual(dt.get_derivatives(1, 0), [])
        self.assertEqual(dt.get_derivatives(1, 1), [])

        dt2 = pickle.loads(pickle.dumps(dt))
        self.assertEqual(len(dt2), 0)
        self.assertEqual(dt2.args, [])
        self.assertEqual(dt2.nouts, 0)
        self.assertEqual(dt2.nargs, 0)
        self.assertEqual(dt2.order, 0)

        dt2 = copy(dt)
        self.assertEqual(len(dt2), 0)
        self.assertEqual(dt2.args, [])
        self.assertEqual(dt2.nouts, 0)
        self.assertEqual(dt2.nargs, 0)
        self.assertEqual(dt2.order, 0)

        dt2 = deepcopy(dt)
        self.assertEqual(len(dt2), 0)
        self.assertEqual(dt2.args, [])
        self.assertEqual(dt2.nouts, 0)
        self.assertEqual(dt2.nargs, 0)
        self.assertEqual(dt2.order, 0)

    def test_diff_tensors(self):
        from . import diff_tensors, make_vars, expression, par, diff_args
        from copy import copy, deepcopy
        import pickle

        x, y = make_vars("x", "y")

        dt = diff_tensors([x + y], [x, y])
        self.assertEqual(len(dt), 3)
        self.assertEqual(dt.args, [x, y])
        self.assertEqual(dt.nouts, 1)
        self.assertEqual(dt.nargs, 2)
        self.assertEqual(dt.order, 1)
        self.assertEqual(dt[0, 0, 0], x + y)
        self.assertEqual(dt[1], ([0, 1, 0], expression(1.0)))
        self.assertEqual(dt[2], ([0, 0, 1], expression(1.0)))
        self.assertFalse([1, 2, 3] in dt)
        self.assertTrue([0, 1, 0] in dt)
        self.assertEqual(list(dt), [[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(dt.index_of(vidx=[1, 2, 3]), 3)
        self.assertEqual(dt.index_of(vidx=[0, 0, 1]), 2)
        self.assertEqual(dt.get_derivatives(0), [([0, 0, 0], (x + y))])
        self.assertEqual(
            dt.get_derivatives(1),
            [([0, 1, 0], expression(1.0)), ([0, 0, 1], expression(1.0))],
        )
        self.assertEqual(dt.get_derivatives(0, 0), [([0, 0, 0], (x + y))])
        self.assertEqual(
            dt.get_derivatives(1, 0),
            [([0, 1, 0], expression(1.0)), ([0, 0, 1], expression(1.0))],
        )
        self.assertEqual(
            dt.get_derivatives(1, 1),
            [],
        )

        dt = diff_tensors([x + y], diff_args=[x], diff_order=2)
        self.assertEqual(len(dt), 3)
        self.assertEqual(dt.args, [x])
        self.assertEqual(dt.nouts, 1)
        self.assertEqual(dt.nargs, 1)
        self.assertEqual(dt.order, 2)
        self.assertEqual(dt[0, 0], x + y)
        self.assertEqual(dt[0, 1], expression(1.0))
        self.assertEqual(dt[0, 2], expression(0.0))

        dt = diff_tensors(
            [x + y + 2.0 * par[0]], diff_args=diff_args.params, diff_order=2
        )
        self.assertEqual(len(dt), 3)
        self.assertEqual(dt.args, [par[0]])
        self.assertEqual(dt.nouts, 1)
        self.assertEqual(dt.nargs, 1)
        self.assertEqual(dt.order, 2)
        self.assertEqual(dt[0, 0], x + y + 2.0 * par[0])
        self.assertEqual(dt[0, 1], expression(2.0))
        self.assertEqual(dt[0, 2], expression(0.0))

        dt = diff_tensors([x + y + 2.0 * par[0]], diff_args=diff_args.all, diff_order=2)
        self.assertEqual(len(dt), 10)
        self.assertEqual(dt.args, [x, y, par[0]])
        self.assertEqual(dt.nouts, 1)
        self.assertEqual(dt.nargs, 3)
        self.assertEqual(dt.order, 2)
        self.assertEqual(dt[0, 0, 0, 0], x + y + 2.0 * par[0])
        self.assertEqual(dt[0, 1, 0, 0], expression(1.0))
        self.assertEqual(dt[0, 0, 1, 0], expression(1.0))
        self.assertEqual(dt[0, 0, 0, 1], expression(2.0))
        self.assertEqual(dt[0, 2, 0, 0], expression(0.0))
        self.assertEqual(dt[0, 1, 1, 0], expression(0.0))
        self.assertEqual(dt[0, 0, 1, 1], expression(0.0))

        dt2 = pickle.loads(pickle.dumps(dt))
        self.assertEqual(len(dt), 10)
        self.assertEqual(dt.args, [x, y, par[0]])
        self.assertEqual(dt.nouts, 1)
        self.assertEqual(dt.nargs, 3)
        self.assertEqual(dt.order, 2)
        self.assertEqual(dt[0, 0, 0, 0], x + y + 2.0 * par[0])
        self.assertEqual(dt[0, 1, 0, 0], expression(1.0))
        self.assertEqual(dt[0, 0, 1, 0], expression(1.0))
        self.assertEqual(dt[0, 0, 0, 1], expression(2.0))
        self.assertEqual(dt[0, 2, 0, 0], expression(0.0))
        self.assertEqual(dt[0, 1, 1, 0], expression(0.0))
        self.assertEqual(dt[0, 0, 1, 1], expression(0.0))

        dt2 = copy(dt)
        self.assertEqual(len(dt), 10)
        self.assertEqual(dt.args, [x, y, par[0]])
        self.assertEqual(dt.nouts, 1)
        self.assertEqual(dt.nargs, 3)
        self.assertEqual(dt.order, 2)
        self.assertEqual(dt[0, 0, 0, 0], x + y + 2.0 * par[0])
        self.assertEqual(dt[0, 1, 0, 0], expression(1.0))
        self.assertEqual(dt[0, 0, 1, 0], expression(1.0))
        self.assertEqual(dt[0, 0, 0, 1], expression(2.0))
        self.assertEqual(dt[0, 2, 0, 0], expression(0.0))
        self.assertEqual(dt[0, 1, 1, 0], expression(0.0))
        self.assertEqual(dt[0, 0, 1, 1], expression(0.0))

        dt.foo = [1, 2, 3, 4]
        dt2 = deepcopy(dt)
        self.assertEqual(len(dt), 10)
        self.assertEqual(dt.args, [x, y, par[0]])
        self.assertEqual(dt.nouts, 1)
        self.assertEqual(dt.nargs, 3)
        self.assertEqual(dt.order, 2)
        self.assertEqual(dt[0, 0, 0, 0], x + y + 2.0 * par[0])
        self.assertEqual(dt[0, 1, 0, 0], expression(1.0))
        self.assertEqual(dt[0, 0, 1, 0], expression(1.0))
        self.assertEqual(dt[0, 0, 0, 1], expression(2.0))
        self.assertEqual(dt[0, 2, 0, 0], expression(0.0))
        self.assertEqual(dt[0, 1, 1, 0], expression(0.0))
        self.assertEqual(dt[0, 0, 1, 1], expression(0.0))
        self.assertEqual(dt2.foo, [1, 2, 3, 4])
