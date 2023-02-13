# Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut

# Small helper to get the epsilon of a floating-point type.


def _get_eps(fp_t):
    import numpy as np

    if fp_t == float or fp_t == np.longdouble:
        return np.finfo(fp_t).eps

    from . import core

    if hasattr(core, "real128"):
        return core._get_real128_eps()

    raise TypeError(
        'Cannot compute the epsilon of the floating-point type "{}"'.format(fp_t)
    )


# Reimplementation of several NumPy functions which do not work correctly with real128.


def _isclose(a, b, rtol, atol):
    from numpy import errstate, less_equal, asanyarray, isfinite, zeros_like, ones_like

    def within_tol(x, y, atol, rtol):
        with errstate(invalid="ignore"):
            return less_equal(abs(x - y), atol + rtol * abs(y))

    x = asanyarray(a)
    y = asanyarray(b)

    xfin = isfinite(x)
    yfin = isfinite(y)
    if all(xfin) and all(yfin):
        return within_tol(x, y, atol, rtol)
    else:
        finite = xfin & yfin
        cond = zeros_like(finite, subok=True)
        # Because we're using boolean indexing, x & y must be the same shape.
        # Ideally, we'd just do x, y = broadcast_arrays(x, y). It's in
        # lib.stride_tricks, though, so we can't import it here.
        x = x * ones_like(cond)
        y = y * ones_like(cond)
        # Avoid subtraction with infinite/nan values...
        cond[finite] = within_tol(x[finite], y[finite], atol, rtol)
        # Check for equality of infinite values...
        cond[~finite] = x[~finite] == y[~finite]

        return cond[()]  # Flatten 0d arrays to scalars


def _allclose(a, b, rtol, atol):
    res = all(_isclose(a, b, rtol=rtol, atol=atol))
    return bool(res)


class taylor_add_jet_test_case(_ut.TestCase):
    def test_basic(self):
        from . import (
            taylor_add_jet,
            make_vars,
            sin,
            taylor_adaptive,
            par,
            time,
            taylor_adaptive_batch,
            tpoly,
            core,
        )
        from .core import _ppc_arch
        import numpy as np

        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]
        sys_par = [(x, v), (v, -par[0] * sin(x))]
        sys_par_t = [(x, v), (v, -par[0] * sin(x) + time)]
        sys_par_t2 = [(x, v), (v, -par[0] * sin(x) + tpoly(par[1], par[6]))]

        if _ppc_arch:
            fp_types = [float]
        else:
            fp_types = [float, np.longdouble]

        if hasattr(core, "real128"):
            fp_types.append(core.real128)

        for fp_t in fp_types:
            # Check that the jet is consistent
            # with the Taylor coefficients.
            init_state = [fp_t(0.05), fp_t(0.025)]
            pars = [fp_t(-9.8)]
            pars2 = [
                fp_t(-9.8),
                fp_t(0.01),
                fp_t(0.02),
                fp_t(0.03),
                fp_t(0.04),
                fp_t(0.05),
                fp_t(0.06),
            ]

            ta = taylor_adaptive(sys, init_state, tol=fp_t(1e-9), fp_type=fp_t)

            jet = taylor_add_jet(sys, 5, fp_type=fp_t)
            st = np.full((6, 2), fp_t(0), dtype=fp_t)
            st[0] = init_state

            ta.step(write_tc=True)
            jet(st)

            self.assertTrue(np.all(ta.tc[:, :6].transpose() == st))

            # Try adding an sv_func.
            jet = taylor_add_jet(sys, 5, fp_type=fp_t, sv_funcs=[x + v])
            st = np.full((6, 3), fp_t(0), dtype=fp_t)
            st[0, :2] = init_state

            jet(st)

            self.assertTrue(np.all(ta.tc[:, :6].transpose() == st[:, :2]))
            self.assertTrue(
                np.all((ta.tc[0, :6] + ta.tc[1, :6]).transpose() == st[:, 2])
            )

            # An example with params.
            ta_par = taylor_adaptive(
                sys_par, init_state, tol=fp_t(1e-9), fp_type=fp_t, pars=pars
            )

            jet_par = taylor_add_jet(sys_par, 5, fp_type=fp_t)
            st = np.full((6, 2), fp_t(0), dtype=fp_t)
            st[0] = init_state
            par_arr = np.full((1,), fp_t(-9.8), dtype=fp_t)

            ta_par.step(write_tc=True)
            jet_par(st, pars=par_arr)

            self.assertTrue(np.all(ta_par.tc[:, :6].transpose() == st))

            # Params + time.
            ta_par_t = taylor_adaptive(
                sys_par_t, init_state, tol=fp_t(1e-9), fp_type=fp_t, pars=pars
            )
            ta_par_t.time = fp_t(0.01)

            jet_par_t = taylor_add_jet(sys_par_t, 5, fp_type=fp_t)
            st = np.full((6, 2), fp_t(0), dtype=fp_t)
            st[0] = init_state
            par_arr = np.full((1,), fp_t(-9.8), dtype=fp_t)
            time_arr = np.full((1,), fp_t(0.01), dtype=fp_t)

            ta_par_t.step(write_tc=True)
            jet_par_t(st, pars=par_arr, time=time_arr)

            self.assertTrue(np.all(ta_par_t.tc[:, :6].transpose() == st))

            ta_par_t2 = taylor_adaptive(
                sys_par_t2, init_state, tol=fp_t(1e-9), fp_type=fp_t, pars=pars2
            )
            ta_par_t2.time = fp_t(0.01)

            jet_par_t2 = taylor_add_jet(sys_par_t2, 5, fp_type=fp_t)
            st = np.full((6, 2), fp_t(0), dtype=fp_t)
            st[0] = init_state
            par_arr2 = np.array(pars2, dtype=fp_t)
            time_arr = np.full((1,), fp_t(0.01), dtype=fp_t)

            ta_par_t2.step(write_tc=True)
            jet_par_t2(st, pars=par_arr2, time=time_arr)

            self.assertTrue(np.all(ta_par_t2.tc[:, :6].transpose() == st))

            # Failure modes.

            # Non-contiguous state.
            with self.assertRaises(ValueError) as cm:
                jet(st[::2])
            self.assertTrue(
                "Invalid state vector passed to a function for the computation of the jet of Taylor derivatives: the NumPy array is not C contiguous or not writeable"
                in str(cm.exception)
            )

            # Non-writeable state.
            st.flags.writeable = False
            with self.assertRaises(ValueError) as cm:
                jet(st)
            self.assertTrue(
                "Invalid state vector passed to a function for the computation of the jet of Taylor derivatives: the NumPy array is not C contiguous or not writeable"
                in str(cm.exception)
            )

            # Non-contiguous pars.
            st = np.full((6, 2), fp_t(0), dtype=fp_t)
            par_arr = np.full((5,), fp_t(-9.8), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=par_arr[::2])
            self.assertTrue(
                "Invalid parameters vector passed to a function for the computation of the jet of Taylor derivatives: the NumPy array is not C contiguous"
                in str(cm.exception)
            )

            # Non-contiguous time.
            time_arr = np.full((5,), fp_t(0.01), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=par_arr, time=time_arr[::2])
            self.assertTrue(
                "Invalid time vector passed to a function for the computation of the jet of Taylor derivatives: the NumPy array is not C contiguous"
                in str(cm.exception)
            )

            # Overlapping arrays.
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=st, time=time_arr)
            self.assertTrue(
                "Invalid vectors passed to a function for the computation of the jet of Taylor derivatives: the NumPy arrays must not share any memory"
                in str(cm.exception)
            )
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=par_arr, time=st)
            self.assertTrue(
                "Invalid vectors passed to a function for the computation of the jet of Taylor derivatives: the NumPy arrays must not share any memory"
                in str(cm.exception)
            )
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=st, time=st)
            self.assertTrue(
                "Invalid vectors passed to a function for the computation of the jet of Taylor derivatives: the NumPy arrays must not share any memory"
                in str(cm.exception)
            )

            # Params needed but not provided.
            with self.assertRaises(ValueError) as cm:
                jet_par(st)
            self.assertTrue(
                "Invalid vectors passed to a function for the computation of the jet of "
                "Taylor derivatives: the ODE system contains parameters, but no parameter array was "
                "passed as input argument" in str(cm.exception)
            )

            # Time needed but not provided.
            with self.assertRaises(ValueError) as cm:
                jet_par_t(st, pars=par_arr)
            self.assertTrue(
                "Invalid vectors passed to a function for the computation of the jet of "
                "Taylor derivatives: the ODE system is non-autonomous, but no time array was "
                "passed as input argument" in str(cm.exception)
            )
            with self.assertRaises(ValueError) as cm:
                jet_par_t2(st, pars=par_arr2)
            self.assertTrue(
                "Invalid vectors passed to a function for the computation of the jet of "
                "Taylor derivatives: the ODE system is non-autonomous, but no time array was "
                "passed as input argument" in str(cm.exception)
            )

            # Wrong st shape, scalar case.
            st = np.full((6, 2, 1), fp_t(0), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet(st)
            self.assertTrue(
                "Invalid state vector passed to a function for the computation of the jet of "
                "Taylor derivatives: the number of dimensions must be 2, but it is "
                "3 instead" in str(cm.exception)
            )

            st = np.full((6, 4), fp_t(0), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet(st)
            self.assertTrue(
                "Invalid state vector passed to a function for the computation of the jet of "
                "Taylor derivatives: the shape must be (6, 3), but it is "
                "(6, 4) instead" in str(cm.exception)
            )

            # Wrong param shape, scalar case.
            st = np.full((6, 2), fp_t(0), dtype=fp_t)
            par_arr = np.full((5, 2), fp_t(-9.8), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=par_arr)
            self.assertTrue(
                "Invalid parameters vector passed to a function for the computation of "
                "the jet of "
                "Taylor derivatives: the number of dimensions must be 1, but it is "
                "2 instead" in str(cm.exception)
            )

            par_arr = np.full((5,), fp_t(-9.8), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=par_arr)
            self.assertTrue(
                "Invalid parameters vector passed to a function for the "
                "computation of the jet of "
                "Taylor derivatives: the shape must be (1, ), but it is "
                "(5) instead" in str(cm.exception)
            )

            # Wrong time shape, scalar case.
            par_arr = np.full((1,), fp_t(-9.8), dtype=fp_t)
            time_arr = np.full((2, 1), fp_t(0.01), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par_t(st, pars=par_arr, time=time_arr)
            self.assertTrue(
                "Invalid time vector passed to a function for the computation of the jet of "
                "Taylor derivatives: the number of dimensions must be 1, but it is "
                "2 instead" in str(cm.exception)
            )

            time_arr = np.full((5,), fp_t(0.01), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par_t(st, pars=par_arr, time=time_arr)
            self.assertTrue(
                "Invalid time vector passed to a function for the computation of the jet of "
                "Taylor derivatives: the shape must be (1, ), but it is "
                "(5) instead" in str(cm.exception)
            )

            # No pars in the system, wrong par array provided, scalar case.
            jet = taylor_add_jet(sys, 5, fp_type=fp_t)
            st = np.full((6, 2), fp_t(0), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet(st, pars=np.zeros((1,), dtype=fp_t))
            self.assertTrue(
                "Invalid parameters vector passed to a function for the computation of the jet of Taylor derivatives: the shape must be (0, ), but it is (1) instead"
                in str(cm.exception)
            )

            # No time in the system, wrong time array provided, scalar case.
            with self.assertRaises(ValueError) as cm:
                jet(st, time=np.zeros((0,), dtype=fp_t))
            self.assertTrue(
                "Invalid time vector passed to a function for the computation of the jet of Taylor derivatives: the shape must be (1, ), but it is (0) instead"
                in str(cm.exception)
            )

        # Check throwing behaviour with long double on PPC.
        if _ppc_arch:
            with self.assertRaises(NotImplementedError):
                taylor_add_jet(sys, 5, fp_type=np.longdouble)

        # Batch mode testing.
        fp_types = [float]

        for fp_t in fp_types:
            batch_size = 4

            # Check that the jet is consistent
            # with the Taylor coefficients.
            init_state = [
                [fp_t(0.05), fp_t(0.06), fp_t(0.07), fp_t(0.08)],
                [fp_t(0.025), fp_t(0.026), fp_t(0.027), fp_t(0.028)],
            ]
            pars = [[fp_t(-9.8), fp_t(-9.7), fp_t(-9.6), fp_t(-9.5)]]

            ta = taylor_adaptive_batch(sys, init_state, tol=fp_t(1e-9), fp_type=fp_t)

            jet = taylor_add_jet(sys, 5, fp_type=fp_t, batch_size=batch_size)
            st = np.full((6, 2, batch_size), fp_t(0), dtype=fp_t)
            st[0] = init_state

            ta.step(write_tc=True)
            jet(st)

            self.assertTrue(np.all(ta.tc[:, :6, :].transpose((1, 0, 2)) == st))

            # Try adding an sv_func.
            jet = taylor_add_jet(
                sys, 5, fp_type=fp_t, sv_funcs=[x + v], batch_size=batch_size
            )
            st = np.full((6, 3, batch_size), fp_t(0), dtype=fp_t)
            st[0, :2] = init_state

            jet(st)

            self.assertTrue(np.all(ta.tc[:, :6, :].transpose((1, 0, 2)) == st[:, :2]))
            self.assertTrue(np.all((ta.tc[0, :6, :] + ta.tc[1, :6, :]) == st[:, 2, :]))

            # An example with params.
            ta_par = taylor_adaptive_batch(
                sys_par, init_state, tol=fp_t(1e-9), fp_type=fp_t, pars=pars
            )

            jet_par = taylor_add_jet(sys_par, 5, fp_type=fp_t, batch_size=batch_size)
            st = np.full((6, 2, batch_size), fp_t(0), dtype=fp_t)
            st[0] = init_state
            par_arr = np.array(pars)

            ta_par.step(write_tc=True)
            jet_par(st, pars=par_arr)

            self.assertTrue(np.all(ta_par.tc[:, :6, :].transpose((1, 0, 2)) == st))

            # Params + time.
            ta_par_t = taylor_adaptive_batch(
                sys_par_t, init_state, tol=fp_t(1e-9), fp_type=fp_t, pars=pars
            )
            ta_par_t.set_time([fp_t(0.01), fp_t(0.02), fp_t(0.03), fp_t(0.04)])

            jet_par_t = taylor_add_jet(
                sys_par_t, 5, fp_type=fp_t, batch_size=batch_size
            )
            st = np.full((6, 2, batch_size), fp_t(0), dtype=fp_t)
            st[0] = init_state
            par_arr = np.array(pars)
            time_arr = np.array([fp_t(0.01), fp_t(0.02), fp_t(0.03), fp_t(0.04)])

            ta_par_t.step(write_tc=True)
            jet_par_t(st, pars=par_arr, time=time_arr)

            self.assertTrue(np.all(ta_par_t.tc[:, :6, :].transpose((1, 0, 2)) == st))

            # Just do shape/dims checks for the batch case.

            # Wrong st shape, batch case.
            st = np.full((6, 2, 1, 1), fp_t(0), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet(st)
            self.assertTrue(
                "Invalid state vector passed to a function for the computation of the jet of "
                "Taylor derivatives: the number of dimensions must be 3, but it is "
                "4 instead" in str(cm.exception)
            )

            st = np.full((6, 4, 4), fp_t(0), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet(st)
            self.assertTrue(
                "Invalid state vector passed to a function for the computation of the jet of "
                "Taylor derivatives: the shape must be (6, 3, 4), but it is "
                "(6, 4, 4) instead" in str(cm.exception)
            )

            # Wrong param shape, batch case.
            st = np.full((6, 2, 4), fp_t(0), dtype=fp_t)
            par_arr = np.full((5, 2, 1), fp_t(-9.8), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=par_arr)
            self.assertTrue(
                "Invalid parameters vector passed to a function for the computation of "
                "the jet of "
                "Taylor derivatives: the number of dimensions must be 2, but it is "
                "3 instead" in str(cm.exception)
            )

            par_arr = np.full((5, 1), fp_t(-9.8), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=par_arr)
            self.assertTrue(
                "Invalid parameters vector passed to a function for the "
                "computation of the jet of "
                "Taylor derivatives: the shape must be (1, 4), but it is "
                "(5, 1) instead" in str(cm.exception)
            )

            # Wrong time shape, batch case.
            par_arr = np.full((1, 4), fp_t(-9.8), dtype=fp_t)
            time_arr = np.full((2, 1), fp_t(0.01), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par_t(st, pars=par_arr, time=time_arr)
            self.assertTrue(
                "Invalid time vector passed to a function for the computation of the jet of "
                "Taylor derivatives: the number of dimensions must be 1, but it is "
                "2 instead" in str(cm.exception)
            )

            time_arr = np.full((5,), fp_t(0.01), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par_t(st, pars=par_arr, time=time_arr)
            self.assertTrue(
                "Invalid time vector passed to a function for the computation of the jet of "
                "Taylor derivatives: the shape must be (4, ), but it is "
                "(5) instead" in str(cm.exception)
            )

            # No pars in the system, wrong par array provided, batch case.
            jet = taylor_add_jet(sys, 5, fp_type=fp_t, batch_size=batch_size)
            st = np.full((6, 2, batch_size), fp_t(0), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet(st, pars=np.zeros((1, 4), dtype=fp_t))
            self.assertTrue(
                "Invalid parameters vector passed to a function for the computation of the jet of Taylor derivatives: the shape must be (0, 4), but it is (1, 4) instead"
                in str(cm.exception)
            )

            # No time in the system, wrong time array provided, batch case.
            with self.assertRaises(ValueError) as cm:
                jet(st, time=np.zeros((0,), dtype=fp_t))
            self.assertTrue(
                "Invalid time vector passed to a function for the computation of the jet of Taylor derivatives: the shape must be (4, ), but it is (0) instead"
                in str(cm.exception)
            )

        # Test that batch mode with long double is not allowed.
        fp_t = np.longdouble

        init_state = [
            [fp_t(0.05), fp_t(0.06), fp_t(0.07), fp_t(0.08)],
            [fp_t(0.025), fp_t(0.026), fp_t(0.027), fp_t(0.028)],
        ]
        pars = [[fp_t(-9.8), fp_t(-9.7), fp_t(-9.6), fp_t(-9.5)]]

        with self.assertRaises(ValueError) as cm:
            taylor_add_jet(sys, 5, fp_type=fp_t, batch_size=2)
        self.assertTrue(
            "Batch sizes greater than 1 are not supported for this floating-point type"
            in str(cm.exception)
        )


class event_classes_test_case(_ut.TestCase):
    def test_basic(self):
        from . import (
            t_event,
            nt_event,
            t_event_batch,
            nt_event_batch,
            make_vars,
            event_direction,
            core,
        )
        from .core import _ppc_arch
        import numpy as np
        import pickle
        import gc
        from copy import copy, deepcopy

        x, v = make_vars("x", "v")

        if _ppc_arch:
            fp_types = [float]
        else:
            fp_types = [float, np.longdouble]

        if hasattr(core, "real128"):
            fp_types.append(core.real128)

        for fp_t in fp_types:
            # Non-terminal event.
            ev = nt_event(x + v, lambda _: _, fp_type=fp_t)

            self.assertTrue(" non-terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::any" in repr(ev))
            self.assertEqual(ev.expression, x + v)
            self.assertEqual(ev.direction, event_direction.any)
            self.assertFalse(ev.callback is None)

            ev = nt_event(ex=x + v, callback=lambda _: _, fp_type=fp_t)
            self.assertTrue(" non-terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::any" in repr(ev))
            self.assertEqual(ev.expression, x + v)
            self.assertEqual(ev.direction, event_direction.any)
            self.assertFalse(ev.callback is None)

            ev = nt_event(
                ex=x + v,
                callback=lambda _: _,
                direction=event_direction.positive,
                fp_type=fp_t,
            )
            self.assertTrue(" non-terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::positive" in repr(ev))
            self.assertEqual(ev.expression, x + v)
            self.assertEqual(ev.direction, event_direction.positive)
            self.assertFalse(ev.callback is None)

            ev = nt_event(
                ex=x + v,
                callback=lambda _: _,
                direction=event_direction.negative,
                fp_type=fp_t,
            )
            self.assertTrue(" non-terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::negative" in repr(ev))
            self.assertEqual(ev.expression, x + v)
            self.assertEqual(ev.direction, event_direction.negative)
            self.assertFalse(ev.callback is None)

            class local_cb:
                def __init__(self):
                    self.n = 0

                def __call__(self, ta, t, d_sgn):
                    self.n = self.n + 1

            lcb = local_cb()
            ev = nt_event(
                ex=x + v, callback=lcb, direction=event_direction.negative, fp_type=fp_t
            )
            self.assertEqual(ev.callback.n, 0)
            cb = ev.callback
            cb(1, 2, 3)
            cb(1, 2, 3)
            cb(1, 2, 3)
            self.assertEqual(ev.callback.n, 3)
            ev.callback.n = 0
            self.assertEqual(ev.callback.n, 0)
            self.assertEqual(id(lcb), id(ev.callback))

            with self.assertRaises(ValueError) as cm:
                nt_event(
                    ex=x + v,
                    callback=lambda _: _,
                    direction=event_direction(10),
                    fp_type=fp_t,
                )
            self.assertTrue(
                "Invalid value selected for the direction of a non-terminal event"
                in str(cm.exception)
            )

            with self.assertRaises(TypeError) as cm:
                nt_event(ex=x + v, callback=3, fp_type=fp_t)
            self.assertTrue(
                "An object of type '{}' cannot be used as an event callback because it is not callable".format(
                    str(type(3))
                )
                in str(cm.exception)
            )

            with self.assertRaises(TypeError) as cm:
                nt_event(ex=x + v, callback=None, fp_type=fp_t)
            self.assertTrue(
                "An object of type '{}' cannot be used as an event callback because it is not callable".format(
                    str(type(None))
                )
                in str(cm.exception)
            )

            ev = nt_event(
                ex=x + v,
                callback=lambda _: _,
                direction=event_direction.negative,
                fp_type=fp_t,
            )
            ev = pickle.loads(pickle.dumps(ev))
            self.assertTrue(" non-terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::negative" in repr(ev))

            # Test dynamic attributes.
            ev.foo = "hello world"
            ev = pickle.loads(pickle.dumps(ev))
            self.assertEqual(ev.foo, "hello world")

            # Test copy semantics.
            class foo:
                pass

            ev.bar = foo()

            self.assertEqual(id(ev.bar), id(copy(ev).bar))
            self.assertNotEqual(id(ev.bar), id(deepcopy(ev).bar))

            # Test to ensure a callback extracted from the event
            # is kept alive and usable when the event is destroyed.
            ev = nt_event(
                ex=x + v,
                callback=local_cb(),
                direction=event_direction.negative,
                fp_type=fp_t,
            )
            out_cb = ev.callback
            del ev
            gc.collect()
            out_cb(1, 2, 3)
            out_cb(1, 2, 3)
            out_cb(1, 2, 3)
            self.assertEqual(out_cb.n, 3)

            # Terminal event.
            ev = t_event(x + v, fp_type=fp_t)

            self.assertTrue(" terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::any" in repr(ev))
            self.assertTrue(": no" in repr(ev))
            self.assertTrue("auto" in repr(ev))
            self.assertEqual(ev.expression, x + v)
            self.assertEqual(ev.direction, event_direction.any)
            self.assertEqual(ev.cooldown, fp_t(-1))
            self.assertTrue(ev.callback is None)

            ev = t_event(
                x + v,
                fp_type=fp_t,
                direction=event_direction.negative,
                cooldown=fp_t(3),
            )

            self.assertTrue(" terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::negative" in repr(ev))
            self.assertTrue(": no" in repr(ev))
            self.assertTrue("3" in repr(ev))
            self.assertEqual(ev.expression, x + v)
            self.assertEqual(ev.direction, event_direction.negative)
            self.assertEqual(ev.cooldown, fp_t(3))
            self.assertTrue(ev.callback is None)

            ev = t_event(
                x + v,
                fp_type=fp_t,
                direction=event_direction.positive,
                cooldown=fp_t(3),
                callback=lambda _: _,
            )

            self.assertTrue(" terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::positive" in repr(ev))
            self.assertTrue(": yes" in repr(ev))
            self.assertTrue("3" in repr(ev))
            self.assertEqual(ev.expression, x + v)
            self.assertEqual(ev.direction, event_direction.positive)
            self.assertEqual(ev.cooldown, fp_t(3))
            self.assertFalse(ev.callback is None)

            class local_cb:
                def __init__(self):
                    self.n = 0

                def __call__(self, ta, mr, d_sgn):
                    self.n = self.n + 1

            lcb = local_cb()
            ev = t_event(
                x + v,
                fp_type=fp_t,
                direction=event_direction.positive,
                cooldown=fp_t(3),
                callback=lcb,
            )

            self.assertTrue(" terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::positive" in repr(ev))
            self.assertTrue(": yes" in repr(ev))
            self.assertTrue("3" in repr(ev))
            self.assertEqual(ev.expression, x + v)
            self.assertEqual(ev.direction, event_direction.positive)
            self.assertEqual(ev.cooldown, fp_t(3))
            self.assertFalse(ev.callback is None)
            self.assertEqual(ev.callback.n, 0)
            cb = ev.callback
            cb(1, 2, 3)
            cb(1, 2, 3)
            cb(1, 2, 3)
            self.assertEqual(ev.callback.n, 3)
            ev.callback.n = 0
            self.assertEqual(ev.callback.n, 0)
            self.assertEqual(id(lcb), id(ev.callback))

            ev = t_event(
                x + v,
                fp_type=fp_t,
                direction=event_direction.positive,
                cooldown=fp_t(3),
                callback=None,
            )
            self.assertTrue(ev.callback is None)

            with self.assertRaises(ValueError) as cm:
                t_event(
                    x + v,
                    fp_type=fp_t,
                    direction=event_direction(45),
                    cooldown=fp_t(3),
                    callback=lambda _: _,
                )
            self.assertTrue(
                "Invalid value selected for the direction of a terminal event"
                in str(cm.exception)
            )

            with self.assertRaises(TypeError) as cm:
                t_event(x + v, callback=3, fp_type=fp_t)
            self.assertTrue(
                "An object of type '{}' cannot be used as an event callback because it is not callable".format(
                    str(type(3))
                )
                in str(cm.exception)
            )

            ev = t_event(
                x + v,
                fp_type=fp_t,
                direction=event_direction.positive,
                cooldown=fp_t(3),
                callback=lambda _: _,
            )

            ev = pickle.loads(pickle.dumps(ev))
            self.assertTrue(" terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::positive" in repr(ev))
            self.assertTrue(": yes" in repr(ev))
            self.assertTrue("3" in repr(ev))

            # Test dynamic attributes.
            ev.foo = "hello world"
            ev = pickle.loads(pickle.dumps(ev))
            self.assertEqual(ev.foo, "hello world")

            # Test copy semantics.
            class foo:
                pass

            ev.bar = foo()

            self.assertEqual(id(ev.bar), id(copy(ev).bar))
            self.assertNotEqual(id(ev.bar), id(deepcopy(ev).bar))

            # Test also with empty callback.
            ev = t_event(
                x + v,
                fp_type=fp_t,
                direction=event_direction.positive,
                cooldown=fp_t(3),
            )

            ev = pickle.loads(pickle.dumps(ev))
            self.assertTrue(" terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::positive" in repr(ev))
            self.assertTrue(": no" in repr(ev))
            self.assertTrue("3" in repr(ev))

            # Test to ensure a callback extracted from the event
            # is kept alive and usable when the event is destroyed.
            ev = t_event(
                ex=x + v,
                callback=local_cb(),
                direction=event_direction.negative,
                fp_type=fp_t,
            )
            out_cb = ev.callback
            del ev
            gc.collect()
            out_cb(1, 2, 3)
            out_cb(1, 2, 3)
            out_cb(1, 2, 3)
            self.assertEqual(out_cb.n, 3)

        # Unsupported fp_type.
        with self.assertRaises(TypeError) as cm:
            nt_event(x + v, lambda _: _, fp_type=str)
        self.assertTrue(
            'The floating-point type "{}" is not recognized/supported'.format(str)
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            t_event(x + v, fp_type=list)
        self.assertTrue(
            'The floating-point type "{}" is not recognized/supported'.format(list)
            in str(cm.exception)
        )

        # Batch events.
        ev = nt_event_batch(x + v, lambda _: _)
        self.assertTrue(" non-terminal" in repr(ev))
        self.assertTrue("(x + v)" in repr(ev))
        self.assertTrue("event_direction::any" in repr(ev))
        self.assertEqual(ev.expression, x + v)
        self.assertEqual(ev.direction, event_direction.any)
        self.assertFalse(ev.callback is None)

        ev = nt_event_batch(ex=x + v, callback=lambda _: _, fp_type=float)
        self.assertTrue(" non-terminal" in repr(ev))
        self.assertTrue("(x + v)" in repr(ev))
        self.assertTrue("event_direction::any" in repr(ev))
        self.assertEqual(ev.expression, x + v)
        self.assertEqual(ev.direction, event_direction.any)
        self.assertFalse(ev.callback is None)

        ev = nt_event_batch(
            ex=x + v, callback=lambda _: _, direction=event_direction.positive
        )
        self.assertTrue(" non-terminal" in repr(ev))
        self.assertTrue("(x + v)" in repr(ev))
        self.assertTrue("event_direction::positive" in repr(ev))
        self.assertEqual(ev.expression, x + v)
        self.assertEqual(ev.direction, event_direction.positive)
        self.assertFalse(ev.callback is None)

        ev = nt_event_batch(
            ex=x + v, callback=lambda _: _, direction=event_direction.negative
        )
        self.assertTrue(" non-terminal" in repr(ev))
        self.assertTrue("(x + v)" in repr(ev))
        self.assertTrue("event_direction::negative" in repr(ev))
        self.assertEqual(ev.expression, x + v)
        self.assertEqual(ev.direction, event_direction.negative)
        self.assertFalse(ev.callback is None)

        class local_cb:
            def __init__(self):
                self.n = 0

            def __call__(self, ta, t, d_sgn):
                self.n = self.n + 1

        lcb = local_cb()
        ev = nt_event_batch(ex=x + v, callback=lcb, direction=event_direction.negative)
        self.assertEqual(ev.callback.n, 0)
        cb = ev.callback
        cb(1, 2, 3)
        cb(1, 2, 3)
        cb(1, 2, 3)
        self.assertEqual(ev.callback.n, 3)
        ev.callback.n = 0
        self.assertEqual(ev.callback.n, 0)
        self.assertEqual(id(lcb), id(ev.callback))

        with self.assertRaises(ValueError) as cm:
            nt_event_batch(
                ex=x + v, callback=lambda _: _, direction=event_direction(10)
            )
        self.assertTrue(
            "Invalid value selected for the direction of a non-terminal event"
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            nt_event_batch(ex=x + v, callback=3)
        self.assertTrue(
            "An object of type '{}' cannot be used as an event callback because it is not callable".format(
                str(type(3))
            )
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            nt_event_batch(ex=x + v, callback=None)
        self.assertTrue(
            "An object of type '{}' cannot be used as an event callback because it is not callable".format(
                str(type(None))
            )
            in str(cm.exception)
        )

        ev = nt_event_batch(
            ex=x + v, callback=lambda _: _, direction=event_direction.negative
        )
        ev = pickle.loads(pickle.dumps(ev))
        self.assertTrue(" non-terminal" in repr(ev))
        self.assertTrue("(x + v)" in repr(ev))
        self.assertTrue("event_direction::negative" in repr(ev))

        # Test dynamic attributes.
        ev.foo = "hello world"
        ev = pickle.loads(pickle.dumps(ev))
        self.assertEqual(ev.foo, "hello world")

        # Test copy semantics.
        class foo:
            pass

        ev.bar = foo()

        self.assertEqual(id(ev.bar), id(copy(ev).bar))
        self.assertNotEqual(id(ev.bar), id(deepcopy(ev).bar))

        # Test to ensure a callback extracted from the event
        # is kept alive and usable when the event is destroyed.
        ev = nt_event_batch(
            ex=x + v, callback=local_cb(), direction=event_direction.negative
        )
        out_cb = ev.callback
        del ev
        gc.collect()
        out_cb(1, 2, 3)
        out_cb(1, 2, 3)
        out_cb(1, 2, 3)
        self.assertEqual(out_cb.n, 3)

        # Terminal event.
        fp_t = float
        ev = t_event_batch(x + v)

        self.assertTrue(" terminal" in repr(ev))
        self.assertTrue("(x + v)" in repr(ev))
        self.assertTrue("event_direction::any" in repr(ev))
        self.assertTrue(": no" in repr(ev))
        self.assertTrue("auto" in repr(ev))
        self.assertEqual(ev.expression, x + v)
        self.assertEqual(ev.direction, event_direction.any)
        self.assertEqual(ev.cooldown, fp_t(-1))
        self.assertTrue(ev.callback is None)

        ev = t_event_batch(x + v, direction=event_direction.negative, cooldown=fp_t(3))

        self.assertTrue(" terminal" in repr(ev))
        self.assertTrue("(x + v)" in repr(ev))
        self.assertTrue("event_direction::negative" in repr(ev))
        self.assertTrue(": no" in repr(ev))
        self.assertTrue("3" in repr(ev))
        self.assertEqual(ev.expression, x + v)
        self.assertEqual(ev.direction, event_direction.negative)
        self.assertEqual(ev.cooldown, fp_t(3))
        self.assertTrue(ev.callback is None)

        ev = t_event_batch(
            x + v,
            direction=event_direction.positive,
            cooldown=fp_t(3),
            callback=lambda _: _,
        )

        self.assertTrue(" terminal" in repr(ev))
        self.assertTrue("(x + v)" in repr(ev))
        self.assertTrue("event_direction::positive" in repr(ev))
        self.assertTrue(": yes" in repr(ev))
        self.assertTrue("3" in repr(ev))
        self.assertEqual(ev.expression, x + v)
        self.assertEqual(ev.direction, event_direction.positive)
        self.assertEqual(ev.cooldown, fp_t(3))
        self.assertFalse(ev.callback is None)

        class local_cb:
            def __init__(self):
                self.n = 0

            def __call__(self, ta, mr, d_sgn):
                self.n = self.n + 1

        lcb = local_cb()
        ev = t_event_batch(
            x + v, direction=event_direction.positive, cooldown=fp_t(3), callback=lcb
        )

        self.assertTrue(" terminal" in repr(ev))
        self.assertTrue("(x + v)" in repr(ev))
        self.assertTrue("event_direction::positive" in repr(ev))
        self.assertTrue(": yes" in repr(ev))
        self.assertTrue("3" in repr(ev))
        self.assertEqual(ev.expression, x + v)
        self.assertEqual(ev.direction, event_direction.positive)
        self.assertEqual(ev.cooldown, fp_t(3))
        self.assertFalse(ev.callback is None)
        self.assertEqual(ev.callback.n, 0)
        cb = ev.callback
        cb(1, 2, 3)
        cb(1, 2, 3)
        cb(1, 2, 3)
        self.assertEqual(ev.callback.n, 3)
        ev.callback.n = 0
        self.assertEqual(ev.callback.n, 0)
        self.assertEqual(id(lcb), id(ev.callback))

        ev = t_event_batch(
            x + v, direction=event_direction.positive, cooldown=fp_t(3), callback=None
        )
        self.assertTrue(ev.callback is None)

        with self.assertRaises(ValueError) as cm:
            t_event_batch(
                x + v,
                direction=event_direction(45),
                cooldown=fp_t(3),
                callback=lambda _: _,
            )
        self.assertTrue(
            "Invalid value selected for the direction of a terminal event"
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            t_event_batch(x + v, callback=3)
        self.assertTrue(
            "An object of type '{}' cannot be used as an event callback because it is not callable".format(
                str(type(3))
            )
            in str(cm.exception)
        )

        ev = t_event_batch(
            x + v,
            direction=event_direction.positive,
            cooldown=fp_t(3),
            callback=lambda _: _,
        )

        ev = pickle.loads(pickle.dumps(ev))
        self.assertTrue(" terminal" in repr(ev))
        self.assertTrue("(x + v)" in repr(ev))
        self.assertTrue("event_direction::positive" in repr(ev))
        self.assertTrue(": yes" in repr(ev))
        self.assertTrue("3" in repr(ev))

        # Test dynamic attributes.
        ev.foo = "hello world"
        ev = pickle.loads(pickle.dumps(ev))
        self.assertEqual(ev.foo, "hello world")

        # Test copy semantics.
        class foo:
            pass

        ev.bar = foo()

        self.assertEqual(id(ev.bar), id(copy(ev).bar))
        self.assertNotEqual(id(ev.bar), id(deepcopy(ev).bar))

        # Test also with empty callback.
        ev = t_event_batch(x + v, direction=event_direction.positive, cooldown=fp_t(3))

        ev = pickle.loads(pickle.dumps(ev))
        self.assertTrue(" terminal" in repr(ev))
        self.assertTrue("(x + v)" in repr(ev))
        self.assertTrue("event_direction::positive" in repr(ev))
        self.assertTrue(": no" in repr(ev))
        self.assertTrue("3" in repr(ev))

        # Test to ensure a callback extracted from the event
        # is kept alive and usable when the event is destroyed.
        ev = t_event_batch(
            ex=x + v, callback=local_cb(), direction=event_direction.negative
        )
        out_cb = ev.callback
        del ev
        gc.collect()
        out_cb(1, 2, 3)
        out_cb(1, 2, 3)
        out_cb(1, 2, 3)
        self.assertEqual(out_cb.n, 3)

        with self.assertRaises(TypeError) as cm:
            nt_event_batch(x + v, lambda _: _, fp_type=str)
        self.assertTrue(
            'The floating-point type "{}" is not recognized/supported'.format(str)
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            t_event_batch(x + v, fp_type=list)
        self.assertTrue(
            'The floating-point type "{}" is not recognized/supported'.format(list)
            in str(cm.exception)
        )


class event_detection_test_case(_ut.TestCase):
    def test_batch(self):
        from . import (
            t_event_batch,
            nt_event_batch,
            make_vars,
            taylor_adaptive_batch,
            sin,
            taylor_outcome,
        )
        from sys import getrefcount
        from copy import deepcopy

        x, v = make_vars("x", "v")

        # Use a pendulum for testing purposes.
        sys = [(x, v), (v, -9.8 * sin(x))]

        # Non-terminal events.
        counter = [0] * 2
        cur_time = [0.0] * 2

        # Track the memory address of the integrator object
        # in order to make sure that it is passed correctly
        # into the callback.
        ta_id = None

        def cb0(ta, t, d_sgn, bidx):
            nonlocal counter
            nonlocal cur_time
            nonlocal ta_id

            self.assertTrue(t > cur_time[bidx])
            self.assertTrue(counter[bidx] % 3 == 0 or counter[bidx] % 3 == 2)
            self.assertEqual(ta_id, id(ta))

            counter[bidx] = counter[bidx] + 1
            cur_time[bidx] = t

        def cb1(ta, t, d_sgn, bidx):
            nonlocal counter
            nonlocal cur_time
            nonlocal ta_id

            self.assertTrue(t > cur_time[bidx])
            self.assertTrue(counter[bidx] % 3 == 1)
            self.assertEqual(ta_id, id(ta))

            counter[bidx] = counter[bidx] + 1
            cur_time[bidx] = t

        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0.0, 0.001], [0.25, 0.2501]],
            nt_events=[nt_event_batch(v * v - 1e-10, cb0), nt_event_batch(v, cb1)],
        )

        ta_id = id(ta)

        ta.propagate_until([4.0, 4.0])
        self.assertTrue(
            all(_[0] == taylor_outcome.time_limit for _ in ta.propagate_res)
        )

        self.assertEqual(counter[0], 12)
        self.assertEqual(counter[1], 12)

        # Make sure that when accessing events
        # from the integrator property we always
        # get the same object.
        class cb0:
            def __init__(self):
                self.lst = []

            def __call__(self, ta, t, d_sgn, bidx):
                pass

        class cb1:
            def __init__(self):
                self.lst = []

            def __call__(self, ta, t, d_sgn, bidx):
                pass

        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0.0, 0.001], [0.25, 0.2501]],
            nt_events=[
                nt_event_batch(v * v - 1e-10, cb0()),
                nt_event_batch(v, cb1()),
                nt_event_batch(v, cb1()),
            ],
        )

        # Check that the refcount increases by 3
        # (the number of events).
        rc = getrefcount(ta)
        nt_list = ta.nt_events
        new_rc = getrefcount(ta)
        self.assertEqual(new_rc, rc + 3)

        self.assertEqual(id(ta.nt_events[0].callback), id(ta.nt_events[0].callback))
        self.assertEqual(
            id(ta.nt_events[0].callback.lst), id(ta.nt_events[0].callback.lst)
        )
        self.assertEqual(id(ta.nt_events[1].callback), id(ta.nt_events[1].callback))
        self.assertEqual(
            id(ta.nt_events[1].callback.lst), id(ta.nt_events[1].callback.lst)
        )
        self.assertEqual(id(ta.nt_events[2].callback), id(ta.nt_events[2].callback))
        self.assertEqual(
            id(ta.nt_events[2].callback.lst), id(ta.nt_events[2].callback.lst)
        )

        # Ensure a deep copy of the integrator performs
        # a deep copy of the events.
        ta_copy = deepcopy(ta)

        self.assertNotEqual(
            id(ta_copy.nt_events[0].callback), id(ta.nt_events[0].callback)
        )
        self.assertNotEqual(
            id(ta_copy.nt_events[0].callback.lst), id(ta.nt_events[0].callback.lst)
        )
        self.assertNotEqual(
            id(ta_copy.nt_events[1].callback), id(ta.nt_events[1].callback)
        )
        self.assertNotEqual(
            id(ta_copy.nt_events[1].callback.lst), id(ta.nt_events[1].callback.lst)
        )
        self.assertNotEqual(
            id(ta_copy.nt_events[2].callback), id(ta.nt_events[2].callback)
        )
        self.assertNotEqual(
            id(ta_copy.nt_events[2].callback.lst), id(ta.nt_events[2].callback.lst)
        )

        # Callback with wrong signature.
        def cb2(ta, t):
            pass

        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0.0, 0.001], [0.25, 0.2501]],
            nt_events=[nt_event_batch(v * v - 1e-10, cb2)],
        )

        with self.assertRaises(RuntimeError):
            ta.propagate_until([4.0, 4.0])

        # Terminal events.
        counter_t = [0] * 2
        counter_nt = [0] * 2
        cur_time = [0.0] * 2

        def cb0(ta, t, d_sgn, bidx):
            nonlocal counter_nt
            nonlocal cur_time
            nonlocal ta_id

            self.assertTrue(t > cur_time[bidx])
            self.assertEqual(ta_id, id(ta))

            counter_nt[bidx] = counter_nt[bidx] + 1
            cur_time[bidx] = t

        def cb1(ta, mr, d_sgn, bidx):
            nonlocal cur_time
            nonlocal counter_t
            nonlocal ta_id

            self.assertFalse(mr)
            self.assertTrue(ta.time[bidx] > cur_time[bidx])
            self.assertEqual(ta_id, id(ta))

            counter_t[bidx] = counter_t[bidx] + 1
            cur_time[bidx] = ta.time[bidx]

            return True

        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0.0, 0.001], [0.25, 0.2501]],
            nt_events=[nt_event_batch(v * v - 1e-10, cb0)],
            t_events=[t_event_batch(v, callback=cb1)],
        )

        ta_id = id(ta)

        while True:
            ta.step()
            if all(_[0] > taylor_outcome.success for _ in ta.step_res):
                break

        self.assertTrue(all(int(_[0]) == 0 for _ in ta.step_res))
        self.assertTrue(all(_ < 1 for _ in ta.time))
        self.assertTrue(all(_ == 1 for _ in counter_nt))
        self.assertTrue(all(_ == 1 for _ in counter_t))

        while True:
            ta.step()
            if all(_[0] > taylor_outcome.success for _ in ta.step_res):
                break

        self.assertTrue(all(int(_[0]) == 0 for _ in ta.step_res))
        self.assertTrue(all(_ > 1 for _ in ta.time))
        self.assertTrue(all(_ == 3 for _ in counter_nt))
        self.assertTrue(all(_ == 2 for _ in counter_t))

        # Make sure that when accessing events
        # from the integrator property we always
        # get the same object.
        class cb0:
            def __init__(self):
                self.lst = []

            def __call__(self, ta, mr, d_sgn, bidx):
                pass

        class cb1:
            def __init__(self):
                self.lst = []

            def __call__(self, ta, mr, d_sgn, bidx):
                pass

        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0.0, 0.001], [0.25, 0.2501]],
            t_events=[
                t_event_batch(v * v - 1e-10, callback=cb0()),
                t_event_batch(v, callback=cb1()),
                t_event_batch(v, callback=cb1()),
            ],
        )

        # Check that the refcount increases by 3
        # (the number of events).
        rc = getrefcount(ta)
        t_list = ta.t_events
        new_rc = getrefcount(ta)
        self.assertEqual(new_rc, rc + 3)

        self.assertEqual(id(ta.t_events[0].callback), id(ta.t_events[0].callback))
        self.assertEqual(
            id(ta.t_events[0].callback.lst), id(ta.t_events[0].callback.lst)
        )
        self.assertEqual(id(ta.t_events[1].callback), id(ta.t_events[1].callback))
        self.assertEqual(
            id(ta.t_events[1].callback.lst), id(ta.t_events[1].callback.lst)
        )
        self.assertEqual(id(ta.t_events[2].callback), id(ta.t_events[2].callback))
        self.assertEqual(
            id(ta.t_events[2].callback.lst), id(ta.t_events[2].callback.lst)
        )

        # Ensure a deep copy of the integrator performs
        # a deep copy of the events.
        ta_copy = deepcopy(ta)

        self.assertNotEqual(
            id(ta_copy.t_events[0].callback), id(ta.t_events[0].callback)
        )
        self.assertNotEqual(
            id(ta_copy.t_events[0].callback.lst), id(ta.t_events[0].callback.lst)
        )
        self.assertNotEqual(
            id(ta_copy.t_events[1].callback), id(ta.t_events[1].callback)
        )
        self.assertNotEqual(
            id(ta_copy.t_events[1].callback.lst), id(ta.t_events[1].callback.lst)
        )
        self.assertNotEqual(
            id(ta_copy.t_events[2].callback), id(ta.t_events[2].callback)
        )
        self.assertNotEqual(
            id(ta_copy.t_events[2].callback.lst), id(ta.t_events[2].callback.lst)
        )

        # Callback with wrong signature.
        def cb2(ta, t):
            pass

        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0.0, 0.001], [0.25, 0.2501]],
            t_events=[t_event_batch(v * v - 1e-10, callback=cb2)],
        )

        with self.assertRaises(RuntimeError):
            ta.propagate_until([4.0, 4.0])

        # Callback with wrong retval.
        def cb3(ta, mr, d_sgn, bidx):
            return "hello"

        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0.0, 0.001], [0.25, 0.2501]],
            t_events=[t_event_batch(v * v - 1e-10, callback=cb3)],
        )

        with self.assertRaises(RuntimeError) as cm:
            ta.propagate_until([4.0, 4.0])
        self.assertTrue(
            "in the construction of the return value of an event callback"
            in str(cm.exception)
        )

    def test_scalar(self):
        from . import (
            t_event,
            nt_event,
            make_vars,
            sin,
            taylor_adaptive,
            taylor_outcome,
            core,
        )
        from .core import _ppc_arch
        from sys import getrefcount
        import numpy as np
        from copy import deepcopy

        x, v = make_vars("x", "v")

        if _ppc_arch:
            fp_types = [float]
        else:
            fp_types = [float, np.longdouble]

        if hasattr(core, "real128"):
            fp_types.append(core.real128)

        # Use a pendulum for testing purposes.
        sys = [(x, v), (v, -9.8 * sin(x))]

        for fp_t in fp_types:
            # Non-terminal events.
            counter = 0
            cur_time = fp_t(0)

            # Track the memory address of the integrator object
            # in order to make sure that it is passed correctly
            # into the callback.
            ta_id = None

            def cb0(ta, t, d_sgn):
                nonlocal counter
                nonlocal cur_time
                nonlocal ta_id

                self.assertTrue(t > cur_time)
                self.assertTrue(counter % 3 == 0 or counter % 3 == 2)
                self.assertEqual(ta_id, id(ta))

                counter = counter + 1
                cur_time = t

            def cb1(ta, t, d_sgn):
                nonlocal counter
                nonlocal cur_time
                nonlocal ta_id

                self.assertTrue(t > cur_time)
                self.assertTrue(counter % 3 == 1)
                self.assertEqual(ta_id, id(ta))

                counter = counter + 1
                cur_time = t

            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0), fp_t(0.25)],
                fp_type=fp_t,
                nt_events=[
                    nt_event(v * v - 1e-10, cb0, fp_type=fp_t),
                    nt_event(v, cb1, fp_type=fp_t),
                ],
            )

            ta_id = id(ta)

            self.assertEqual(ta.propagate_until(fp_t(4))[0], taylor_outcome.time_limit)

            self.assertEqual(counter, 12)

            # Make sure that when accessing events
            # from the integrator property we always
            # get the same object.
            class cb0:
                def __init__(self):
                    self.lst = []

                def __call__(self, ta, t, d_sgn):
                    pass

            class cb1:
                def __init__(self):
                    self.lst = []

                def __call__(self, ta, t, d_sgn):
                    pass

            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0), fp_t(0.25)],
                fp_type=fp_t,
                nt_events=[
                    nt_event(v * v - 1e-10, cb0(), fp_type=fp_t),
                    nt_event(v, cb1(), fp_type=fp_t),
                    nt_event(v, cb1(), fp_type=fp_t),
                ],
            )

            # Check that the refcount increases by 3
            # (the number of events).
            rc = getrefcount(ta)
            nt_list = ta.nt_events
            new_rc = getrefcount(ta)
            self.assertEqual(new_rc, rc + 3)

            self.assertEqual(id(ta.nt_events[0].callback), id(ta.nt_events[0].callback))
            self.assertEqual(
                id(ta.nt_events[0].callback.lst), id(ta.nt_events[0].callback.lst)
            )
            self.assertEqual(id(ta.nt_events[1].callback), id(ta.nt_events[1].callback))
            self.assertEqual(
                id(ta.nt_events[1].callback.lst), id(ta.nt_events[1].callback.lst)
            )
            self.assertEqual(id(ta.nt_events[2].callback), id(ta.nt_events[2].callback))
            self.assertEqual(
                id(ta.nt_events[2].callback.lst), id(ta.nt_events[2].callback.lst)
            )

            # Ensure a deep copy of the integrator performs
            # a deep copy of the events.
            ta_copy = deepcopy(ta)

            self.assertNotEqual(
                id(ta_copy.nt_events[0].callback), id(ta.nt_events[0].callback)
            )
            self.assertNotEqual(
                id(ta_copy.nt_events[0].callback.lst), id(ta.nt_events[0].callback.lst)
            )
            self.assertNotEqual(
                id(ta_copy.nt_events[1].callback), id(ta.nt_events[1].callback)
            )
            self.assertNotEqual(
                id(ta_copy.nt_events[1].callback.lst), id(ta.nt_events[1].callback.lst)
            )
            self.assertNotEqual(
                id(ta_copy.nt_events[2].callback), id(ta.nt_events[2].callback)
            )
            self.assertNotEqual(
                id(ta_copy.nt_events[2].callback.lst), id(ta.nt_events[2].callback.lst)
            )

            # Callback with wrong signature.
            def cb2(ta, t):
                pass

            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0), fp_t(0.25)],
                fp_type=fp_t,
                nt_events=[nt_event(v * v - 1e-10, cb2, fp_type=fp_t)],
            )

            with self.assertRaises(TypeError):
                ta.propagate_until(fp_t(4))

            # Terminal events.
            counter_t = 0
            counter_nt = 0
            cur_time = fp_t(0)

            def cb0(ta, t, d_sgn):
                nonlocal counter_nt
                nonlocal cur_time
                nonlocal ta_id

                self.assertTrue(t > cur_time)
                self.assertEqual(ta_id, id(ta))

                counter_nt = counter_nt + 1
                cur_time = t

            def cb1(ta, mr, d_sgn):
                nonlocal cur_time
                nonlocal counter_t
                nonlocal ta_id

                self.assertFalse(mr)
                self.assertTrue(ta.time > cur_time)
                self.assertEqual(ta_id, id(ta))

                counter_t = counter_t + 1
                cur_time = ta.time

                return True

            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0), fp_t(0.25)],
                fp_type=fp_t,
                nt_events=[nt_event(v * v - 1e-10, cb0, fp_type=fp_t)],
                t_events=[t_event(v, callback=cb1, fp_type=fp_t)],
            )

            ta_id = id(ta)

            while True:
                oc, _ = ta.step()
                if oc > taylor_outcome.success:
                    break
                self.assertEqual(oc, taylor_outcome.success)

            self.assertEqual(int(oc), 0)
            self.assertTrue(ta.time < 1)
            self.assertEqual(counter_nt, 1)
            self.assertEqual(counter_t, 1)

            while True:
                oc, _ = ta.step()
                if oc > taylor_outcome.success:
                    break
                self.assertEqual(oc, taylor_outcome.success)

            self.assertEqual(int(oc), 0)
            self.assertTrue(ta.time > 1)
            self.assertEqual(counter_nt, 3)
            self.assertEqual(counter_t, 2)

            # Make sure that when accessing events
            # from the integrator property we always
            # get the same object.
            class cb0:
                def __init__(self):
                    self.lst = []

                def __call__(self, ta, mr, d_sgn):
                    pass

            class cb1:
                def __init__(self):
                    self.lst = []

                def __call__(self, ta, mr, d_sgn):
                    pass

            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0), fp_t(0.25)],
                fp_type=fp_t,
                t_events=[
                    t_event(v * v - 1e-10, callback=cb0(), fp_type=fp_t),
                    t_event(v, callback=cb1(), fp_type=fp_t),
                    t_event(v, callback=cb1(), fp_type=fp_t),
                ],
            )

            # Check that the refcount increases by 3
            # (the number of events).
            rc = getrefcount(ta)
            t_list = ta.t_events
            new_rc = getrefcount(ta)
            self.assertEqual(new_rc, rc + 3)

            self.assertEqual(id(ta.t_events[0].callback), id(ta.t_events[0].callback))
            self.assertEqual(
                id(ta.t_events[0].callback.lst), id(ta.t_events[0].callback.lst)
            )
            self.assertEqual(id(ta.t_events[1].callback), id(ta.t_events[1].callback))
            self.assertEqual(
                id(ta.t_events[1].callback.lst), id(ta.t_events[1].callback.lst)
            )
            self.assertEqual(id(ta.t_events[2].callback), id(ta.t_events[2].callback))
            self.assertEqual(
                id(ta.t_events[2].callback.lst), id(ta.t_events[2].callback.lst)
            )

            # Ensure a deep copy of the integrator performs
            # a deep copy of the events.
            ta_copy = deepcopy(ta)

            self.assertNotEqual(
                id(ta_copy.t_events[0].callback), id(ta.t_events[0].callback)
            )
            self.assertNotEqual(
                id(ta_copy.t_events[0].callback.lst), id(ta.t_events[0].callback.lst)
            )
            self.assertNotEqual(
                id(ta_copy.t_events[1].callback), id(ta.t_events[1].callback)
            )
            self.assertNotEqual(
                id(ta_copy.t_events[1].callback.lst), id(ta.t_events[1].callback.lst)
            )
            self.assertNotEqual(
                id(ta_copy.t_events[2].callback), id(ta.t_events[2].callback)
            )
            self.assertNotEqual(
                id(ta_copy.t_events[2].callback.lst), id(ta.t_events[2].callback.lst)
            )

            # Callback with wrong signature.
            def cb2(ta, t):
                pass

            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0), fp_t(0.25)],
                fp_type=fp_t,
                t_events=[t_event(v * v - 1e-10, callback=cb2, fp_type=fp_t)],
            )

            with self.assertRaises(TypeError):
                ta.propagate_until(fp_t(4))

            # Callback with wrong retval.
            def cb3(ta, mr, d_sgn):
                return "hello"

            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0), fp_t(0.25)],
                fp_type=fp_t,
                t_events=[t_event(v * v - 1e-10, callback=cb3, fp_type=fp_t)],
            )

            with self.assertRaises(TypeError) as cm:
                ta.propagate_until(fp_t(4))
            self.assertTrue(
                "in the construction of the return value of an event callback"
                in str(cm.exception)
            )


class expression_eval_test_case(_ut.TestCase):
    def test_basic(self):
        from . import make_vars, eval, core
        from .core import _ppc_arch
        import numpy as np
        from math import log10

        (x,) = make_vars("x")

        if _ppc_arch:
            fp_types = [(float, int(-log10(np.finfo(float).eps)) - 1)]
        else:
            fp_types = [
                (float, int(-log10(np.finfo(float).eps)) - 1),
                (np.longdouble, int(-log10(np.finfo(np.longdouble).eps)) - 1),
            ]

        if hasattr(core, "real128"):
            fp_types.append((core.real128, 32))

        for fp_t, places in fp_types:
            target = fp_t("0.123456789012345678901234567890")
            a = eval(x, {"x": target}, fp_type=fp_t)
            self.assertEqual(a, target)
            a = eval(x**3.1, {"x": target}, fp_type=fp_t)
            self.assertAlmostEqual(a, target**3.1, places=places)


class scalar_integrator_test_case(_ut.TestCase):
    def test_type_conversions(self):
        # Test to check automatic conversions of std::vector<T>
        # in the integrator's constructor.

        from . import taylor_adaptive, make_vars, sin
        import numpy as np

        d_digs = np.finfo(np.double).nmant
        ld_digs = np.finfo(np.longdouble).nmant

        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]

        ta = taylor_adaptive(sys=sys, state=(0.0, 0.25), tol=1e-4)
        self.assertTrue(np.all(ta.state == [0.0, 0.25]))
        ta = taylor_adaptive(sys=sys, state=np.array([0.0, 0.25]), tol=1e-4)
        self.assertTrue(np.all(ta.state == [0.0, 0.25]))

        if d_digs == ld_digs:
            return

        # Check that conversion from other fp types is forbidden.
        with self.assertRaises(TypeError) as cm:
            ta = taylor_adaptive(
                sys=sys, state=(np.longdouble(0.0), np.longdouble(0.25)), tol=1e-4
            )

        with self.assertRaises(TypeError) as cm:
            ta = taylor_adaptive(
                sys=sys, state=np.array([0.0, 0.25], dtype=np.longdouble), tol=1e-4
            )

    def test_dtime(self):
        from . import taylor_adaptive, make_vars, sin

        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]

        ta = taylor_adaptive(sys=sys, state=[0.0, 0.25])

        self.assertEqual(ta.dtime, (0.0, 0.0))

        ta.step()
        ta.propagate_for(1001.1)

        self.assertTrue(ta.dtime[1] != 0)

        ta.dtime = (1, 0.5)

        self.assertEqual(ta.dtime, (1.5, 0.0))

    def test_copy(self):
        from . import taylor_adaptive, make_vars, t_event, sin
        import numpy as np
        from copy import copy, deepcopy

        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]

        ta = taylor_adaptive(sys=sys, state=[0.0, 0.25], t_events=[t_event(v)])

        ta.step()

        class foo:
            pass

        ta.bar = foo()

        self.assertEqual(id(ta.bar), id(copy(ta).bar))
        self.assertNotEqual(id(ta.bar), id(deepcopy(ta).bar))
        self.assertTrue(np.all(ta.state == copy(ta).state))
        self.assertTrue(np.all(ta.state == deepcopy(ta).state))

        ta_dc = deepcopy(ta)
        self.assertEqual(ta_dc.state[0], ta.state[0])
        ta.state[0] += 1
        self.assertNotEqual(ta_dc.state[0], ta.state[0])

    def test_basic(self):
        from . import taylor_adaptive, make_vars, t_event, sin

        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]

        ta = taylor_adaptive(sys=sys, state=[0.0, 0.25], t_events=[t_event(v)])

        self.assertTrue(ta.with_events)
        self.assertFalse(ta.compact_mode)
        self.assertFalse(ta.high_accuracy)

        ta = taylor_adaptive(
            sys=sys, state=[0.0, 0.25], compact_mode=True, high_accuracy=True
        )

        self.assertFalse(ta.with_events)
        self.assertTrue(ta.compact_mode)
        self.assertTrue(ta.high_accuracy)
        self.assertFalse(ta.llvm_state.fast_math)
        self.assertFalse(ta.llvm_state.force_avx512)
        self.assertEqual(ta.llvm_state.opt_level, 3)

        # Check that certain properties are read-only
        # arrays and the writeability cannot be changed.
        self.assertFalse(ta.tc.flags.writeable)
        with self.assertRaises(ValueError):
            ta.tc.flags.writeable = True
        self.assertFalse(ta.d_output.flags.writeable)
        with self.assertRaises(ValueError):
            ta.d_output.flags.writeable = True

        # Test the custom llvm_state flags.
        ta = taylor_adaptive(
            sys=sys,
            state=[0.0, 0.25],
            compact_mode=True,
            high_accuracy=True,
            force_avx512=True,
            fast_math=True,
            opt_level=0,
        )

        self.assertTrue(ta.llvm_state.fast_math)
        self.assertTrue(ta.llvm_state.force_avx512)
        self.assertEqual(ta.llvm_state.opt_level, 0)

        # Test that adding dynattrs to the integrator
        # object via the propagate callback works.
        def cb(ta):
            if hasattr(ta, "counter"):
                ta.counter += 1
            else:
                ta.counter = 0

            return True

        ta.propagate_until(10.0, callback=cb)

        self.assertTrue(ta.counter > 0)
        orig_ct = ta.counter

        ta.propagate_for(10.0, callback=cb)

        self.assertTrue(ta.counter > orig_ct)
        orig_ct = ta.counter

        ta.time = 0.0
        ta.propagate_grid([0.0, 1.0, 2.0], callback=cb)

        self.assertTrue(ta.counter > orig_ct)

        # Test that no copies of the callback are performed.
        class cb:
            def __call__(_, ta):
                self.assertEqual(id(_), _.orig_id)

                return True

        cb_inst = cb()
        cb_inst.orig_id = id(cb_inst)

        ta.time = 0.0
        ta.propagate_until(10.0, callback=cb_inst)
        ta.propagate_for(10.0, callback=cb_inst)
        ta.time = 0.0
        ta.propagate_grid([0.0, 1.0, 2.0], callback=cb_inst)

    def test_events(self):
        from . import nt_event, t_event, make_vars, sin, taylor_adaptive

        x, v = make_vars("x", "v")

        # Use a pendulum for testing purposes.
        sys = [(x, v), (v, -9.8 * sin(x))]

        def cb0(ta, t, d_sgn):
            pass

        ta = taylor_adaptive(
            sys=sys,
            state=[0.0, 0.25],
            nt_events=[nt_event(v * v - 1e-10, cb0)],
            t_events=[t_event(v)],
        )

        self.assertTrue(ta.with_events)
        self.assertEqual(len(ta.t_events), 1)
        self.assertEqual(len(ta.nt_events), 1)

        oc = ta.propagate_until(1e9)[0]
        self.assertEqual(int(oc), -1)
        self.assertFalse(ta.te_cooldowns[0] is None)

        ta.reset_cooldowns()
        self.assertTrue(ta.te_cooldowns[0] is None)

    def test_s11n(self):
        from . import nt_event, make_vars, sin, taylor_adaptive, core
        from .core import _ppc_arch
        import numpy as np
        import pickle

        x, v = make_vars("x", "v")

        if _ppc_arch:
            fp_types = [float]
        else:
            fp_types = [float, np.longdouble]

        if hasattr(core, "real128"):
            fp_types.append(core.real128)

        # Use a pendulum for testing purposes.
        sys = [(x, v), (v, -9.8 * sin(x))]

        def cb0(ta, t, d_sgn):
            pass

        for fp_t in fp_types:
            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0), fp_t(0.25)],
                fp_type=fp_t,
                nt_events=[nt_event(v * v - 1e-10, cb0, fp_type=fp_t)],
            )

            ta.step()
            ta.step()
            ta.step()
            ta.step()

            ta2 = pickle.loads(pickle.dumps(ta))

            self.assertEqual(len(ta.t_events), len(ta2.t_events))
            self.assertEqual(len(ta.nt_events), len(ta2.nt_events))

            # Test dynamic attributes.
            ta.foo = "hello world"
            ta = pickle.loads(pickle.dumps(ta))
            self.assertEqual(ta.foo, "hello world")

            self.assertTrue(np.all(ta.state == ta2.state))
            self.assertTrue(np.all(ta.time == ta2.time))

            ta.step()
            ta2.step()

            self.assertTrue(np.all(ta.state == ta2.state))
            self.assertTrue(np.all(ta.time == ta2.time))

            # Try also an integrator with stateful event callback.
            class cb1:
                def __init__(self):
                    self.n = 0

                def __call__(self, ta, t, d_sgn):
                    self.n = self.n + 1

            clb = cb1()
            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0), fp_t(0.25)],
                fp_type=fp_t,
                nt_events=[nt_event(v * v - 1e-10, clb, fp_type=fp_t)],
            )

            self.assertNotEqual(id(clb), id(ta.nt_events[0].callback))

            self.assertEqual(ta.nt_events[0].callback.n, 0)

            ta.propagate_until(fp_t(10))

            ta2 = pickle.loads(pickle.dumps(ta))

            self.assertEqual(ta.nt_events[0].callback.n, ta2.nt_events[0].callback.n)

            # Test dynamic attributes.
            ta.foo = "hello world"
            ta = pickle.loads(pickle.dumps(ta))
            self.assertEqual(ta.foo, "hello world")

            ta = taylor_adaptive(
                sys=sys, state=[fp_t(0), fp_t(0.25)], fp_type=fp_t, tol=fp_t(1e-6)
            )

            self.assertEqual(ta.tol, fp_t(1e-6))

        # Check throwing behaviour with long double on PPC.
        if _ppc_arch:
            fp_t = np.longdouble

            with self.assertRaises(NotImplementedError):
                taylor_adaptive(
                    sys=sys, state=[fp_t(0), fp_t(0.25)], fp_type=np.longdouble
                )


class batch_integrator_test_case(_ut.TestCase):
    def test_type_conversions(self):
        # Test to check automatic conversions of std::vector<T>
        # in the integrator's constructor.

        from . import taylor_adaptive_batch, make_vars, sin
        import numpy as np

        d_digs = np.finfo(np.double).nmant
        ld_digs = np.finfo(np.longdouble).nmant

        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]

        ta = taylor_adaptive_batch(sys=sys, state=((0.0, 0.1), (0.25, 0.26)), tol=1e-4)
        self.assertTrue(np.all(ta.state == ((0.0, 0.1), (0.25, 0.26))))

        ta = taylor_adaptive_batch(
            sys=sys, state=np.array([[0.0, 0.1], [0.25, 0.26]]), tol=1e-4
        )
        self.assertTrue(np.all(ta.state == ((0.0, 0.1), (0.25, 0.26))))

        if d_digs == ld_digs:
            return

        ld = np.longdouble

        # Check that conversion from other fp types is forbidden.
        with self.assertRaises(TypeError) as cm:
            ta = taylor_adaptive_batch(
                sys=sys, state=((ld(0.0), ld(0.1)), (ld(0.25), ld(0.26))), tol=1e-4
            )

        with self.assertRaises(TypeError) as cm:
            ta = taylor_adaptive_batch(
                sys=sys, state=np.array([[0.0, 0.1], [0.25, 0.26]], dtype=ld), tol=1e-4
            )

    def test_copy(self):
        from . import nt_event_batch, make_vars, sin, taylor_adaptive_batch
        from copy import copy, deepcopy
        import numpy as np

        x, v = make_vars("x", "v")

        # Use a pendulum for testing purposes.
        sys = [(x, v), (v, -9.8 * sin(x))]

        def cb0(ta, t, d_sgn, bidx):
            pass

        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0, 0.01], [0.25, 0.26]],
            nt_events=[nt_event_batch(v * v - 1e-10, cb0)],
        )

        ta.step()
        ta.step()
        ta.step()
        ta.step()

        class foo:
            pass

        ta.bar = foo()

        self.assertEqual(id(ta.bar), id(copy(ta).bar))
        self.assertNotEqual(id(ta.bar), id(deepcopy(ta).bar))
        self.assertTrue(np.all(ta.state == copy(ta).state))
        self.assertTrue(np.all(ta.state == deepcopy(ta).state))

        ta_dc = deepcopy(ta)
        self.assertEqual(ta_dc.state[0, 0], ta.state[0, 0])
        ta.state[0, 0] += 1
        self.assertNotEqual(ta_dc.state[0, 0], ta.state[0, 0])

    def test_propagate_for(self):
        from . import taylor_adaptive_batch, make_vars, sin
        from copy import deepcopy
        import numpy as np

        ic = [[0.0, 0.1, 0.2, 0.3], [0.25, 0.26, 0.27, 0.28]]

        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]

        ta = taylor_adaptive_batch(sys=sys, state=ic)

        # Compare vector/scalar delta_t and max_delta_t.
        ta.propagate_for([10.0] * 4)
        st = deepcopy(ta.state)
        res = deepcopy(ta.propagate_res)

        ta.set_time(0.0)
        ta.state[:] = ic

        ta.propagate_for(10.0)
        self.assertTrue(np.all(ta.state == st))
        self.assertEqual(res, ta.propagate_res)

        ta.set_time(0.0)
        ta.state[:] = ic

        ta.propagate_for([10.0] * 4, max_delta_t=[1e-4] * 4)
        st = deepcopy(ta.state)
        res = deepcopy(ta.propagate_res)

        ta.set_time(0.0)
        ta.state[:] = ic

        ta.propagate_for(10.0, max_delta_t=1e-4)
        self.assertTrue(np.all(ta.state == st))
        self.assertEqual(res, ta.propagate_res)

        # Test that adding dynattrs to the integrator
        # object via the propagate callback works.
        def cb(ta):
            if hasattr(ta, "counter"):
                ta.counter += 1
            else:
                ta.counter = 0

            return True

        ta.propagate_for(10.0, callback=cb)

        self.assertTrue(ta.counter > 0)

        # Test that no copies of the callback are performed.
        class cb:
            def __call__(_, ta):
                self.assertEqual(id(_), _.orig_id)

                return True

        cb_inst = cb()
        cb_inst.orig_id = id(cb_inst)

        ta.propagate_for(10.0, callback=cb_inst)

    def test_propagate_until(self):
        from . import taylor_adaptive_batch, make_vars, sin
        from copy import deepcopy
        import numpy as np

        ic = [[0.0, 0.1, 0.2, 0.3], [0.25, 0.26, 0.27, 0.28]]

        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]

        ta = taylor_adaptive_batch(sys=sys, state=ic)

        # Compare vector/scalar delta_t and max_delta_t.
        ta.propagate_until([10.0] * 4)
        st = deepcopy(ta.state)
        res = deepcopy(ta.propagate_res)

        ta.set_time(0.0)
        ta.state[:] = ic

        ta.propagate_until(10.0)
        self.assertTrue(np.all(ta.state == st))
        self.assertEqual(res, ta.propagate_res)

        ta.set_time(0.0)
        ta.state[:] = ic

        ta.propagate_until([10.0] * 4, max_delta_t=[1e-4] * 4)
        st = deepcopy(ta.state)
        res = deepcopy(ta.propagate_res)

        ta.set_time(0.0)
        ta.state[:] = ic

        ta.propagate_until(10.0, max_delta_t=1e-4)
        self.assertTrue(np.all(ta.state == st))
        self.assertEqual(res, ta.propagate_res)

        # Test that adding dynattrs to the integrator
        # object via the propagate callback works.
        def cb(ta):
            if hasattr(ta, "counter"):
                ta.counter += 1
            else:
                ta.counter = 0

            return True

        ta.propagate_until(20.0, callback=cb)

        self.assertTrue(ta.counter > 0)

        # Test that no copies of the callback are performed.
        class cb:
            def __call__(_, ta):
                self.assertEqual(id(_), _.orig_id)

                return True

        cb_inst = cb()
        cb_inst.orig_id = id(cb_inst)

        ta.propagate_until(30.0, callback=cb_inst)

    def test_update_d_output(self):
        from . import taylor_adaptive_batch, make_vars, sin
        from sys import getrefcount
        from copy import deepcopy
        import numpy as np

        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]

        ta = taylor_adaptive_batch(
            sys=sys, state=[[0.0, 0.1, 0.2, 0.3], [0.25, 0.26, 0.27, 0.28]]
        )

        ta.step(write_tc=True)

        # Scalar overload.
        with self.assertRaises(ValueError) as cm:
            ta.update_d_output(0.3)[0] = 0.5

        d_out = ta.update_d_output(0.3)
        self.assertEqual(d_out.shape, (2, 4))
        rc = getrefcount(ta)
        tmp_out = ta.update_d_output(0.2)
        new_rc = getrefcount(ta)
        self.assertEqual(new_rc, rc + 1)

        # Vector overload.
        with self.assertRaises(ValueError) as cm:
            ta.update_d_output([0.3, 0.4, 0.45, 0.46])[0] = 0.5

        d_out2 = ta.update_d_output([0.3, 0.4, 0.45, 0.46])
        self.assertEqual(d_out2.shape, (2, 4))
        rc = getrefcount(ta)
        tmp_out2 = ta.update_d_output([0.31, 0.41, 0.66, 0.67])
        new_rc = getrefcount(ta)
        self.assertEqual(new_rc, rc + 1)

        cp = deepcopy(ta.update_d_output(0.3))
        self.assertTrue(np.all(cp == ta.update_d_output([0.3] * 4)))

        # Functional testing.
        ta.set_time(0.0)
        ta.state[:] = [[0.0, 0.01, 0.02, 0.03], [0.205, 0.206, 0.207, 0.208]]
        ta.step(write_tc=True)
        ta.update_d_output(ta.time)
        self.assertTrue(
            np.allclose(
                ta.d_output,
                ta.state,
                rtol=np.finfo(float).eps * 10,
                atol=np.finfo(float).eps * 10,
            )
        )
        ta.update_d_output(0.0, rel_time=True)
        self.assertTrue(
            np.allclose(
                ta.d_output,
                ta.state,
                rtol=np.finfo(float).eps * 10,
                atol=np.finfo(float).eps * 10,
            )
        )

    def test_set_time(self):
        from . import taylor_adaptive_batch, make_vars, sin
        import numpy as np

        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]

        ta = taylor_adaptive_batch(sys=sys, state=[[0.0, 0.1], [0.25, 0.26]])

        self.assertTrue(np.all(ta.time == [0, 0]))

        ta.set_time([-1.0, 1.0])
        self.assertTrue(np.all(ta.time == [-1, 1]))

        ta.set_time(5.0)
        self.assertTrue(np.all(ta.time == [5, 5]))

    def test_dtime(self):
        from . import taylor_adaptive_batch, make_vars, sin
        import numpy as np

        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]

        ta = taylor_adaptive_batch(sys=sys, state=[[0.0, 0.1], [0.25, 0.26]])

        self.assertTrue(np.all(ta.dtime[0] == [0, 0]))
        self.assertTrue(np.all(ta.dtime[1] == [0, 0]))

        # Check not writeable,
        with self.assertRaises(ValueError) as cm:
            ta.dtime[0][0] = 0.5

        with self.assertRaises(ValueError) as cm:
            ta.dtime[1][0] = 0.5

        ta.step()
        ta.propagate_for(1000.1)

        self.assertFalse(np.all(ta.dtime[1] == [0, 0]))

        ta.set_dtime(1.0, 0.5)

        self.assertTrue(np.all(ta.dtime[0] == [1.5, 1.5]))
        self.assertTrue(np.all(ta.dtime[1] == [0, 0]))

        ta.set_dtime([1.0, 2.0], [0.5, 0.25])

        self.assertTrue(np.all(ta.dtime[0] == [1.5, 2.25]))
        self.assertTrue(np.all(ta.dtime[1] == [0, 0]))

        # Failure modes.
        with self.assertRaises(TypeError) as cm:
            ta.set_dtime([1.0, 2.0], 0.5)
        self.assertTrue(
            "The two arguments to the set_dtime() method must be of the same type"
            in str(cm.exception)
        )

    def test_basic(self):
        from . import taylor_adaptive_batch, make_vars, t_event_batch, sin

        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]

        ta = taylor_adaptive_batch(
            sys=sys, state=[[0.0, 0.1], [0.25, 0.26]], t_events=[t_event_batch(v)]
        )

        self.assertTrue(ta.with_events)
        self.assertFalse(ta.compact_mode)
        self.assertFalse(ta.high_accuracy)

        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0.0, 0.1], [0.25, 0.26]],
            compact_mode=True,
            high_accuracy=True,
        )

        self.assertFalse(ta.with_events)
        self.assertTrue(ta.compact_mode)
        self.assertTrue(ta.high_accuracy)
        self.assertFalse(ta.llvm_state.fast_math)
        self.assertFalse(ta.llvm_state.force_avx512)
        self.assertEqual(ta.llvm_state.opt_level, 3)

        # Test the custom llvm_state flags.
        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0.0, 0.1], [0.25, 0.26]],
            compact_mode=True,
            high_accuracy=True,
            force_avx512=True,
            fast_math=True,
            opt_level=0,
        )

        self.assertTrue(ta.llvm_state.fast_math)
        self.assertTrue(ta.llvm_state.force_avx512)
        self.assertEqual(ta.llvm_state.opt_level, 0)

    def test_events(self):
        from . import (
            nt_event_batch,
            t_event_batch,
            make_vars,
            sin,
            taylor_adaptive_batch,
        )

        x, v = make_vars("x", "v")

        # Use a pendulum for testing purposes.
        sys = [(x, v), (v, -9.8 * sin(x))]

        def cb0(ta, t, d_sgn, bidx):
            pass

        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0.0, 0.001], [0.25, 0.2501]],
            nt_events=[nt_event_batch(v * v - 1e-10, cb0)],
            t_events=[t_event_batch(v)],
        )

        self.assertTrue(ta.with_events)
        self.assertEqual(len(ta.t_events), 1)
        self.assertEqual(len(ta.nt_events), 1)

        ta.propagate_until([1e9, 1e9])
        self.assertTrue(all(int(_[0]) == -1 for _ in ta.propagate_res))

        self.assertFalse(ta.te_cooldowns[0][0] is None)
        self.assertFalse(ta.te_cooldowns[1][0] is None)

        ta.reset_cooldowns(0)
        self.assertTrue(ta.te_cooldowns[0][0] is None)
        self.assertFalse(ta.te_cooldowns[1][0] is None)

        ta.reset_cooldowns()
        self.assertTrue(ta.te_cooldowns[0][0] is None)
        self.assertTrue(ta.te_cooldowns[1][0] is None)

    def test_s11n(self):
        from . import (
            nt_event_batch,
            t_event_batch,
            make_vars,
            sin,
            taylor_adaptive_batch,
        )
        import numpy as np
        import pickle

        x, v = make_vars("x", "v")

        # Use a pendulum for testing purposes.
        sys = [(x, v), (v, -9.8 * sin(x))]

        def cb0(ta, t, d_sgn, bidx):
            pass

        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0, 0.01], [0.25, 0.26]],
            nt_events=[nt_event_batch(v * v - 1e-10, cb0)],
        )

        ta.step()
        ta.step()
        ta.step()
        ta.step()

        ta2 = pickle.loads(pickle.dumps(ta))

        self.assertTrue(np.all(ta.state == ta2.state))
        self.assertTrue(np.all(ta.time == ta2.time))

        self.assertEqual(len(ta.t_events), len(ta2.t_events))
        self.assertEqual(len(ta.nt_events), len(ta2.nt_events))

        ta.step()
        ta2.step()

        self.assertTrue(np.all(ta.state == ta2.state))
        self.assertTrue(np.all(ta.time == ta2.time))

        ta = taylor_adaptive_batch(sys=sys, state=[[0, 0.01], [0.25, 0.26]], tol=1e-6)

        self.assertEqual(ta.tol, 1e-6)

        # Test dynamic attributes.
        ta.foo = "hello world"
        ta = pickle.loads(pickle.dumps(ta))
        self.assertEqual(ta.foo, "hello world")

        # Try also an integrator with stateful event callback.
        class cb1:
            def __init__(self):
                self.n = 0

            def __call__(self, ta, bool, d_sgn, bidx):
                self.n = self.n + 1

                return True

        clb = cb1()
        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0, 0.01], [0.25, 0.26]],
            t_events=[t_event_batch(v, callback=clb)],
        )

        self.assertNotEqual(id(clb), id(ta.t_events[0].callback))

        self.assertEqual(ta.t_events[0].callback.n, 0)

        ta.propagate_until([100.0, 100.0])

        ta2 = pickle.loads(pickle.dumps(ta))

        self.assertEqual(ta.t_events[0].callback.n, ta2.t_events[0].callback.n)

    def test_propagate_grid(self):
        from . import make_vars, taylor_adaptive, taylor_adaptive_batch, sin
        import numpy as np
        from copy import deepcopy

        x, v = make_vars("x", "v")
        eqns = [(x, v), (v, -9.8 * sin(x))]

        x_ic = [0.06, 0.07, 0.08, 0.09]
        v_ic = [0.025, 0.026, 0.027, 0.028]

        ta = taylor_adaptive_batch(eqns, [x_ic, v_ic])

        # Failure modes.
        with self.assertRaises(ValueError) as cm:
            ta.propagate_grid([])
        self.assertTrue(
            "Invalid grid passed to the propagate_grid() method of a batch integrator: "
            "the expected number of dimensions is 2, but the input array has a dimension of 1"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            ta.propagate_grid([[1, 2], [3, 4]])
        self.assertTrue(
            "Invalid grid passed to the propagate_grid() method of a batch integrator: "
            "the shape must be (n, 4) but the number of columns is 2 instead"
            in str(cm.exception)
        )

        # Run a simple scalar/batch comparison.
        tas = []

        for x0, v0 in zip(x_ic, v_ic):
            tas.append(taylor_adaptive(eqns, [x0, v0]))

        grid = np.array(
            [
                [-0.1, -0.2, -0.3, -0.4],
                [0.01, 0.02, 0.03, 0.9],
                [1.0, 1.1, 1.2, 1.3],
                [11.0, 11.1, 11.2, 11.3],
            ]
        )

        bres = ta.propagate_grid(grid)

        sres = [
            tas[0].propagate_grid(grid[:, 0]),
            tas[1].propagate_grid(grid[:, 1]),
            tas[2].propagate_grid(grid[:, 2]),
            tas[3].propagate_grid(grid[:, 3]),
        ]

        self.assertTrue(np.max(np.abs(sres[0][4] - bres[:, :, 0]).flatten()) < 1e-14)
        self.assertTrue(np.max(np.abs(sres[1][4] - bres[:, :, 1]).flatten()) < 1e-14)
        self.assertTrue(np.max(np.abs(sres[2][4] - bres[:, :, 2]).flatten()) < 1e-14)
        self.assertTrue(np.max(np.abs(sres[3][4] - bres[:, :, 3]).flatten()) < 1e-14)

        # Test vector/scalar max_delta_t.
        ta.set_time(0.0)
        ta.state[:] = [x_ic, v_ic]

        bres = ta.propagate_grid(grid, max_delta_t=[1e-3] * 4)
        res = deepcopy(ta.propagate_res)

        ta.set_time(0.0)
        ta.state[:] = [x_ic, v_ic]

        bres2 = ta.propagate_grid(grid, max_delta_t=1e-3)

        self.assertTrue(np.all(bres == bres2))
        self.assertEqual(ta.propagate_res, res)

        # Test that adding dynattrs to the integrator
        # object via the propagate callback works.
        def cb(ta):
            if hasattr(ta, "counter"):
                ta.counter += 1
            else:
                ta.counter = 0

            return True

        ta.set_time(0.0)
        ta.propagate_grid(grid, callback=cb)

        self.assertTrue(ta.counter > 0)

        # Test that no copies of the callback are performed.
        class cb:
            def __call__(_, ta):
                self.assertEqual(id(_), _.orig_id)

                return True

        cb_inst = cb()
        cb_inst.orig_id = id(cb_inst)

        ta.set_time(0.0)
        ta.propagate_grid(grid, callback=cb_inst)


class kepE_test_case(_ut.TestCase):
    def test_expr(self):
        from . import kepE, diff, make_vars, sin, cos, core
        from .core import _ppc_arch
        import numpy as np

        x, y = make_vars("x", "y")
        self.assertEqual(
            diff(kepE(x, y), x), sin(kepE(x, y)) / (1.0 - x * cos(kepE(x, y)))
        )
        self.assertEqual(diff(kepE(x, y), y), 1.0 / (1.0 - x * cos(kepE(x, y))))

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


class sympy_test_case(_ut.TestCase):
    def test_basic(self):
        try:
            import sympy
        except ImportError:
            return

        from . import from_sympy, make_vars, sum as hsum

        with self.assertRaises(TypeError) as cm:
            from_sympy(3.5)
        self.assertTrue(
            "The 'ex' parameter must be a sympy expression but it is of type"
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            from_sympy(sympy.Symbol("x"), [])
        self.assertTrue(
            "The 's_dict' parameter must be a dict but it is of type"
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            from_sympy(sympy.Symbol("x"), {3.5: 3.5})
        self.assertTrue(
            "The keys in 's_dict' must all be sympy expressions" in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            from_sympy(sympy.Symbol("x"), {sympy.Symbol("x"): 3.5})
        self.assertTrue(
            "The values in 's_dict' must all be heyoka expressions" in str(cm.exception)
        )

        # Test the s_dict functionality of from_sympy().
        x, y = sympy.symbols("x y", real=True)
        hx, hy, hz = make_vars("x", "y", "z")

        self.assertEqual(
            from_sympy((x - y) * (x + y), s_dict={x - y: hz}), hsum([hx, hy]) * hz
        )

    def test_number_conversion(self):
        try:
            import sympy
        except ImportError:
            return

        from . import to_sympy, from_sympy, expression, core
        from .core import _ppc_arch
        from sympy import Float, Rational, Integer
        from mpmath import workprec
        import numpy as np

        with self.assertRaises(ValueError) as cm:
            from_sympy(Rational(3, 5))
        self.assertTrue(
            "Cannot convert from sympy a rational number whose denominator is not a power of 2"
            in str(cm.exception)
        )

        # From integer.
        self.assertEqual(from_sympy(Integer(-42)), expression(-42.0))

        # From rational.
        self.assertEqual(from_sympy(Rational(42, -2)), expression(-21.0))

        # Double precision.
        with workprec(53):
            self.assertEqual(to_sympy(expression(1.1)), Float(1.1))
            self.assertEqual(from_sympy(Float(1.1)), expression(1.1))

            self.assertEqual(
                to_sympy(expression((2**40 + 1) / (2**128))),
                Rational(2**40 + 1, 2**128),
            )
            self.assertEqual(
                from_sympy(Rational(2**40 + 1, 2**128)),
                expression((2**40 + 1) / (2**128)),
            )

        # Long double precision.
        if not _ppc_arch:
            with workprec(np.finfo(np.longdouble).nmant + 1):
                self.assertEqual(
                    to_sympy(expression(np.longdouble("1.1"))),
                    Float("1.1", precision=np.finfo(np.longdouble).nmant + 1),
                )

                # NOTE: on platforms where long double is not wider than
                # double (e.g., MSVC), conversion from sympy will produce a double
                # and these tests will fail.
                if np.finfo(np.longdouble).nmant > np.finfo(float).nmant:
                    self.assertEqual(
                        from_sympy(Float("1.1")), expression(np.longdouble("1.1"))
                    )

                    expo = np.finfo(np.longdouble).nmant - 10
                    self.assertEqual(
                        to_sympy(
                            expression(
                                np.longdouble(2**expo + 1) / np.longdouble(2**128)
                            )
                        ),
                        Rational(2**expo + 1, 2**128),
                    )
                    self.assertEqual(
                        from_sympy(Rational(2**expo + 1, 2**128)),
                        expression(
                            np.longdouble(2**expo + 1) / np.longdouble(2**128)
                        ),
                    )

        # Too high precision.
        if not hasattr(core, "real"):
            with self.assertRaises(ValueError) as cm:
                from_sympy(Integer(2**500 + 1))
            self.assertTrue("the required precision" in str(cm.exception))

        if not hasattr(core, "real128") or _ppc_arch:
            return

        from .core import real128

        # Quad precision.
        with workprec(113):
            self.assertEqual(
                to_sympy(expression(real128("1.1"))), Float("1.1", precision=113)
            )
            self.assertEqual(from_sympy(Float("1.1")), expression(real128("1.1")))

            expo = 100
            self.assertEqual(
                to_sympy(expression(real128(2**expo + 1) / real128(2**128))),
                Rational(2**expo + 1, 2**128),
            )
            self.assertEqual(
                from_sympy(Rational(2**expo + 1, 2**128)),
                expression(real128(2**expo + 1) / real128(2**128)),
            )

    def test_sympar_conversion(self):
        try:
            import sympy
        except ImportError:
            return

        from . import to_sympy, from_sympy, expression, par
        from sympy import Symbol

        self.assertEqual(Symbol("x", real=True), to_sympy(expression("x")))
        self.assertEqual(Symbol("par[0]", real=True), to_sympy(par[0]))
        self.assertEqual(Symbol("par[9]", real=True), to_sympy(par[9]))
        self.assertEqual(Symbol("par[123]", real=True), to_sympy(par[123]))
        self.assertEqual(
            Symbol("par[-123]", real=True), to_sympy(expression("par[-123]"))
        )
        self.assertEqual(Symbol("par[]", real=True), to_sympy(expression("par[]")))

        self.assertEqual(from_sympy(Symbol("x")), expression("x"))
        self.assertEqual(from_sympy(Symbol("par[0]")), par[0])
        self.assertEqual(from_sympy(Symbol("par[9]")), par[9])
        self.assertEqual(from_sympy(Symbol("par[123]")), par[123])
        self.assertEqual(from_sympy(Symbol("par[-123]")), expression("par[-123]"))
        self.assertEqual(from_sympy(Symbol("par[]")), expression("par[]"))

    def test_func_conversion(self):
        try:
            import sympy
        except ImportError:
            return

        import sympy as spy

        # NOTE: if we ever change in heyoka addition to return a sum(),
        # we can probably get rid of hsum.
        from . import (
            core,
            make_vars,
            from_sympy,
            to_sympy,
            pi,
            sum as hsum,
            sum_sq,
            make_nbody_sys,
        )

        x, y, z, a, b, c = spy.symbols("x y z a b c", real=True)
        hx, hy, hz, ha, hb, hc = make_vars("x", "y", "z", "a", "b", "c")

        self.assertEqual(core.acos(hx), from_sympy(spy.acos(x)))
        self.assertEqual(to_sympy(core.acos(hx)), spy.acos(x))

        self.assertEqual(core.acosh(hx), from_sympy(spy.acosh(x)))
        self.assertEqual(to_sympy(core.acosh(hx)), spy.acosh(x))

        self.assertEqual(core.asin(hx), from_sympy(spy.asin(x)))
        self.assertEqual(to_sympy(core.asin(hx)), spy.asin(x))

        self.assertEqual(core.asinh(hx), from_sympy(spy.asinh(x)))
        self.assertEqual(to_sympy(core.asinh(hx)), spy.asinh(x))

        self.assertEqual(core.atan(hx), from_sympy(spy.atan(x)))
        self.assertEqual(to_sympy(core.atan(hx)), spy.atan(x))

        self.assertEqual(core.atan2(hy, hx), from_sympy(spy.atan2(y, x)))
        self.assertEqual(to_sympy(core.atan2(hy, hx)), spy.atan2(y, x))

        self.assertEqual(core.atanh(hx), from_sympy(spy.atanh(x)))
        self.assertEqual(to_sympy(core.atanh(hx)), spy.atanh(x))

        self.assertEqual(core.cos(hx), from_sympy(spy.cos(x)))
        self.assertEqual(to_sympy(core.cos(hx)), spy.cos(x))

        self.assertEqual(core.cosh(hx), from_sympy(spy.cosh(x)))
        self.assertEqual(to_sympy(core.cosh(hx)), spy.cosh(x))

        self.assertEqual(core.erf(hx), from_sympy(spy.erf(x)))
        self.assertEqual(to_sympy(core.erf(hx)), spy.erf(x))

        self.assertEqual(core.exp(hx), from_sympy(spy.exp(x)))
        self.assertEqual(to_sympy(core.exp(hx)), spy.exp(x))

        self.assertEqual(core.log(hx), from_sympy(spy.log(x)))
        self.assertEqual(to_sympy(core.log(hx)), spy.log(x))

        self.assertEqual(core.sin(hx), from_sympy(spy.sin(x)))
        self.assertEqual(to_sympy(core.sin(hx)), spy.sin(x))

        self.assertEqual(core.sinh(hx), from_sympy(spy.sinh(x)))
        self.assertEqual(to_sympy(core.sinh(hx)), spy.sinh(x))

        self.assertEqual(core.sqrt(hx), from_sympy(spy.sqrt(x)))
        self.assertEqual(to_sympy(core.sqrt(hx)), spy.sqrt(x))

        self.assertEqual(core.tan(hx), from_sympy(spy.tan(x)))
        self.assertEqual(to_sympy(core.tan(hx)), spy.tan(x))

        self.assertEqual(core.tanh(hx), from_sympy(spy.tanh(x)))
        self.assertEqual(to_sympy(core.tanh(hx)), spy.tanh(x))

        self.assertEqual(hx**3.5, from_sympy(x**3.5))
        self.assertEqual(to_sympy(hx**3.5), x**3.5)

        self.assertEqual(hsum([hx, hy, hz]), from_sympy(x + y + z))
        self.assertEqual(to_sympy(hx + hy + hz), x + y + z)
        self.assertEqual(to_sympy(hsum([hx, hy, hz])), x + y + z)
        self.assertEqual(to_sympy(hsum([hx])), x)
        self.assertEqual(to_sympy(hsum([])), 0.0)
        self.assertEqual(
            hsum([ha, hb, hc, hx, hy, hz]), from_sympy(x + y + z + a + b + c)
        )
        self.assertEqual(to_sympy(ha + hb + hc + hx + hy + hz), x + y + z + a + b + c)
        self.assertEqual(
            to_sympy(hsum([ha, hb, hc, hx, hy, hz])), x + y + z + a + b + c
        )

        self.assertEqual(to_sympy(sum_sq([hx, hy, hz])), x * x + y * y + z * z)
        self.assertEqual(to_sympy(sum_sq([hx])), x * x)
        self.assertEqual(to_sympy(sum_sq([])), 0.0)

        self.assertEqual(hx * hy * hz, from_sympy(x * y * z))
        self.assertEqual(to_sympy(hx * hy * hz), x * y * z)
        self.assertEqual(
            (ha * hb) * (hc * hx) * (hy * hz), from_sympy(x * y * z * a * b * c)
        )
        self.assertEqual(to_sympy(ha * hb * hc * hx * hy * hz), x * y * z * a * b * c)

        self.assertEqual(hsum([hx, -1.0 * hy, -1.0 * hz]), from_sympy(x - y - z))
        self.assertEqual(to_sympy(hx - hy - hz), x - y - z)

        self.assertEqual(hx * hz**-1.0, from_sympy(x / z))
        self.assertEqual(to_sympy(hx / hz), x / z)

        self.assertEqual(
            core.kepE(hx, hy), from_sympy(spy.Function("heyoka_kepE")(x, y))
        )
        self.assertEqual(to_sympy(core.kepE(hx, hy)), spy.Function("heyoka_kepE")(x, y))

        self.assertEqual(-1.0 * hx, from_sympy(-x))
        self.assertEqual(to_sympy(-hx), -x)

        self.assertEqual(to_sympy(core.sigmoid(hx + hy)), 1.0 / (1.0 + spy.exp(-x - y)))

        self.assertEqual(core.square(hx + hy), (hx + hy) * (hx + hy))

        self.assertEqual(core.time, from_sympy(spy.Function("heyoka_time")()))
        self.assertEqual(to_sympy(core.time), spy.Function("heyoka_time")())

        self.assertEqual(
            core.tpoly(core.par[0], core.par[10]),
            from_sympy(
                spy.Function("heyoka_tpoly")(
                    spy.Symbol("par[0]"), spy.Symbol("par[10]")
                )
            ),
        )
        self.assertEqual(
            to_sympy(core.tpoly(core.par[0], core.par[10])),
            spy.Function("heyoka_tpoly")(
                spy.Symbol("par[0]", real=True), spy.Symbol("par[10]", real=True)
            ),
        )

        with self.assertRaises(TypeError) as cm:
            from_sympy(abs(x))
        self.assertTrue("Unable to convert the sympy object" in str(cm.exception))

        # Test caching behaviour.
        foo = hx + hy
        bar = foo / (foo * hz + 1.0)
        bar_spy = to_sympy(bar)
        self.assertEqual(
            id(bar_spy.args[1]), id(bar_spy.args[0].args[0].args[1].args[1])
        )

        # pi constant.
        self.assertEqual(to_sympy(pi), spy.pi)
        self.assertEqual(from_sympy(spy.pi), pi)
        self.assertEqual(to_sympy(from_sympy(spy.pi)), spy.pi)

        # nbody helper.
        [to_sympy(_[1]) for _ in make_nbody_sys(2)]
        [to_sympy(_[1]) for _ in make_nbody_sys(4)]
        [to_sympy(_[1]) for _ in make_nbody_sys(10)]


class zero_division_error_test_case(_ut.TestCase):
    def test_basic(self):
        from . import make_vars

        (x,) = make_vars("x")

        with self.assertRaises(ZeroDivisionError) as cm:
            x / 0.0
        self.assertTrue("Division by zero" in str(cm.exception))


class expression_test_case(_ut.TestCase):
    def test_basic(self):
        from . import expression as ex, core
        import numpy as np

        with_real128 = hasattr(core, "real128")
        ld_63bit = np.finfo(np.longdouble).nmant == 63

        if with_real128:
            real128 = core.real128

        # Constructors.
        self.assertEqual(ex(), ex(0))
        self.assertEqual(ex(123), ex(123.0))
        self.assertEqual(str(ex(123)), "123.00000000000000")

        # Error with large integer.
        with self.assertRaises(TypeError) as cm:
            ex(123 << 56)
        self.assertTrue("incompatible constructor arguments" in str(cm.exception))

        self.assertEqual(str(ex(1.1)), "1.1000000000000001")

        if ld_63bit:
            self.assertEqual(str(ex(np.longdouble("1.1"))), "1.10000000000000000002")

        if with_real128:
            self.assertEqual(
                str(ex(real128("1.1"))), "1.10000000000000000000000000000000008"
            )

        self.assertEqual(str(ex("x")), "x")

        # Unary arithmetics.
        self.assertEqual(ex(42), +ex(42))
        self.assertEqual(ex(-42), -ex(42))

        # Addition.
        self.assertEqual(ex(42) + ex(-1), ex(41))
        self.assertEqual(ex(42) + -1, ex(41))
        self.assertEqual(-1 + ex(42), ex(41))
        with self.assertRaises(TypeError) as cm:
            ex(42) + (2 << 112)
        with self.assertRaises(TypeError) as cm:
            (2 << 112) + ex(42)
        self.assertEqual(ex(42) + -1.1, ex(40.899999999999999))
        self.assertEqual(-1.1 + ex(42), ex(40.899999999999999))
        if ld_63bit:
            self.assertEqual(
                ex(42) + np.longdouble("-1.1"),
                ex(np.longdouble("40.9000000000000000014")),
            )
            self.assertEqual(
                np.longdouble("-1.1") + ex(42),
                ex(np.longdouble("40.9000000000000000014")),
            )
        if with_real128:
            self.assertEqual(
                ex(42) + real128("-1.1"),
                ex(real128("40.8999999999999999999999999999999988")),
            )
            self.assertEqual(
                real128("-1.1") + ex(42),
                ex(real128("40.8999999999999999999999999999999988")),
            )

        # Subtraction.
        self.assertEqual(ex(42) - ex(-1), ex(43))
        self.assertEqual(ex(42) - -1, ex(43))
        self.assertEqual(-1 - ex(42), ex(-43))
        with self.assertRaises(TypeError) as cm:
            ex(42) - (2 << 112)
        with self.assertRaises(TypeError) as cm:
            (2 << 112) - ex(42)
        self.assertEqual(ex(42) - -1.1, ex(43.100000000000001))
        self.assertEqual(-1.1 - ex(42), ex(-43.100000000000001))
        if ld_63bit:
            self.assertEqual(
                ex(42) - np.longdouble("-1.1"),
                ex(np.longdouble("43.0999999999999999986")),
            )
            self.assertEqual(
                np.longdouble("-1.1") - ex(42),
                ex(np.longdouble("-43.0999999999999999986")),
            )
        if with_real128:
            self.assertEqual(
                ex(42) - real128("-1.1"),
                ex(real128("43.1000000000000000000000000000000012")),
            )
            self.assertEqual(
                real128("-1.1") - ex(42),
                ex(real128("-43.1000000000000000000000000000000012")),
            )

        # Multiplication.
        self.assertEqual(ex(42) * ex(-1), ex(-42))
        self.assertEqual(ex(42) * -1, ex(-42))
        self.assertEqual(-1 * ex(42), ex(-42))
        with self.assertRaises(TypeError) as cm:
            ex(42) * (2 << 112)
        with self.assertRaises(TypeError) as cm:
            (2 << 112) * ex(42)
        self.assertEqual(ex(42) * -1.1, ex(-46.200000000000003))
        self.assertEqual(-1.1 * ex(42), ex(-46.200000000000003))
        if ld_63bit:
            self.assertEqual(
                ex(42) * np.longdouble("-1.1"),
                ex(np.longdouble("-46.2000000000000000007")),
            )
            self.assertEqual(
                np.longdouble("-1.1") * ex(42),
                ex(np.longdouble("-46.2000000000000000007")),
            )
        if with_real128:
            self.assertEqual(
                ex(42) * real128("-1.1"),
                ex(real128("-46.2000000000000000000000000000000025")),
            )
            self.assertEqual(
                real128("-1.1") * ex(42),
                ex(real128("-46.2000000000000000000000000000000025")),
            )

        # Division.
        self.assertEqual(ex(42) / ex(-1), ex(-42))
        self.assertEqual(ex(42) / -1, ex(-42))
        self.assertEqual(-42 / ex(1), ex(-42))
        with self.assertRaises(TypeError) as cm:
            ex(42) / (2 << 112)
        with self.assertRaises(TypeError) as cm:
            (2 << 112) / ex(42)
        self.assertEqual(ex(42) / -1.1, ex(-38.181818181818180))
        self.assertEqual(-1.1 / ex(42), ex(-0.02619047619047619))
        if ld_63bit:
            self.assertEqual(
                ex(42) / np.longdouble("-1.1"),
                ex(np.longdouble("-38.1818181818181818163")),
            )
            self.assertEqual(
                np.longdouble("-1.1") / ex(42),
                ex(np.longdouble("-0.0261904761904761904772")),
            )
        if with_real128:
            self.assertEqual(
                ex(42) / real128("-1.1"),
                ex(real128("-38.1818181818181818181818181818181801")),
            )
            self.assertEqual(
                real128("-1.1") / ex(42),
                ex(real128("-0.0261904761904761904761904761904761910")),
            )

        # Comparison.
        self.assertEqual(ex(42), ex(42))
        self.assertEqual(ex("x"), ex("x"))
        self.assertNotEqual(ex(41), ex(42))
        self.assertNotEqual(ex("x"), ex("y"))

        # Exponentiation.
        self.assertEqual(str(ex("x") ** ex("y")), "pow(x, y)")
        self.assertEqual(str(ex("x") ** ex(2)), "x**2")
        self.assertEqual(str(ex("x") ** ex(1.1)), "pow(x, 1.1000000000000001)")
        with self.assertRaises(TypeError) as cm:
            ex(42) ** (2 << 112)
        if ld_63bit:
            self.assertEqual(
                ex(42) / np.longdouble("-1.1"),
                ex(np.longdouble("-38.1818181818181818163")),
            )
        if ld_63bit:
            self.assertEqual(
                str(ex("x") ** ex(np.longdouble("1.1"))),
                "pow(x, 1.10000000000000000002)",
            )
        if with_real128:
            self.assertEqual(
                str(ex("x") ** ex(real128("1.1"))),
                "pow(x, 1.10000000000000000000000000000000008)",
            )

        # Copy and deepcopy.
        from copy import copy, deepcopy

        tmp = ex("x") + ex("y")
        tmp.foo = [1, 2, 3]
        id_foo = id(tmp.foo)
        tmp_copy = copy(tmp)
        self.assertEqual(id_foo, id(tmp_copy.foo))
        tmp_dcopy = deepcopy(tmp)
        self.assertNotEqual(id_foo, id(tmp_dcopy.foo))
        self.assertEqual(tmp.foo, tmp_dcopy.foo)

    def test_copy(self):
        from . import make_vars
        from copy import copy, deepcopy

        x, y = make_vars("x", "y")
        ex = x + y

        class foo:
            pass

        ex.bar = foo()

        self.assertEqual(id(ex.bar), id(copy(ex).bar))
        self.assertNotEqual(id(ex.bar), id(deepcopy(ex).bar))
        self.assertEqual(ex, copy(ex))
        self.assertEqual(ex, deepcopy(ex))

    def test_diff(self):
        from . import make_vars, sin, cos, diff, par

        x, y = make_vars("x", "y")
        self.assertEqual(diff(cos(x * x - y), "x"), -sin(x * x - y) * (2.0 * x))
        self.assertEqual(diff(cos(x * x - y), x), -sin(x * x - y) * (2.0 * x))
        self.assertEqual(
            diff(cos(par[0] * par[0] - y), par[0]),
            -sin(par[0] * par[0] - y) * (2.0 * par[0]),
        )

    def test_s11n(self):
        from . import make_vars, sin, cos, core
        from .core import _ppc_arch
        from numpy import longdouble
        import pickle

        x, y = make_vars("x", "y")

        ex = x + 2.0 * y
        self.assertEqual(ex, pickle.loads(pickle.dumps(ex)))

        # Test dynamic attributes.
        ex.foo = "hello world"
        ex = pickle.loads(pickle.dumps(ex))
        self.assertEqual(ex.foo, "hello world")

        if not _ppc_arch:
            ex = sin(longdouble("1.1") * x) + 2.0 * y
            self.assertEqual(ex, pickle.loads(pickle.dumps(ex)))

        if not hasattr(core, "real128"):
            return

        from .core import real128

        # Quad precision.
        if not _ppc_arch:
            ex = sin(longdouble("1.1") * x) + real128("1.3") * cos(2.0 * y)
            self.assertEqual(ex, pickle.loads(pickle.dumps(ex)))

    def test_len(self):
        from . import make_vars

        x, y, z = make_vars("x", "y", "z")

        self.assertEqual(len(x), 1)
        self.assertEqual(len((x - y - z) + (y * z)), 9)


class llvm_state_test_case(_ut.TestCase):
    def test_copy(self):
        from . import make_vars, sin, taylor_adaptive
        from copy import copy, deepcopy

        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]

        ta = taylor_adaptive(sys=sys, state=[0.0, 0.25])

        ls = ta.llvm_state

        class foo:
            pass

        ls.bar = foo()

        self.assertEqual(id(ls.bar), id(copy(ls).bar))
        self.assertNotEqual(id(ls.bar), id(deepcopy(ls).bar))
        self.assertEqual(ls.get_ir(), copy(ls).get_ir())
        self.assertEqual(ls.get_ir(), deepcopy(ls).get_ir())

    def test_s11n(self):
        from . import make_vars, sin, taylor_adaptive
        import pickle
        from sys import getrefcount

        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]

        ta = taylor_adaptive(sys=sys, state=[0.0, 0.25])

        # Verify that the reference count of ta
        # is increased when we fetch the llvm_state.
        rc = getrefcount(ta)
        ls = ta.llvm_state
        self.assertEqual(getrefcount(ta), rc + 1)

        self.assertEqual(ls.get_ir(), pickle.loads(pickle.dumps(ls)).get_ir())

        # Test dynamic attributes.
        ls.foo = "hello world"
        ls = pickle.loads(pickle.dumps(ls))
        self.assertEqual(ls.foo, "hello world")


class c_output_test_case(_ut.TestCase):
    def test_batch(self):
        from copy import copy, deepcopy
        from . import (
            make_vars,
            sin,
            taylor_adaptive_batch,
            continuous_output_batch_dbl,
            taylor_adaptive,
        )
        from pickle import dumps, loads
        from sys import getrefcount
        import numpy as np

        x, v = make_vars("x", "v")

        fp_types = [(float, continuous_output_batch_dbl)]

        # Use a pendulum for testing purposes.
        sys = [(x, v), (v, -9.8 * sin(x))]

        for fp_t, c_out_t in fp_types:
            # Test the default cted object.
            c_out = c_out_t()

            with self.assertRaises(ValueError) as cm:
                c_out([])
            self.assertTrue(
                "Cannot use a default-constructed continuous_output_batch object"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                c_out(fp_t(1))
            self.assertTrue(
                "Cannot use a default-constructed continuous_output_batch object"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                c_out(time=[fp_t(0), fp_t(0)])
            self.assertTrue(
                "Cannot use a default-constructed continuous_output_batch object"
                in str(cm.exception)
            )

            self.assertTrue(c_out.output is None)
            self.assertTrue(c_out.times is None)
            self.assertTrue(c_out.tcs is None)

            with self.assertRaises(ValueError) as cm:
                c_out.bounds
            self.assertTrue(
                "Cannot use a default-constructed continuous_output_batch object"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                c_out.n_steps
            self.assertTrue(
                "Cannot use a default-constructed continuous_output_batch object"
                in str(cm.exception)
            )

            self.assertEqual(c_out.batch_size, 0)

            self.assertFalse("forward" in repr(c_out))

            self.assertFalse(c_out.llvm_state.get_ir() == "")

            # Try copies as well.
            c_out = copy(c_out)

            with self.assertRaises(ValueError) as cm:
                c_out.n_steps
            self.assertTrue(
                "Cannot use a default-constructed continuous_output_batch object"
                in str(cm.exception)
            )

            self.assertEqual(c_out.batch_size, 0)

            self.assertFalse("forward" in repr(c_out))

            self.assertFalse(c_out.llvm_state.get_ir() == "")

            c_out = deepcopy(c_out)

            with self.assertRaises(ValueError) as cm:
                c_out.n_steps
            self.assertTrue(
                "Cannot use a default-constructed continuous_output_batch object"
                in str(cm.exception)
            )

            self.assertEqual(c_out.batch_size, 0)

            self.assertFalse("forward" in repr(c_out))

            self.assertFalse(c_out.llvm_state.get_ir() == "")

            # Pickling.
            c_out = loads(dumps(c_out))

            with self.assertRaises(ValueError) as cm:
                c_out.n_steps
            self.assertTrue(
                "Cannot use a default-constructed continuous_output_batch object"
                in str(cm.exception)
            )

            self.assertEqual(c_out.batch_size, 0)

            self.assertFalse("forward" in repr(c_out))

            self.assertFalse(c_out.llvm_state.get_ir() == "")

            ic = [
                [fp_t(0), fp_t(0.01), fp_t(0.02), fp_t(0.03)],
                [fp_t(0.25), fp_t(0.26), fp_t(0.27), fp_t(0.28)],
            ]

            arr_ic = np.array(ic)

            ta = taylor_adaptive_batch(sys=sys, state=ic, fp_type=fp_t)

            # Create scalar integrators for comparison.
            ta_scalar = taylor_adaptive(
                sys=sys, state=[ic[0][0], ic[1][0]], fp_type=fp_t
            )
            ta_scals = [deepcopy(ta_scalar) for _ in range(4)]

            # Helper to reset the state of ta and ta_scals.
            def reset():
                ta.state[:] = ic
                ta.set_time([fp_t(0)] * 4)

                for idx, tint in enumerate(ta_scals):
                    tint.state[:] = arr_ic[:, idx]
                    tint.time = 0

            final_tm = [fp_t(10), fp_t(10.4), fp_t(10.5), fp_t(11.0)]

            check_tm = [fp_t(0.1), fp_t(1.3), fp_t(5.6), fp_t(9.1)]

            c_out = ta.propagate_until(final_tm)

            self.assertTrue(c_out is None)

            reset()

            c_out = ta.propagate_until(final_tm, c_output=False)

            self.assertTrue(c_out is None)

            reset()

            c_out = ta.propagate_until(final_tm, c_output=True)

            self.assertFalse(c_out is None)

            self.assertTrue(c_out(check_tm).shape == (2, 4))

            with self.assertRaises(ValueError) as cm:
                c_out(check_tm)[0] = 0.5

            rc = getrefcount(c_out)
            tmp_out = c_out(check_tm)
            new_rc = getrefcount(c_out)
            self.assertEqual(new_rc, rc + 1)

            with self.assertRaises(ValueError) as cm:
                c_out(np.zeros((1, 1, 1)))
            self.assertTrue(
                "Invalid time array passed to a continuous_output_batch object: the number of dimensions must be 1 or 2, but it is 3 instead"
                in str(cm.exception)
            )

            # Single batch tests.
            with self.assertRaises(ValueError) as cm:
                c_out(np.zeros((1,)))
            self.assertTrue(
                "Invalid time array passed to a continuous_output_batch object: the length must be 4 but it is 1 instead"
                in str(cm.exception)
            )
            with self.assertRaises(ValueError) as cm:
                c_out(np.zeros((0,)))
            self.assertTrue(
                "Invalid time array passed to a continuous_output_batch object: the length must be 4 but it is 0 instead"
                in str(cm.exception)
            )
            with self.assertRaises(ValueError) as cm:
                c_out(np.zeros((5,)))
            self.assertTrue(
                "Invalid time array passed to a continuous_output_batch object: the length must be 4 but it is 5 instead"
                in str(cm.exception)
            )

            # Contiguous single batch.
            c_out_scals = [
                ta_scals[idx].propagate_until(final_tm[idx], c_output=True)[4]
                for idx in range(4)
            ]
            c_out(check_tm)

            for idx in range(4):
                c_out_scals[idx](check_tm[idx])
                self.assertTrue(
                    np.allclose(
                        c_out_scals[idx].output,
                        c_out.output[:, idx],
                        rtol=np.finfo(fp_t).eps * 10,
                        atol=np.finfo(fp_t).eps * 10,
                    )
                )

            rc = getrefcount(c_out)
            tmp_out2 = c_out(check_tm)
            new_rc = getrefcount(c_out)
            self.assertEqual(new_rc, rc + 1)

            # Scalar time.
            scal_res = deepcopy(c_out(fp_t(0.42)))
            self.assertTrue(np.all(scal_res == c_out([fp_t(0.42)] * 4)))

            # Non-contiguous single batch.
            nc_check_tm = np.vstack([check_tm, np.zeros((4,))]).T.flatten()[::2]
            c_out(nc_check_tm)
            for idx in range(4):
                self.assertTrue(
                    np.allclose(
                        c_out_scals[idx].output,
                        c_out.output[:, idx],
                        rtol=np.finfo(fp_t).eps * 10,
                        atol=np.finfo(fp_t).eps * 10,
                    )
                )

            # Multiple time batches.
            with self.assertRaises(ValueError) as cm:
                c_out(np.zeros((5, 3)))
            self.assertTrue(
                "Invalid time array passed to a continuous_output_batch object: the number of columns must be 4 but it is 3 instead"
                in str(cm.exception)
            )

            b_check_tm = np.repeat(check_tm, 5, axis=0).reshape((4, 5)).T
            out_b = c_out(b_check_tm)
            self.assertEqual(out_b.shape, (5, 2, 4))
            for idx in range(4):
                c_out_scals[idx](check_tm[idx])

                for j in range(5):
                    self.assertTrue(
                        np.allclose(
                            c_out_scals[idx].output,
                            out_b[j, :, idx],
                            rtol=np.finfo(fp_t).eps * 10,
                            atol=np.finfo(fp_t).eps * 10,
                        )
                    )

            # Zero rows in input.
            out_b = c_out(np.zeros((0, 4)))
            self.assertEqual(out_b.shape, (0, 2, 4))

            # Times.
            self.assertEqual(c_out.times.shape, (c_out.n_steps + 1, 4))
            self.assertTrue(np.all(np.isfinite(c_out.times)))
            with self.assertRaises(ValueError) as cm:
                c_out.times[0] = 0.5

            rc = getrefcount(c_out)
            tmp_out3 = c_out.times
            new_rc = getrefcount(c_out)
            self.assertEqual(new_rc, rc + 1)

            # TCs.
            self.assertEqual(c_out.tcs.shape, (c_out.n_steps, 2, ta.order + 1, 4))
            with self.assertRaises(ValueError) as cm:
                c_out.tcs[0] = 0.5

            rc = getrefcount(c_out)
            tmp_out4 = c_out.tcs
            new_rc = getrefcount(c_out)
            self.assertEqual(new_rc, rc + 1)

            # Bounds.
            self.assertTrue(np.all(c_out.bounds[0] == [0.0] * 4))
            self.assertTrue(
                np.allclose(
                    c_out.bounds[1],
                    final_tm,
                    rtol=np.finfo(fp_t).eps * 10,
                    atol=np.finfo(fp_t).eps * 10,
                )
            )

            # Batch size.
            self.assertEqual(c_out.batch_size, 4)

            # Repr.
            self.assertTrue("forward" in repr(c_out))

            self.assertFalse(c_out.llvm_state.get_ir() == "")

            # Try copies as well.
            c_out = copy(c_out)

            self.assertEqual(c_out.tcs.shape, (c_out.n_steps, 2, ta.order + 1, 4))

            self.assertFalse(c_out.llvm_state.get_ir() == "")

            c_out = deepcopy(c_out)

            self.assertEqual(c_out.tcs.shape, (c_out.n_steps, 2, ta.order + 1, 4))

            # Pickling.
            c_out = loads(dumps(c_out))

            self.assertFalse(c_out.llvm_state.get_ir() == "")

            self.assertEqual(c_out.tcs.shape, (c_out.n_steps, 2, ta.order + 1, 4))

            class foo:
                pass

            c_out_copy = deepcopy(c_out)
            orig_tmp = deepcopy(c_out_copy(fp_t(0.1)))
            c_out_copy.bar = foo()

            self.assertEqual(id(c_out_copy.bar), id(copy(c_out_copy).bar))
            self.assertNotEqual(id(c_out_copy.bar), id(deepcopy(c_out_copy).bar))
            self.assertTrue(
                np.all(c_out_copy(fp_t(0.1)) == copy(c_out_copy)(fp_t(0.1)))
            )
            self.assertTrue(
                np.all(c_out_copy(fp_t(0.1)) == deepcopy(c_out_copy)(fp_t(0.1)))
            )

            # Pickling with dynattrs.
            c_out.foo = []
            c_out = loads(dumps(c_out))

            self.assertFalse(c_out.llvm_state.get_ir() == "")

            self.assertEqual(c_out.tcs.shape, (c_out.n_steps, 2, ta.order + 1, 4))

            self.assertEqual(c_out.foo, [])

    def test_scalar(self):
        from copy import copy, deepcopy
        from . import make_vars, sin, taylor_adaptive, continuous_output_dbl, core
        from .core import _ppc_arch
        import numpy as np
        from pickle import dumps, loads
        from sys import getrefcount

        x, v = make_vars("x", "v")

        if _ppc_arch:
            fp_types = [(float, continuous_output_dbl)]
        else:
            from . import continuous_output_ldbl

            fp_types = [
                (float, continuous_output_dbl),
                (np.longdouble, continuous_output_ldbl),
            ]

        if hasattr(core, "real128"):
            from . import continuous_output_f128

            fp_types.append((core.real128, continuous_output_f128))

        # Use a pendulum for testing purposes.
        sys = [(x, v), (v, -9.8 * sin(x))]

        for fp_t, c_out_t in fp_types:
            # Test the default cted object.
            c_out = c_out_t()

            with self.assertRaises(ValueError) as cm:
                c_out(fp_t(0))
            self.assertTrue(
                "Cannot use a default-constructed continuous_output object"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                c_out(time=[fp_t(0), fp_t(0)])
            self.assertTrue(
                "Cannot use a default-constructed continuous_output object"
                in str(cm.exception)
            )

            self.assertTrue(c_out(time=[]).shape == (0, 0))
            self.assertTrue(c_out.output is None)
            self.assertTrue(c_out.times is None)
            self.assertTrue(c_out.tcs is None)

            with self.assertRaises(ValueError) as cm:
                c_out.bounds
            self.assertTrue(
                "Cannot use a default-constructed continuous_output object"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                c_out.n_steps
            self.assertTrue(
                "Cannot use a default-constructed continuous_output object"
                in str(cm.exception)
            )

            self.assertFalse("forward" in repr(c_out))

            self.assertFalse(c_out.llvm_state.get_ir() == "")

            # Try copies as well.
            c_out = copy(c_out)

            with self.assertRaises(ValueError) as cm:
                c_out.n_steps
            self.assertTrue(
                "Cannot use a default-constructed continuous_output object"
                in str(cm.exception)
            )

            self.assertFalse("forward" in repr(c_out))

            self.assertFalse(c_out.llvm_state.get_ir() == "")

            c_out = deepcopy(c_out)

            with self.assertRaises(ValueError) as cm:
                c_out.n_steps
            self.assertTrue(
                "Cannot use a default-constructed continuous_output object"
                in str(cm.exception)
            )

            self.assertFalse("forward" in repr(c_out))

            self.assertFalse(c_out.llvm_state.get_ir() == "")

            # Pickling.
            c_out = loads(dumps(c_out))

            with self.assertRaises(ValueError) as cm:
                c_out.n_steps
            self.assertTrue(
                "Cannot use a default-constructed continuous_output object"
                in str(cm.exception)
            )

            self.assertFalse("forward" in repr(c_out))

            self.assertFalse(c_out.llvm_state.get_ir() == "")

            ic = [fp_t(0), fp_t(0.25)]

            ta = taylor_adaptive(sys=sys, state=ic, fp_type=fp_t)

            # Helper to reset the state of ta.
            def reset():
                ta.state[:] = ic
                ta.time = fp_t(0)

            c_out = ta.propagate_until(fp_t(10))[4]

            self.assertTrue(c_out is None)

            reset()

            c_out = ta.propagate_until(fp_t(10), c_output=False)[4]

            self.assertTrue(c_out is None)

            reset()

            _, _, _, nsteps, c_out = ta.propagate_until(fp_t(10), c_output=True)

            self.assertFalse(c_out is None)

            self.assertTrue(c_out(fp_t(0.1)).shape == (2,))

            with self.assertRaises(ValueError) as cm:
                c_out(fp_t(0.1))[0] = 0.5

            rc = getrefcount(c_out)
            tmp_out = c_out(fp_t(0.1))
            new_rc = getrefcount(c_out)
            self.assertEqual(new_rc, rc + 1)

            self.assertTrue(c_out([]).shape == (0, 2))

            tmp = c_out([fp_t(0), fp_t(1), fp_t(2)])
            self.assertTrue(np.all(c_out(fp_t(0)) == tmp[0]))
            self.assertTrue(np.all(c_out(fp_t(1)) == tmp[1]))
            self.assertTrue(np.all(c_out(fp_t(2)) == tmp[2]))

            # Check wrong shape for the input array.
            with self.assertRaises(ValueError) as cm:
                c_out([[fp_t(0)], [fp_t(1)], [fp_t(2)]])
            self.assertTrue(
                "Invalid time array passed to a continuous_output object: the number of dimensions must be 1, but it is 2 instead"
                in str(cm.exception)
            )

            self.assertTrue(np.all(c_out.output == tmp[2]))
            with self.assertRaises(ValueError) as cm:
                c_out.output[0] = 0.5

            rc = getrefcount(c_out)
            tmp_out2 = c_out.output
            new_rc = getrefcount(c_out)
            self.assertEqual(new_rc, rc + 1)

            self.assertEqual(c_out.times.shape, (nsteps + 1,))
            with self.assertRaises(ValueError) as cm:
                c_out.times[0] = 0.5

            rc = getrefcount(c_out)
            tmp_out3 = c_out.times
            new_rc = getrefcount(c_out)
            self.assertEqual(new_rc, rc + 1)

            self.assertEqual(c_out.tcs.shape, (nsteps, 2, ta.order + 1))
            self.assertTrue(np.all(np.isfinite(c_out.tcs)))
            with self.assertRaises(ValueError) as cm:
                c_out.tcs[0, 0, 0] = 0.5

            rc = getrefcount(c_out)
            tmp_out4 = c_out.tcs
            new_rc = getrefcount(c_out)
            self.assertEqual(new_rc, rc + 1)

            self.assertEqual(c_out.bounds, (0, 10))
            self.assertTrue(c_out.n_steps > 0)

            self.assertTrue("forward" in repr(c_out))

            self.assertFalse(c_out.llvm_state.get_ir() == "")

            # Try copies as well.
            c_out = copy(c_out)

            self.assertEqual(c_out.tcs.shape, (nsteps, 2, ta.order + 1))
            self.assertTrue(np.all(np.isfinite(c_out.tcs)))
            with self.assertRaises(ValueError) as cm:
                c_out.tcs[0, 0, 0] = 0.5

            self.assertFalse(c_out.llvm_state.get_ir() == "")

            c_out = deepcopy(c_out)

            self.assertEqual(c_out.tcs.shape, (nsteps, 2, ta.order + 1))
            self.assertTrue(np.all(np.isfinite(c_out.tcs)))
            with self.assertRaises(ValueError) as cm:
                c_out.tcs[0, 0, 0] = 0.5

            class foo:
                pass

            c_out_copy = deepcopy(c_out)
            orig_tmp = deepcopy(c_out_copy(fp_t(0.1)))
            c_out_copy.bar = foo()

            self.assertEqual(id(c_out_copy.bar), id(copy(c_out_copy).bar))
            self.assertNotEqual(id(c_out_copy.bar), id(deepcopy(c_out_copy).bar))
            self.assertTrue(
                np.all(c_out_copy(fp_t(0.1)) == copy(c_out_copy)(fp_t(0.1)))
            )
            self.assertTrue(
                np.all(c_out_copy(fp_t(0.1)) == deepcopy(c_out_copy)(fp_t(0.1)))
            )

            # Pickling.
            c_out = loads(dumps(c_out))

            self.assertFalse(c_out.llvm_state.get_ir() == "")

            self.assertEqual(c_out.tcs.shape, (nsteps, 2, ta.order + 1))
            self.assertTrue(np.all(np.isfinite(c_out.tcs)))
            with self.assertRaises(ValueError) as cm:
                c_out.tcs[0, 0, 0] = 0.5

            # Pickling with dynattrs.
            c_out.foo = []
            c_out = loads(dumps(c_out))

            self.assertFalse(c_out.llvm_state.get_ir() == "")

            self.assertEqual(c_out.tcs.shape, (nsteps, 2, ta.order + 1))
            self.assertTrue(np.all(np.isfinite(c_out.tcs)))
            with self.assertRaises(ValueError) as cm:
                c_out.tcs[0, 0, 0] = 0.5

            self.assertEqual(c_out.foo, [])


class recommended_simd_size_test_case(_ut.TestCase):
    def test_basic(self):
        from . import recommended_simd_size

        self.assertTrue(recommended_simd_size() >= 1)
        self.assertEqual(recommended_simd_size(), recommended_simd_size(fp_type=float))


class s11n_backend_test_case(_ut.TestCase):
    def test_basic(self):
        from . import set_serialization_backend, get_serialization_backend
        import cloudpickle as cp
        import pickle as pk

        self.assertEqual(get_serialization_backend(), cp)
        set_serialization_backend("pickle")
        self.assertEqual(get_serialization_backend(), pk)
        set_serialization_backend("cloudpickle")
        self.assertEqual(get_serialization_backend(), cp)

        with self.assertRaises(TypeError) as cm:
            set_serialization_backend(1)
        self.assertTrue(
            "The serialization backend must be specified as a string"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            set_serialization_backend("pippo")
        self.assertTrue(
            "The serialization backend 'pippo' is not valid. The valid backends are:"
            in str(cm.exception)
        )

        self.assertEqual(get_serialization_backend(), cp)


class ensemble_test_case(_ut.TestCase):
    def test_batch(self):
        from . import (
            ensemble_propagate_until_batch,
            ensemble_propagate_for_batch,
            ensemble_propagate_grid_batch,
            make_vars,
            sin,
            taylor_adaptive_batch,
        )
        import numpy as np

        x, v = make_vars("x", "v")

        # Use a pendulum for testing purposes.
        sys = [(x, v), (v, -9.8 * sin(x))]

        algos = ["thread", "process"]

        ta = taylor_adaptive_batch(sys=sys, state=[[0.0] * 4] * 2)

        ics = np.zeros((10, 2, 4))
        for i in range(10):
            ics[i, 0] = [
                0.05 + i / 100,
                0.051 + i / 100,
                0.052 + i / 100,
                0.053 + i / 100.0,
            ]
            ics[i, 0] = [
                0.025 + i / 100,
                0.026 + i / 100,
                0.027 + i / 100,
                0.028 + i / 100.0,
            ]

        # propagate_until().
        def gen(ta, idx):
            ta.set_time(0.0)
            ta.state[:] = ics[idx]

            return ta

        for algo in algos:
            if algo == "thread":
                ret = ensemble_propagate_until_batch(
                    ta, 20.0, 10, gen, algorithm=algo, max_workers=8
                )
            elif algo == "process":
                ret = ensemble_propagate_until_batch(
                    ta, 20.0, 10, gen, algorithm=algo, max_workers=8, chunksize=3
                )

            self.assertEqual(len(ret), 10)

            for i in range(10):
                ta.set_time(0.0)
                ta.state[:] = ics[i]
                loc_ret = ta.propagate_until(20.0)

                for j in range(4):
                    self.assertAlmostEqual(ret[i][0].time[j], 20.0)
                self.assertTrue(np.all(ta.state == ret[i][0].state))
                self.assertTrue(ret[i][1] is None)
                self.assertTrue(np.all(ta.time == ret[i][0].time))
                self.assertEqual(ta.propagate_res, ret[i][0].propagate_res)

            # Run a test with c_output too.
            if algo == "thread":
                ret = ensemble_propagate_until_batch(
                    ta, 20.0, 10, gen, algorithm=algo, max_workers=8, c_output=True
                )
            elif algo == "process":
                ret = ensemble_propagate_until_batch(
                    ta,
                    20.0,
                    10,
                    gen,
                    algorithm=algo,
                    max_workers=8,
                    chunksize=3,
                    c_output=True,
                )

            self.assertEqual(len(ret), 10)

            for i in range(10):
                ta.set_time(0.0)
                ta.state[:] = ics[i]
                loc_ret = ta.propagate_until(20.0, c_output=True)

                for j in range(4):
                    self.assertAlmostEqual(ret[i][0].time[j], 20.0)
                self.assertTrue(np.all(ta.state == ret[i][0].state))
                self.assertFalse(ret[i][1] is None)
                self.assertTrue(np.all(ta.time == ret[i][0].time))
                self.assertEqual(ta.propagate_res, ret[i][0].propagate_res)

                self.assertTrue(np.all(loc_ret(5.0) == ret[i][1](5.0)))

        # propagate_for().
        def gen(ta, idx):
            ta.set_time(10.0)
            ta.state[:] = ics[idx]

            return ta

        for algo in algos:
            if algo == "thread":
                ret = ensemble_propagate_for_batch(
                    ta, 20.0, 10, gen, algorithm=algo, max_workers=8
                )
            elif algo == "process":
                ret = ensemble_propagate_for_batch(
                    ta, 20.0, 10, gen, algorithm=algo, max_workers=8, chunksize=3
                )

            self.assertEqual(len(ret), 10)

            for i in range(10):
                ta.set_time(10.0)
                ta.state[:] = ics[i]
                loc_ret = ta.propagate_for(20.0)

                for j in range(4):
                    self.assertAlmostEqual(ret[i][0].time[j], 30.0)
                self.assertTrue(np.all(ta.state == ret[i][0].state))
                self.assertTrue(ret[i][1] is None)
                self.assertTrue(np.all(ta.time == ret[i][0].time))
                self.assertEqual(ta.propagate_res, ret[i][0].propagate_res)

        # propagate_grid().
        grid = np.linspace(0.0, 20.0, 80)

        splat_grid = np.repeat(grid, 4).reshape(-1, 4)

        def gen(ta, idx):
            ta.set_time(0.0)
            ta.state[:] = ics[idx]

            return ta

        for algo in algos:
            if algo == "thread":
                ret = ensemble_propagate_grid_batch(
                    ta, grid, 10, gen, algorithm=algo, max_workers=8
                )
            elif algo == "process":
                ret = ensemble_propagate_grid_batch(
                    ta, grid, 10, gen, algorithm=algo, max_workers=8, chunksize=3
                )

            self.assertEqual(len(ret), 10)

            for i in range(10):
                ta.set_time(0.0)
                ta.state[:] = ics[i]
                loc_ret = ta.propagate_grid(splat_grid)

                self.assertTrue(np.all(loc_ret == ret[i][1]))

                for j in range(4):
                    self.assertAlmostEqual(ret[i][0].time[j], 20.0)
                self.assertTrue(np.all(ta.state == ret[i][0].state))
                self.assertTrue(np.all(ta.time == ret[i][0].time))
                self.assertEqual(ta.propagate_res, ret[i][0].propagate_res)

    def test_scalar(self):
        from . import (
            ensemble_propagate_until,
            ensemble_propagate_for,
            ensemble_propagate_grid,
            make_vars,
            sin,
            taylor_adaptive,
            taylor_outcome,
        )
        import numpy as np

        x, v = make_vars("x", "v")

        # Use a pendulum for testing purposes.
        sys = [(x, v), (v, -9.8 * sin(x))]

        algos = ["thread", "process"]

        ta = taylor_adaptive(sys=sys, state=[0.0] * 2)

        ics = np.array([[0.05, 0.025]] * 10)
        for i in range(10):
            ics[i] += i / 100.0

        # propagate_until().
        def gen(ta, idx):
            ta.time = 0.0
            ta.state[:] = ics[idx]

            return ta

        for algo in algos:
            if algo == "thread":
                ret = ensemble_propagate_until(
                    ta, 20.0, 10, gen, algorithm=algo, max_workers=8
                )
            elif algo == "process":
                ret = ensemble_propagate_until(
                    ta, 20.0, 10, gen, algorithm=algo, max_workers=8, chunksize=3
                )

            self.assertEqual(len(ret), 10)

            self.assertTrue(all([_[1] == taylor_outcome.time_limit for _ in ret]))

            for i in range(10):
                ta.time = 0.0
                ta.state[:] = ics[i]
                loc_ret = ta.propagate_until(20.0)

                self.assertAlmostEqual(ret[i][0].time, 20.0)
                self.assertTrue(np.all(ta.state == ret[i][0].state))
                self.assertEqual(loc_ret, ret[i][1:])
                self.assertEqual(ta.time, ret[i][0].time)

            # Run a test with c_output too.
            if algo == "thread":
                ret = ensemble_propagate_until(
                    ta, 20.0, 10, gen, algorithm=algo, max_workers=8, c_output=True
                )
            elif algo == "process":
                ret = ensemble_propagate_until(
                    ta,
                    20.0,
                    10,
                    gen,
                    algorithm=algo,
                    max_workers=8,
                    chunksize=3,
                    c_output=True,
                )

            self.assertEqual(len(ret), 10)

            self.assertTrue(all([_[1] == taylor_outcome.time_limit for _ in ret]))

            for i in range(10):
                ta.time = 0.0
                ta.state[:] = ics[i]
                loc_ret = ta.propagate_until(20.0, c_output=True)

                self.assertAlmostEqual(ret[i][0].time, 20.0)
                self.assertTrue(np.all(ta.state == ret[i][0].state))
                self.assertEqual(loc_ret[:-1], ret[i][1:-1])
                self.assertEqual(ta.time, ret[i][0].time)

                self.assertTrue(np.all(loc_ret[-1](5.0) == ret[i][-1](5.0)))

        # propagate_for().
        def gen(ta, idx):
            ta.time = 10.0
            ta.state[:] = ics[idx]

            return ta

        for algo in algos:
            if algo == "thread":
                ret = ensemble_propagate_for(
                    ta, 20.0, 10, gen, algorithm=algo, max_workers=8
                )
            elif algo == "process":
                ret = ensemble_propagate_for(
                    ta, 20.0, 10, gen, algorithm=algo, max_workers=8, chunksize=3
                )

            self.assertEqual(len(ret), 10)

            self.assertTrue(all([_[1] == taylor_outcome.time_limit for _ in ret]))

            for i in range(10):
                ta.time = 10.0
                ta.state[:] = ics[i]
                loc_ret = ta.propagate_for(20.0)

                self.assertAlmostEqual(ret[i][0].time, 30.0)
                self.assertTrue(np.all(ta.state == ret[i][0].state))
                self.assertEqual(loc_ret, ret[i][1:])
                self.assertEqual(ta.time, ret[i][0].time)

        # propagate_grid().
        grid = np.linspace(0.0, 20.0, 80)

        def gen(ta, idx):
            ta.time = 0.0
            ta.state[:] = ics[idx]

            return ta

        for algo in algos:
            if algo == "thread":
                ret = ensemble_propagate_grid(
                    ta, grid, 10, gen, algorithm=algo, max_workers=8
                )
            elif algo == "process":
                ret = ensemble_propagate_grid(
                    ta, grid, 10, gen, algorithm=algo, max_workers=8, chunksize=3
                )

            self.assertEqual(len(ret), 10)

            self.assertTrue(all([_[1] == taylor_outcome.time_limit for _ in ret]))

            for i in range(10):
                ta.time = 0.0
                ta.state[:] = ics[i]
                loc_ret = ta.propagate_grid(grid)

                self.assertAlmostEqual(ret[i][0].time, 20.0)
                self.assertTrue(np.all(ta.state == ret[i][0].state))
                self.assertEqual(loc_ret[:-1], ret[i][1:-1])
                self.assertTrue(np.all(loc_ret[-1] == ret[i][-1]))
                self.assertEqual(ta.time, ret[i][0].time)

        # Error handling.
        with self.assertRaises(TypeError) as cm:
            ensemble_propagate_until(ta, 20.0, "a", gen)
        self.assertTrue(
            "The n_iter parameter must be an integer, but an object of type"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            ensemble_propagate_until(ta, 20.0, -1, gen)
        self.assertTrue(
            "The n_iter parameter must be non-negative" in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            ensemble_propagate_until(ta, [20.0], 10, gen)
        self.assertTrue(
            "Cannot perform an ensemble propagate_until/for(): the final epoch/time interval must be a scalar, not an iterable object"
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            ensemble_propagate_for(ta, [20.0], 10, gen)
        self.assertTrue(
            "Cannot perform an ensemble propagate_until/for(): the final epoch/time interval must be a scalar, not an iterable object"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            ensemble_propagate_grid(ta, [[20.0, 20.0]], 10, gen)
        self.assertTrue(
            "Cannot perform an ensemble propagate_grid(): the input time grid must be one-dimensional, but instead it has 2 dimensions"
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            ensemble_propagate_until(ta, 20.0, 10, gen, max_delta_t=[10])
        self.assertTrue(
            'Cannot perform an ensemble propagate_until/for/grid(): the "max_delta_t" argument must be a scalar, not an iterable object'
            in str(cm.exception)
        )

        # NOTE: check that the chunksize option is not recognised
        # in threaded mode.
        with self.assertRaises(TypeError) as cm:
            ensemble_propagate_until(ta, 20.0, 10, gen, chunksize=1)


def run_test_suite():
    from . import (
        make_nbody_sys,
        taylor_adaptive,
        _test_real,
        _test_real128,
        _test_mp,
        _test_cfunc,
    )
    import numpy as np

    sys = make_nbody_sys(2, masses=[1.1, 2.1], Gconst=1)
    ta = taylor_adaptive(
        sys, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
    )

    retval = 0

    suite = _ut.TestLoader().loadTestsFromTestCase(taylor_add_jet_test_case)
    suite.addTest(_ut.makeSuite(_test_mp.mp_test_case))
    suite.addTest(_ut.makeSuite(_test_real.real_test_case))
    suite.addTest(_ut.makeSuite(_test_real128.real128_test_case))
    suite.addTest(_ut.makeSuite(_test_cfunc.cfunc_test_case))
    suite.addTest(_ut.makeSuite(ensemble_test_case))
    suite.addTest(_ut.makeSuite(s11n_backend_test_case))
    suite.addTest(_ut.makeSuite(recommended_simd_size_test_case))
    suite.addTest(_ut.makeSuite(c_output_test_case))
    suite.addTest(_ut.makeSuite(expression_test_case))
    suite.addTest(_ut.makeSuite(llvm_state_test_case))
    suite.addTest(_ut.makeSuite(event_classes_test_case))
    suite.addTest(_ut.makeSuite(event_detection_test_case))
    suite.addTest(_ut.makeSuite(expression_eval_test_case))
    suite.addTest(_ut.makeSuite(batch_integrator_test_case))
    suite.addTest(_ut.makeSuite(scalar_integrator_test_case))
    suite.addTest(_ut.makeSuite(kepE_test_case))
    suite.addTest(_ut.makeSuite(sympy_test_case))
    suite.addTest(_ut.makeSuite(zero_division_error_test_case))

    test_result = _ut.TextTestRunner(verbosity=2).run(suite)

    if len(test_result.failures) > 0 or len(test_result.errors) > 0:
        retval = 1
    if retval != 0:
        raise RuntimeError("One or more tests failed.")
