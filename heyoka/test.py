# Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class taylor_add_jet_test_case(_ut.TestCase):
    def runTest(self):
        from . import taylor_add_jet, make_vars, sin, taylor_adaptive, par, time, taylor_adaptive_batch, with_real128, tpoly
        from .core import _ppc_arch
        import numpy as np

        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]
        sys_par = [(x, v), (v, -par[0] * sin(x))]
        sys_par_t = [(x, v), (v, -par[0] * sin(x) + time)]
        sys_par_t2 = [(x, v), (v, -par[0] * sin(x) + tpoly(par[1], par[6]))]

        if _ppc_arch:
            fp_types = [("double", float)]
        else:
            fp_types = [("double", float), ("long double", np.longdouble)]

        for desc, fp_t in fp_types:
            # Check that the jet is consistent
            # with the Taylor coefficients.
            init_state = [fp_t(0.05), fp_t(0.025)]
            pars = [fp_t(-9.8)]
            pars2 = [fp_t(-9.8), fp_t(.01), fp_t(.02), fp_t(.03),
                     fp_t(.04), fp_t(.05), fp_t(.06)]

            ta = taylor_adaptive(sys, init_state, tol=fp_t(1e-9), fp_type=desc)

            jet = taylor_add_jet(sys, 5, fp_type=desc)
            st = np.full((6, 2), fp_t(0), dtype=fp_t)
            st[0] = init_state

            ta.step(write_tc=True)
            jet(st)

            self.assertTrue(np.all(ta.tc[:, :6].transpose() == st))

            # Try adding an sv_func.
            jet = taylor_add_jet(sys, 5, fp_type=desc, sv_funcs=[x + v])
            st = np.full((6, 3), fp_t(0), dtype=fp_t)
            st[0, :2] = init_state

            jet(st)

            self.assertTrue(np.all(ta.tc[:, :6].transpose() == st[:, :2]))
            self.assertTrue(
                np.all((ta.tc[0, :6] + ta.tc[1, :6]).transpose() == st[:, 2]))

            # An example with params.
            ta_par = taylor_adaptive(
                sys_par, init_state, tol=fp_t(1e-9), fp_type=desc, pars=pars)

            jet_par = taylor_add_jet(sys_par, 5, fp_type=desc)
            st = np.full((6, 2), fp_t(0), dtype=fp_t)
            st[0] = init_state
            par_arr = np.full((1,), fp_t(-9.8), dtype=fp_t)

            ta_par.step(write_tc=True)
            jet_par(st, pars=par_arr)

            self.assertTrue(np.all(ta_par.tc[:, :6].transpose() == st))

            # Params + time.
            ta_par_t = taylor_adaptive(
                sys_par_t, init_state, tol=fp_t(1e-9), fp_type=desc, pars=pars)
            ta_par_t.time = fp_t(0.01)

            jet_par_t = taylor_add_jet(sys_par_t, 5, fp_type=desc)
            st = np.full((6, 2), fp_t(0), dtype=fp_t)
            st[0] = init_state
            par_arr = np.full((1,), fp_t(-9.8), dtype=fp_t)
            time_arr = np.full((1,), fp_t(0.01), dtype=fp_t)

            ta_par_t.step(write_tc=True)
            jet_par_t(st, pars=par_arr, time=time_arr)

            self.assertTrue(np.all(ta_par_t.tc[:, :6].transpose() == st))

            ta_par_t2 = taylor_adaptive(
                sys_par_t2, init_state, tol=fp_t(1e-9), fp_type=desc, pars=pars2)
            ta_par_t2.time = fp_t(0.01)

            jet_par_t2 = taylor_add_jet(sys_par_t2, 5, fp_type=desc)
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
                "Invalid state vector passed to a function for the computation of the jet of Taylor derivatives: the NumPy array is not C contiguous" in str(cm.exception))

            # Non-writeable state.
            st.flags.writeable = False
            with self.assertRaises(ValueError) as cm:
                jet(st)
            self.assertTrue(
                "Invalid state vector passed to a function for the computation of the jet of Taylor derivatives: the NumPy array is not writeable" in str(cm.exception))

            # Non-contiguous pars.
            st = np.full((6, 2), fp_t(0), dtype=fp_t)
            par_arr = np.full((5,), fp_t(-9.8), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=par_arr[::2])
            self.assertTrue(
                "Invalid parameters vector passed to a function for the computation of the jet of Taylor derivatives: the NumPy array is not C contiguous" in str(cm.exception))

            # Non-contiguous time.
            time_arr = np.full((5,), fp_t(0.01), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=par_arr, time=time_arr[::2])
            self.assertTrue(
                "Invalid time vector passed to a function for the computation of the jet of Taylor derivatives: the NumPy array is not C contiguous" in str(cm.exception))

            # Overlapping arrays.
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=st, time=time_arr)
            self.assertTrue(
                "Invalid vectors passed to a function for the computation of the jet of Taylor derivatives: the NumPy arrays must all be distinct" in str(cm.exception))
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=par_arr, time=st)
            self.assertTrue(
                "Invalid vectors passed to a function for the computation of the jet of Taylor derivatives: the NumPy arrays must all be distinct" in str(cm.exception))
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=st, time=st)
            self.assertTrue(
                "Invalid vectors passed to a function for the computation of the jet of Taylor derivatives: the NumPy arrays must all be distinct" in str(cm.exception))

            # Params needed but not provided.
            with self.assertRaises(ValueError) as cm:
                jet_par(st)
            self.assertTrue("Invalid vectors passed to a function for the computation of the jet of "
                            "Taylor derivatives: the ODE system contains parameters, but no parameter array was "
                            "passed as input argument" in str(cm.exception))

            # Time needed but not provided.
            with self.assertRaises(ValueError) as cm:
                jet_par_t(st, pars=par_arr)
            self.assertTrue("Invalid vectors passed to a function for the computation of the jet of "
                            "Taylor derivatives: the ODE system is non-autonomous, but no time array was "
                            "passed as input argument" in str(cm.exception))
            with self.assertRaises(ValueError) as cm:
                jet_par_t2(st, pars=par_arr2)
            self.assertTrue("Invalid vectors passed to a function for the computation of the jet of "
                            "Taylor derivatives: the ODE system is non-autonomous, but no time array was "
                            "passed as input argument" in str(cm.exception))

            # Wrong st shape, scalar case.
            st = np.full((6, 2, 1), fp_t(0), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet(st)
            self.assertTrue("Invalid state vector passed to a function for the computation of the jet of "
                            "Taylor derivatives: the number of dimensions must be 2, but it is "
                            "3 instead" in str(cm.exception))

            st = np.full((6, 4), fp_t(0), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet(st)
            self.assertTrue("Invalid state vector passed to a function for the computation of the jet of "
                            "Taylor derivatives: the shape must be (6, 3), but it is "
                            "(6, 4) instead" in str(cm.exception))

            # Wrong param shape, scalar case.
            st = np.full((6, 2), fp_t(0), dtype=fp_t)
            par_arr = np.full((5, 2), fp_t(-9.8), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=par_arr)
            self.assertTrue("Invalid parameters vector passed to a function for the computation of "
                            "the jet of "
                            "Taylor derivatives: the number of dimensions must be 1, but it is "
                            "2 instead" in str(cm.exception))

            par_arr = np.full((5,), fp_t(-9.8), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=par_arr)
            self.assertTrue("Invalid parameters vector passed to a function for the "
                            "computation of the jet of "
                            "Taylor derivatives: the shape must be (1), but it is "
                            "(5) instead" in str(cm.exception))

            # Wrong time shape, scalar case.
            par_arr = np.full((1,), fp_t(-9.8), dtype=fp_t)
            time_arr = np.full((2, 1), fp_t(0.01), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par_t(st, pars=par_arr, time=time_arr)
            self.assertTrue("Invalid time vector passed to a function for the computation of the jet of "
                            "Taylor derivatives: the number of dimensions must be 1, but it is "
                            "2 instead" in str(cm.exception))

            time_arr = np.full((5,), fp_t(0.01), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par_t(st, pars=par_arr, time=time_arr)
            self.assertTrue("Invalid time vector passed to a function for the computation of the jet of "
                            "Taylor derivatives: the shape must be (1), but it is "
                            "(5) instead" in str(cm.exception))

        # Check throwing behaviour with long double on PPC.
        if _ppc_arch:
            with self.assertRaises(NotImplementedError):
                taylor_add_jet(sys, 5, fp_type="long double")

        # Batch mode testing.
        fp_types = [("double", float)]

        for desc, fp_t in fp_types:
            batch_size = 4

            # Check that the jet is consistent
            # with the Taylor coefficients.
            init_state = [[fp_t(0.05), fp_t(0.06), fp_t(0.07), fp_t(0.08)],
                          [fp_t(0.025), fp_t(0.026), fp_t(0.027), fp_t(0.028)]]
            pars = [[fp_t(-9.8), fp_t(-9.7), fp_t(-9.6), fp_t(-9.5)]]

            ta = taylor_adaptive_batch(
                sys, init_state, tol=fp_t(1e-9), fp_type=desc)

            jet = taylor_add_jet(sys, 5, fp_type=desc, batch_size=batch_size)
            st = np.full((6, 2, batch_size), fp_t(0), dtype=fp_t)
            st[0] = init_state

            ta.step(write_tc=True)
            jet(st)

            self.assertTrue(np.all(ta.tc[:, :6, :].transpose((1, 0, 2)) == st))

            # Try adding an sv_func.
            jet = taylor_add_jet(sys, 5, fp_type=desc, sv_funcs=[
                                 x + v], batch_size=batch_size)
            st = np.full((6, 3, batch_size), fp_t(0), dtype=fp_t)
            st[0, :2] = init_state

            jet(st)

            self.assertTrue(
                np.all(ta.tc[:, :6, :].transpose((1, 0, 2)) == st[:, :2]))
            self.assertTrue(
                np.all((ta.tc[0, :6, :] + ta.tc[1, :6, :]) == st[:, 2, :]))

            # An example with params.
            ta_par = taylor_adaptive_batch(
                sys_par, init_state, tol=fp_t(1e-9), fp_type=desc, pars=pars)

            jet_par = taylor_add_jet(
                sys_par, 5, fp_type=desc, batch_size=batch_size)
            st = np.full((6, 2, batch_size), fp_t(0), dtype=fp_t)
            st[0] = init_state
            par_arr = np.array(pars)

            ta_par.step(write_tc=True)
            jet_par(st, pars=par_arr)

            self.assertTrue(
                np.all(ta_par.tc[:, :6, :].transpose((1, 0, 2)) == st))

            # Params + time.
            ta_par_t = taylor_adaptive_batch(
                sys_par_t, init_state, tol=fp_t(1e-9), fp_type=desc, pars=pars)
            ta_par_t.set_time([fp_t(0.01), fp_t(0.02), fp_t(0.03), fp_t(0.04)])

            jet_par_t = taylor_add_jet(
                sys_par_t, 5, fp_type=desc, batch_size=batch_size)
            st = np.full((6, 2, batch_size), fp_t(0), dtype=fp_t)
            st[0] = init_state
            par_arr = np.array(pars)
            time_arr = np.array(
                [fp_t(0.01), fp_t(0.02), fp_t(0.03), fp_t(0.04)])

            ta_par_t.step(write_tc=True)
            jet_par_t(st, pars=par_arr, time=time_arr)

            self.assertTrue(
                np.all(ta_par_t.tc[:, :6, :].transpose((1, 0, 2)) == st))

            # Just do shape/dims checks for the batch case.

            # Wrong st shape, batch case.
            st = np.full((6, 2, 1, 1), fp_t(0), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet(st)
            self.assertTrue("Invalid state vector passed to a function for the computation of the jet of "
                            "Taylor derivatives: the number of dimensions must be 3, but it is "
                            "4 instead" in str(cm.exception))

            st = np.full((6, 4, 4), fp_t(0), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet(st)
            self.assertTrue("Invalid state vector passed to a function for the computation of the jet of "
                            "Taylor derivatives: the shape must be (6, 3, 4), but it is "
                            "(6, 4, 4) instead" in str(cm.exception))

            # Wrong param shape, batch case.
            st = np.full((6, 2, 4), fp_t(0), dtype=fp_t)
            par_arr = np.full((5, 2, 1), fp_t(-9.8), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=par_arr)
            self.assertTrue("Invalid parameters vector passed to a function for the computation of "
                            "the jet of "
                            "Taylor derivatives: the number of dimensions must be 2, but it is "
                            "3 instead" in str(cm.exception))

            par_arr = np.full((5, 1), fp_t(-9.8), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=par_arr)
            self.assertTrue("Invalid parameters vector passed to a function for the "
                            "computation of the jet of "
                            "Taylor derivatives: the shape must be (1, 4), but it is "
                            "(5, 1) instead" in str(cm.exception))

            # Wrong time shape, batch case.
            par_arr = np.full((1, 4), fp_t(-9.8), dtype=fp_t)
            time_arr = np.full((2, 1), fp_t(0.01), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par_t(st, pars=par_arr, time=time_arr)
            self.assertTrue("Invalid time vector passed to a function for the computation of the jet of "
                            "Taylor derivatives: the number of dimensions must be 1, but it is "
                            "2 instead" in str(cm.exception))

            time_arr = np.full((5,), fp_t(0.01), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par_t(st, pars=par_arr, time=time_arr)
            self.assertTrue("Invalid time vector passed to a function for the computation of the jet of "
                            "Taylor derivatives: the shape must be (4), but it is "
                            "(5) instead" in str(cm.exception))

        if not with_real128:
            return

        from mpmath import mpf

        # Check that the jet is consistent
        # with the Taylor coefficients.
        init_state = [mpf(0.05), mpf(0.025)]
        pars = [mpf(-9.8)]

        ta = taylor_adaptive(sys, init_state, tol=mpf(1e-9), fp_type="real128")

        jet = taylor_add_jet(sys, 5, fp_type="real128")
        st = np.full((6, 2), mpf(0))
        st[0] = init_state

        ta.step(write_tc=True)
        st = jet(st)

        self.assertTrue(np.all(ta.tc[:, :6].transpose() == st))

        # Try adding an sv_func.
        jet = taylor_add_jet(sys, 5, fp_type="real128", sv_funcs=[x + v])
        st = np.full((6, 3), mpf(0))
        st[0, :2] = init_state

        st = jet(st)

        self.assertTrue(np.all(ta.tc[:, :6].transpose() == st[:, :2]))
        self.assertTrue(
            np.all((ta.tc[0, :6] + ta.tc[1, :6]).transpose() == st[:, 2]))

        # An example with params.
        ta_par = taylor_adaptive(sys_par, init_state, tol=mpf(
            1e-9), fp_type="real128", pars=pars)

        jet_par = taylor_add_jet(sys_par, 5, fp_type="real128")
        st = np.full((6, 2), mpf(0))
        st[0] = init_state
        par_arr = np.full((1,), mpf(-9.8))

        ta_par.step(write_tc=True)
        st = jet_par(st, pars=par_arr)

        self.assertTrue(np.all(ta_par.tc[:, :6].transpose() == st))

        # Params + time.
        ta_par_t = taylor_adaptive(sys_par_t, init_state, tol=mpf(
            1e-9), fp_type="real128", pars=pars)
        ta_par_t.time = mpf(0.01)

        jet_par_t = taylor_add_jet(sys_par_t, 5, fp_type="real128")
        st = np.full((6, 2), mpf(0))
        st[0] = init_state
        par_arr = np.full((1,), mpf(-9.8))
        time_arr = np.full((1,), mpf(0.01))

        ta_par_t.step(write_tc=True)
        st = jet_par_t(st, pars=par_arr, time=time_arr)

        self.assertTrue(np.all(ta_par_t.tc[:, :6].transpose() == st))

        # Failure modes.

        # Params needed but not provided.
        with self.assertRaises(ValueError) as cm:
            jet_par(st)
        self.assertTrue("Invalid vectors passed to a function for the computation of the jet of "
                        "Taylor derivatives: the ODE system contains parameters, but no parameter array was "
                        "passed as input argument" in str(cm.exception))

        # Time needed but not provided.
        with self.assertRaises(ValueError) as cm:
            jet_par_t(st, pars=par_arr)
        self.assertTrue("Invalid vectors passed to a function for the computation of the jet of "
                        "Taylor derivatives: the ODE system is non-autonomous, but no time array was "
                        "passed as input argument" in str(cm.exception))

        # Wrong st shape, scalar case.
        st = np.full((6, 2, 1), mpf(0))
        with self.assertRaises(ValueError) as cm:
            jet(st)
        self.assertTrue("Invalid state vector passed to a function for the computation of the jet of "
                        "Taylor derivatives: the number of dimensions must be 2, but it is "
                        "3 instead" in str(cm.exception))

        st = np.full((6, 4), mpf(0))
        with self.assertRaises(ValueError) as cm:
            jet(st)
        self.assertTrue("Invalid state vector passed to a function for the computation of the jet of "
                        "Taylor derivatives: the shape must be (6, 3), but it is "
                        "(6, 4) instead" in str(cm.exception))

        # Wrong param shape, scalar case.
        st = np.full((6, 2), mpf(0))
        par_arr = np.full((5, 2), mpf(-9.8))
        with self.assertRaises(ValueError) as cm:
            jet_par(st, pars=par_arr)
        self.assertTrue("Invalid parameters vector passed to a function for the computation of "
                        "the jet of "
                        "Taylor derivatives: the number of dimensions must be 1, but it is "
                        "2 instead" in str(cm.exception))

        par_arr = np.full((5,), mpf(-9.8))
        with self.assertRaises(ValueError) as cm:
            jet_par(st, pars=par_arr)
        self.assertTrue("Invalid parameters vector passed to a function for the "
                        "computation of the jet of "
                        "Taylor derivatives: the shape must be (1), but it is "
                        "(5) instead" in str(cm.exception))

        # Wrong time shape, scalar case.
        par_arr = np.full((1,), mpf(-9.8))
        time_arr = np.full((2, 1), mpf(0.01))
        with self.assertRaises(ValueError) as cm:
            jet_par_t(st, pars=par_arr, time=time_arr)
        self.assertTrue("Invalid time vector passed to a function for the computation of the jet of "
                        "Taylor derivatives: the number of dimensions must be 1, but it is "
                        "2 instead" in str(cm.exception))

        time_arr = np.full((5,), mpf(0.01))
        with self.assertRaises(ValueError) as cm:
            jet_par_t(st, pars=par_arr, time=time_arr)
        self.assertTrue("Invalid time vector passed to a function for the computation of the jet of "
                        "Taylor derivatives: the shape must be (1), but it is "
                        "(5) instead" in str(cm.exception))


class event_classes_test_case(_ut.TestCase):
    def runTest(self):
        from . import t_event, nt_event, make_vars, event_direction, with_real128
        from .core import _ppc_arch
        import numpy as np
        import pickle
        import gc

        x, v = make_vars("x", "v")

        if _ppc_arch:
            fp_types = [("double", float)]
        else:
            fp_types = [("double", float), ("long double", np.longdouble)]

        if with_real128:
            from mpmath import mpf
            fp_types.append(("real128", mpf))

        for desc, fp_t in fp_types:
            # Non-terminal event.
            ev = nt_event(x + v, lambda _: _, fp_type=desc)

            self.assertTrue(" non-terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::any" in repr(ev))
            self.assertEqual(ev.expression, x + v)
            self.assertEqual(ev.direction, event_direction.any)
            self.assertFalse(ev.callback is None)

            ev = nt_event(ex=x + v, callback=lambda _: _, fp_type=desc)
            self.assertTrue(" non-terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::any" in repr(ev))
            self.assertEqual(ev.expression, x + v)
            self.assertEqual(ev.direction, event_direction.any)
            self.assertFalse(ev.callback is None)

            ev = nt_event(ex=x + v, callback=lambda _: _,
                          direction=event_direction.positive, fp_type=desc)
            self.assertTrue(" non-terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::positive" in repr(ev))
            self.assertEqual(ev.expression, x + v)
            self.assertEqual(ev.direction, event_direction.positive)
            self.assertFalse(ev.callback is None)

            ev = nt_event(ex=x + v, callback=lambda _: _,
                          direction=event_direction.negative, fp_type=desc)
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
            ev = nt_event(ex=x + v, callback=lcb,
                          direction=event_direction.negative, fp_type=desc)
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
                nt_event(ex=x + v, callback=lambda _: _,
                         direction=event_direction(10), fp_type=desc)
            self.assertTrue(
                "Invalid value selected for the direction of a non-terminal event" in str(cm.exception))

            with self.assertRaises(TypeError) as cm:
                nt_event(ex=x + v, callback=3, fp_type=desc)
            self.assertTrue(
                "An object of type '{}' cannot be used as an event callback because it is not callable".format(str(type(3))) in str(cm.exception))

            with self.assertRaises(TypeError) as cm:
                nt_event(ex=x + v, callback=None, fp_type=desc)
            self.assertTrue(
                "An object of type '{}' cannot be used as an event callback because it is not callable".format(str(type(None))) in str(cm.exception))

            ev = nt_event(ex=x + v, callback=lambda _: _,
                          direction=event_direction.negative, fp_type=desc)
            ev = pickle.loads(pickle.dumps(ev))
            self.assertTrue(" non-terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::negative" in repr(ev))

            # Test dynamic attributes.
            ev.foo = "hello world"
            ev = pickle.loads(pickle.dumps(ev))
            self.assertEqual(ev.foo, "hello world")

            # Test to ensure a callback extracted from the event
            # is kept alive and usable when the event is destroyed.
            ev = nt_event(ex=x + v, callback=local_cb(),
                          direction=event_direction.negative, fp_type=desc)
            out_cb = ev.callback
            del(ev)
            gc.collect()
            out_cb(1, 2, 3)
            out_cb(1, 2, 3)
            out_cb(1, 2, 3)
            self.assertEqual(out_cb.n, 3)

            # Terminal event.
            ev = t_event(x + v, fp_type=desc)

            self.assertTrue(" terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::any" in repr(ev))
            self.assertTrue(": no" in repr(ev))
            self.assertTrue("auto" in repr(ev))
            self.assertEqual(ev.expression, x + v)
            self.assertEqual(ev.direction, event_direction.any)
            self.assertEqual(ev.cooldown, fp_t(-1))
            self.assertTrue(ev.callback is None)

            ev = t_event(x + v, fp_type=desc,
                         direction=event_direction.negative, cooldown=fp_t(3))

            self.assertTrue(" terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::negative" in repr(ev))
            self.assertTrue(": no" in repr(ev))
            self.assertTrue("3" in repr(ev))
            self.assertEqual(ev.expression, x + v)
            self.assertEqual(ev.direction, event_direction.negative)
            self.assertEqual(ev.cooldown, fp_t(3))
            self.assertTrue(ev.callback is None)

            ev = t_event(x + v, fp_type=desc, direction=event_direction.positive,
                         cooldown=fp_t(3), callback=lambda _: _)

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
            ev = t_event(x + v, fp_type=desc, direction=event_direction.positive,
                         cooldown=fp_t(3), callback=lcb)

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

            ev = t_event(x + v, fp_type=desc, direction=event_direction.positive,
                         cooldown=fp_t(3), callback=None)
            self.assertTrue(ev.callback is None)

            with self.assertRaises(ValueError) as cm:
                t_event(x + v, fp_type=desc, direction=event_direction(45),
                        cooldown=fp_t(3), callback=lambda _: _)
            self.assertTrue(
                "Invalid value selected for the direction of a terminal event" in str(cm.exception))

            with self.assertRaises(TypeError) as cm:
                t_event(x + v, callback=3, fp_type=desc)
            self.assertTrue(
                "An object of type '{}' cannot be used as an event callback because it is not callable".format(str(type(3))) in str(cm.exception))

            ev = t_event(x + v, fp_type=desc, direction=event_direction.positive,
                         cooldown=fp_t(3), callback=lambda _: _)

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

            # Test also with empty callback.
            ev = t_event(x + v, fp_type=desc, direction=event_direction.positive,
                         cooldown=fp_t(3))

            ev = pickle.loads(pickle.dumps(ev))
            self.assertTrue(" terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::positive" in repr(ev))
            self.assertTrue(": no" in repr(ev))
            self.assertTrue("3" in repr(ev))

            # Test to ensure a callback extracted from the event
            # is kept alive and usable when the event is destroyed.
            ev = t_event(ex=x + v, callback=local_cb(),
                         direction=event_direction.negative, fp_type=desc)
            out_cb = ev.callback
            del(ev)
            gc.collect()
            out_cb(1, 2, 3)
            out_cb(1, 2, 3)
            out_cb(1, 2, 3)
            self.assertEqual(out_cb.n, 3)


class event_detection_test_case(_ut.TestCase):
    def runTest(self):
        from . import t_event, nt_event, make_vars, event_direction, with_real128, sin, taylor_adaptive, taylor_outcome
        from .core import _ppc_arch
        from sys import getrefcount
        import numpy as np
        from copy import deepcopy

        x, v = make_vars("x", "v")

        if _ppc_arch:
            fp_types = [("double", float)]
        else:
            fp_types = [("double", float), ("long double", np.longdouble)]

        if with_real128:
            from mpmath import mpf
            fp_types.append(("real128", mpf))

        # Use a pendulum for testing purposes.
        sys = [(x, v), (v, -9.8 * sin(x))]

        for desc, fp_t in fp_types:
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

            ta = taylor_adaptive(sys=sys, state=[fp_t(0), fp_t(0.25)], fp_type=desc,
                                 nt_events=[nt_event(v*v-1e-10, cb0, fp_type=desc),
                                            nt_event(v, cb1, fp_type=desc)])

            ta_id = id(ta)

            self.assertEqual(ta.propagate_until(fp_t(4))[
                             0], taylor_outcome.time_limit)

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

            ta = taylor_adaptive(sys=sys, state=[fp_t(0), fp_t(0.25)], fp_type=desc,
                                 nt_events=[nt_event(v*v-1e-10, cb0(), fp_type=desc),
                                            nt_event(v, cb1(), fp_type=desc),
                                            nt_event(v, cb1(), fp_type=desc)])

            # Check that the refcount increases by 3
            # (the number of events).
            rc = getrefcount(ta)
            nt_list = ta.nt_events
            new_rc = getrefcount(ta)
            self.assertEqual(new_rc, rc + 3)

            self.assertEqual(id(ta.nt_events[0].callback), id(
                ta.nt_events[0].callback))
            self.assertEqual(id(ta.nt_events[0].callback.lst), id(
                ta.nt_events[0].callback.lst))
            self.assertEqual(id(ta.nt_events[1].callback), id(
                ta.nt_events[1].callback))
            self.assertEqual(id(ta.nt_events[1].callback.lst), id(
                ta.nt_events[1].callback.lst))
            self.assertEqual(id(ta.nt_events[2].callback), id(
                ta.nt_events[2].callback))
            self.assertEqual(id(ta.nt_events[2].callback.lst), id(
                ta.nt_events[2].callback.lst))

            # Ensure a deep copy of the integrator performs
            # a deep copy of the events.
            ta_copy = deepcopy(ta)

            self.assertNotEqual(id(ta_copy.nt_events[0].callback), id(
                ta.nt_events[0].callback))
            self.assertNotEqual(id(ta_copy.nt_events[0].callback.lst), id(
                ta.nt_events[0].callback.lst))
            self.assertNotEqual(id(ta_copy.nt_events[1].callback), id(
                ta.nt_events[1].callback))
            self.assertNotEqual(id(ta_copy.nt_events[1].callback.lst), id(
                ta.nt_events[1].callback.lst))
            self.assertNotEqual(id(ta_copy.nt_events[2].callback), id(
                ta.nt_events[2].callback))
            self.assertNotEqual(id(ta_copy.nt_events[2].callback.lst), id(
                ta.nt_events[2].callback.lst))

            # Callback with wrong signature.
            def cb2(ta, t):
                pass

            ta = taylor_adaptive(sys=sys, state=[fp_t(0), fp_t(0.25)], fp_type=desc,
                                 nt_events=[nt_event(v*v-1e-10, cb2, fp_type=desc)])

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

            ta = taylor_adaptive(sys=sys, state=[fp_t(0), fp_t(0.25)], fp_type=desc,
                                 nt_events=[
                                     nt_event(v*v-1e-10, cb0, fp_type=desc)],
                                 t_events=[t_event(v, callback=cb1, fp_type=desc)])

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

            ta = taylor_adaptive(sys=sys, state=[fp_t(0), fp_t(0.25)], fp_type=desc,
                                 t_events=[t_event(v*v-1e-10, callback=cb0(), fp_type=desc),
                                           t_event(v, callback=cb1(),
                                                   fp_type=desc),
                                           t_event(v, callback=cb1(), fp_type=desc)])

            # Check that the refcount increases by 3
            # (the number of events).
            rc = getrefcount(ta)
            t_list = ta.t_events
            new_rc = getrefcount(ta)
            self.assertEqual(new_rc, rc + 3)

            self.assertEqual(id(ta.t_events[0].callback), id(
                ta.t_events[0].callback))
            self.assertEqual(id(ta.t_events[0].callback.lst), id(
                ta.t_events[0].callback.lst))
            self.assertEqual(id(ta.t_events[1].callback), id(
                ta.t_events[1].callback))
            self.assertEqual(id(ta.t_events[1].callback.lst), id(
                ta.t_events[1].callback.lst))
            self.assertEqual(id(ta.t_events[2].callback), id(
                ta.t_events[2].callback))
            self.assertEqual(id(ta.t_events[2].callback.lst), id(
                ta.t_events[2].callback.lst))

            # Ensure a deep copy of the integrator performs
            # a deep copy of the events.
            ta_copy = deepcopy(ta)

            self.assertNotEqual(id(ta_copy.t_events[0].callback), id(
                ta.t_events[0].callback))
            self.assertNotEqual(id(ta_copy.t_events[0].callback.lst), id(
                ta.t_events[0].callback.lst))
            self.assertNotEqual(id(ta_copy.t_events[1].callback), id(
                ta.t_events[1].callback))
            self.assertNotEqual(id(ta_copy.t_events[1].callback.lst), id(
                ta.t_events[1].callback.lst))
            self.assertNotEqual(id(ta_copy.t_events[2].callback), id(
                ta.t_events[2].callback))
            self.assertNotEqual(id(ta_copy.t_events[2].callback.lst), id(
                ta.t_events[2].callback.lst))

            # Callback with wrong signature.
            def cb2(ta, t):
                pass

            ta = taylor_adaptive(sys=sys, state=[fp_t(0), fp_t(0.25)], fp_type=desc,
                                 t_events=[t_event(v*v-1e-10, callback=cb2, fp_type=desc)])

            with self.assertRaises(TypeError):
                ta.propagate_until(fp_t(4))

            # Callback with wrong retval.
            def cb3(ta, mr, d_sgn):
                return "hello"

            ta = taylor_adaptive(sys=sys, state=[fp_t(0), fp_t(0.25)], fp_type=desc,
                                 t_events=[t_event(v*v-1e-10, callback=cb3, fp_type=desc)])

            with self.assertRaises(TypeError) as cm:
                ta.propagate_until(fp_t(4))
            self.assertTrue(
                "in the construction of the return value of an event callback" in str(cm.exception))


class expression_eval_test_case(_ut.TestCase):
    def runTest(self):
        from . import sin, make_vars, with_real128, eval
        from .core import _ppc_arch
        import numpy as np
        from math import log10

        x, = make_vars("x")

        if _ppc_arch:
            fp_types = [
                ("double", float, int(-log10(np.finfo(float).eps)) - 1)]
        else:
            fp_types = [("double", float, int(-log10(np.finfo(float).eps)) - 1),
                        ("long double", np.longdouble, int(-log10(np.finfo(np.longdouble).eps)) - 1)]

        if with_real128:
            from mpmath import mpf
            fp_types.append(("real128", mpf, 32))

        for desc, fp_t, places in fp_types:
            target = fp_t("0.123456789012345678901234567890")
            a = eval(x, {"x": target}, fp_type=desc)
            self.assertEqual(a, target)
            a = eval(x**3.1, {"x": target}, fp_type=desc)
            self.assertAlmostEqual(a, target**3.1, places=places)


class scalar_integrator_test_case(_ut.TestCase):
    def runTest(self):
        self.test_s11n()

    def test_s11n(self):
        from . import nt_event, make_vars, with_real128, sin, taylor_adaptive
        from .core import _ppc_arch
        import numpy as np
        import pickle

        x, v = make_vars("x", "v")

        if _ppc_arch:
            fp_types = [("double", float)]
        else:
            fp_types = [("double", float), ("long double", np.longdouble)]

        if with_real128:
            from mpmath import mpf
            fp_types.append(("real128", mpf))

        # Use a pendulum for testing purposes.
        sys = [(x, v), (v, -9.8 * sin(x))]

        def cb0(ta, t, d_sgn):
            pass

        for desc, fp_t in fp_types:
            ta = taylor_adaptive(sys=sys, state=[fp_t(0), fp_t(0.25)], fp_type=desc,
                                 nt_events=[nt_event(v*v-1e-10, cb0, fp_type=desc)])

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

            if desc != "real128":
                self.assertTrue(np.all(ta.state == ta2.state))
                self.assertTrue(np.all(ta.time == ta2.time))
            else:
                self.assertTrue(np.all(ta.get_state() == ta2.get_state()))
                self.assertTrue(np.all(ta.time == ta2.time))

            ta.step()
            ta2.step()

            if desc != "real128":
                self.assertTrue(np.all(ta.state == ta2.state))
                self.assertTrue(np.all(ta.time == ta2.time))
            else:
                self.assertTrue(np.all(ta.get_state() == ta2.get_state()))
                self.assertTrue(np.all(ta.time == ta2.time))

            # Try also an integrator with stateful event callback.
            class cb1:
                def __init__(self):
                    self.n = 0

                def __call__(self, ta, t, d_sgn):
                    self.n = self.n + 1

            clb = cb1()
            ta = taylor_adaptive(sys=sys, state=[fp_t(0), fp_t(0.25)], fp_type=desc,
                                 nt_events=[nt_event(v*v-1e-10, clb, fp_type=desc)])

            self.assertNotEqual(id(clb), id(ta.nt_events[0].callback))

            self.assertEqual(ta.nt_events[0].callback.n, 0)

            ta.propagate_until(fp_t(10))

            ta2 = pickle.loads(pickle.dumps(ta))

            self.assertEqual(
                ta.nt_events[0].callback.n, ta2.nt_events[0].callback.n)

            # Test dynamic attributes.
            ta.foo = "hello world"
            ta = pickle.loads(pickle.dumps(ta))
            self.assertEqual(ta.foo, "hello world")

            ta = taylor_adaptive(sys=sys, state=[fp_t(0), fp_t(0.25)], fp_type=desc,
                                 tol=fp_t(1e-6))

            self.assertEqual(ta.tol, fp_t(1e-6))

        # Check throwing behaviour with long double on PPC.
        if _ppc_arch:
            fp_t = np.longdouble

            with self.assertRaises(NotImplementedError):
                taylor_adaptive(sys=sys, state=[fp_t(
                    0), fp_t(0.25)], fp_type='long double')


class batch_integrator_test_case(_ut.TestCase):
    def runTest(self):
        self.run_propagate_grid_tests()
        self.test_s11n()

    def test_s11n(self):
        from . import make_vars, sin, taylor_adaptive_batch
        import numpy as np
        import pickle

        x, v = make_vars("x", "v")

        # Use a pendulum for testing purposes.
        sys = [(x, v), (v, -9.8 * sin(x))]

        ta = taylor_adaptive_batch(sys=sys, state=[[0, .01], [0.25, 0.26]])

        ta.step()
        ta.step()
        ta.step()
        ta.step()

        ta2 = pickle.loads(pickle.dumps(ta))

        self.assertTrue(np.all(ta.state == ta2.state))
        self.assertTrue(np.all(ta.time == ta2.time))

        ta.step()
        ta2.step()

        self.assertTrue(np.all(ta.state == ta2.state))
        self.assertTrue(np.all(ta.time == ta2.time))

        ta = taylor_adaptive_batch(
            sys=sys, state=[[0, .01], [0.25, 0.26]], tol=1e-6)

        self.assertEqual(ta.tol, 1e-6)

        # Test dynamic attributes.
        ta.foo = "hello world"
        ta = pickle.loads(pickle.dumps(ta))
        self.assertEqual(ta.foo, "hello world")

    def run_propagate_grid_tests(self):
        from . import make_vars, taylor_adaptive, taylor_adaptive_batch, sin
        import numpy as np

        x, v = make_vars("x", "v")
        eqns = [(x, v),
                (v, -9.8*sin(x))]

        x_ic = [0.06, 0.07, 0.08,  0.09]
        v_ic = [0.025, 0.026, 0.027, 0.028]

        ta = taylor_adaptive_batch(eqns, [x_ic, v_ic])

        # Failure modes.
        with self.assertRaises(ValueError) as cm:
            ta.propagate_grid([])
        self.assertTrue(
            "Invalid grid passed to the propagate_grid() method of a batch integrator: "
            "the expected number of dimensions is 2, but the input array has a dimension of 1" in str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            ta.propagate_grid([[1, 2], [3, 4]])
        self.assertTrue(
            "Invalid grid passed to the propagate_grid() method of a batch integrator: "
            "the shape must be (n, 4) but the number of columns is 2 instead" in str(cm.exception))

        # Run a simple scalar/batch comparison.
        tas = []

        for x0, v0 in zip(x_ic, v_ic):
            tas.append(taylor_adaptive(eqns, [x0, v0]))

        grid = np.array([[-0.1, -0.2, -0.3, -0.4],
                         [0.01, 0.02, 0.03, 0.9],
                         [1., 1.1, 1.2, 1.3],
                         [11., 11.1, 11.2, 11.3]])

        bres = ta.propagate_grid(grid)

        sres = [tas[0].propagate_grid(grid[:, 0]),
                tas[1].propagate_grid(grid[:, 1]),
                tas[2].propagate_grid(grid[:, 2]),
                tas[3].propagate_grid(grid[:, 3])]

        self.assertTrue(
            np.max(np.abs(sres[0][4] - bres[:, :, 0]).flatten()) < 1e-14)
        self.assertTrue(
            np.max(np.abs(sres[1][4] - bres[:, :, 1]).flatten()) < 1e-14)
        self.assertTrue(
            np.max(np.abs(sres[2][4] - bres[:, :, 2]).flatten()) < 1e-14)
        self.assertTrue(
            np.max(np.abs(sres[3][4] - bres[:, :, 3]).flatten()) < 1e-14)


class kepE_test_case(_ut.TestCase):
    def runTest(self):
        from . import kepE, with_real128, diff, make_vars, sin, cos
        from .core import _ppc_arch
        import numpy as np

        x, y = make_vars("x", "y")
        self.assertEqual(diff(kepE(x, y), x), sin(
            kepE(x, y)) / (1. - x * cos(kepE(x, y))))
        self.assertEqual(diff(kepE(x, y), y), 1. / (1. - x * cos(kepE(x, y))))

        if not _ppc_arch:
            self.assertEqual(diff(kepE(x, np.longdouble("1.1")), x), sin(
                kepE(x, np.longdouble("1.1"))) / (1. - x * cos(kepE(x, np.longdouble("1.1")))))
            self.assertEqual(diff(kepE(np.longdouble("1.1"), y), y), 1. /
                             (1. - np.longdouble("1.1") * cos(kepE(np.longdouble("1.1"), y))))

        if not with_real128:
            return

        from mpmath import mpf

        self.assertEqual(diff(kepE(x, mpf("1.1")), x), sin(
            kepE(x, mpf("1.1"))) / (1. - x * cos(kepE(x, mpf("1.1")))))
        self.assertEqual(diff(kepE(mpf("1.1"), y), y), 1. /
                         (1. - mpf("1.1") * cos(kepE(mpf("1.1"), y))))


class sympy_test_case(_ut.TestCase):
    def runTest(self):
        try:
            import sympy
        except ImportError:
            return

        self.test_number_conversion()
        self.test_sympar_conversion()
        self.test_func_conversion()

        from . import from_sympy, make_vars, sum as hsum

        with self.assertRaises(TypeError) as cm:
            from_sympy(3.5)
        self.assertTrue(
            "The 'ex' parameter must be a sympy expression but it is of type" in str(cm.exception))

        with self.assertRaises(TypeError) as cm:
            from_sympy(sympy.Symbol("x"), [])
        self.assertTrue(
            "The 's_dict' parameter must be a dict but it is of type" in str(cm.exception))

        with self.assertRaises(TypeError) as cm:
            from_sympy(sympy.Symbol("x"), {3.5: 3.5})
        self.assertTrue(
            "The keys in 's_dict' must all be sympy expressions" in str(cm.exception))

        with self.assertRaises(TypeError) as cm:
            from_sympy(sympy.Symbol("x"), {sympy.Symbol("x"): 3.5})
        self.assertTrue(
            "The values in 's_dict' must all be heyoka expressions" in str(cm.exception))

        # Test the s_dict functionality of from_sympy().
        x, y = sympy.symbols("x y", real=True)
        hx, hy, hz = make_vars("x", "y", "z")

        self.assertEqual(from_sympy(
            (x-y)*(x+y), s_dict={x-y: hz}), hsum([hx, hy])*hz)

    def test_number_conversion(self):
        from . import to_sympy, from_sympy, expression, with_real128
        from .core import _ppc_arch
        from sympy import Float, Rational, Integer
        from mpmath import workprec, mpf
        import numpy as np

        with self.assertRaises(ValueError) as cm:
            from_sympy(Rational(3, 5))
        self.assertTrue(
            "Cannot convert from sympy a rational number whose denominator is not a power of 2" in str(cm.exception))

        # From integer.
        self.assertEqual(from_sympy(Integer(-42)), expression(-42.))

        # From rational.
        self.assertEqual(from_sympy(Rational(42, -2)), expression(-21.))

        # Double precision.
        with workprec(53):
            self.assertEqual(to_sympy(expression(1.1)), Float(1.1))
            self.assertEqual(from_sympy(Float(1.1)), expression(1.1))

            self.assertEqual(
                to_sympy(expression((2**40+1)/(2**128))), Rational(2**40+1, 2**128))
            self.assertEqual(from_sympy(Rational(2**40+1, 2**128)),
                             expression((2**40+1)/(2**128)))

        # Long double precision.
        if not _ppc_arch:
            with workprec(np.finfo(np.longdouble).nmant + 1):
                self.assertEqual(to_sympy(expression(np.longdouble("1.1"))), Float(
                    "1.1", precision=np.finfo(np.longdouble).nmant + 1))
                self.assertEqual(from_sympy(Float("1.1")),
                                 expression(np.longdouble("1.1")))

                expo = np.finfo(np.longdouble).nmant - 10
                self.assertEqual(to_sympy(expression(np.longdouble(
                    2**expo+1)/np.longdouble(2**128))), Rational(2**expo+1, 2**128))
                self.assertEqual(from_sympy(Rational(2**expo+1, 2**128)),
                                 expression(np.longdouble(2**expo+1)/np.longdouble(2**128)))

        # Too high precision.
        with self.assertRaises(ValueError) as cm:
            from_sympy(Integer(2**500 + 1))
        self.assertTrue("the required precision" in str(cm.exception))

        if not with_real128 or _ppc_arch:
            return

        # Quad precision.
        with workprec(113):
            self.assertEqual(to_sympy(expression(mpf("1.1"))),
                             Float("1.1", precision=113))
            self.assertEqual(from_sympy(Float("1.1")), expression(mpf("1.1")))

            expo = 100
            self.assertEqual(
                to_sympy(expression(mpf(2**expo+1)/mpf(2**128))), Rational(2**expo+1, 2**128))
            self.assertEqual(from_sympy(Rational(2**expo+1, 2**128)),
                             expression(mpf(2**expo+1)/mpf(2**128)))

    def test_sympar_conversion(self):
        from . import to_sympy, from_sympy, expression, par
        from sympy import Symbol

        self.assertEqual(Symbol("x", real=True), to_sympy(expression("x")))
        self.assertEqual(Symbol("par[0]", real=True), to_sympy(par[0]))
        self.assertEqual(Symbol("par[9]", real=True), to_sympy(par[9]))
        self.assertEqual(Symbol("par[123]", real=True), to_sympy(par[123]))
        self.assertEqual(Symbol("par[-123]", real=True),
                         to_sympy(expression("par[-123]")))
        self.assertEqual(Symbol("par[]", real=True),
                         to_sympy(expression("par[]")))

        self.assertEqual(from_sympy(Symbol("x")), expression("x"))
        self.assertEqual(from_sympy(Symbol("par[0]")), par[0])
        self.assertEqual(from_sympy(Symbol("par[9]")), par[9])
        self.assertEqual(from_sympy(Symbol("par[123]")), par[123])
        self.assertEqual(from_sympy(
            Symbol("par[-123]")), expression("par[-123]"))
        self.assertEqual(from_sympy(Symbol("par[]")), expression("par[]"))

    def test_func_conversion(self):
        import sympy as spy
        from . import core, make_vars, from_sympy, to_sympy, pi, sum as hsum

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

        self.assertEqual(hsum([hx, hy, hz]), from_sympy(x+y+z))
        self.assertEqual(to_sympy(hx+hy+hz), x+y+z)
        self.assertEqual(hsum([ha, hb, hc, hx, hy, hz]),
                         from_sympy(x+y+z+a+b+c))
        self.assertEqual(to_sympy(ha+hb+hc+hx+hy+hz), x+y+z+a+b+c)

        self.assertEqual(hx*hy*hz, from_sympy(x*y*z))
        self.assertEqual(to_sympy(hx*hy*hz), x*y*z)
        self.assertEqual((ha*hb)*(hc*hx)*(hy*hz), from_sympy(x*y*z*a*b*c))
        self.assertEqual(to_sympy(ha*hb*hc*hx*hy*hz), x*y*z*a*b*c)

        self.assertEqual(hsum([hx, -1.*hy, -1.*hz]), from_sympy(x-y-z))
        self.assertEqual(to_sympy(hx - hy - hz), x-y-z)

        self.assertEqual(hx * hz**-1., from_sympy(x/z))
        self.assertEqual(to_sympy(hx / hz), x / z)

        self.assertEqual(core.kepE(hx, hy), from_sympy(
            spy.Function("heyoka_kepE")(x, y)))
        self.assertEqual(to_sympy(core.kepE(hx, hy)),
                         spy.Function("heyoka_kepE")(x, y))

        self.assertEqual(-1. * hx, from_sympy(-x))
        self.assertEqual(to_sympy(-hx), -x)

        self.assertEqual(to_sympy(core.sigmoid(hx + hy)),
                         1. / (1. + spy.exp(-x - y)))

        self.assertEqual(core.square(hx + hy), (hx + hy) * (hx + hy))

        self.assertEqual(core.time, from_sympy(spy.Function("heyoka_time")()))
        self.assertEqual(to_sympy(core.time), spy.Function("heyoka_time")())

        self.assertEqual(core.tpoly(core.par[0], core.par[10]), from_sympy(
            spy.Function("heyoka_tpoly")(spy.Symbol("par[0]"), spy.Symbol("par[10]"))))
        self.assertEqual(to_sympy(core.tpoly(core.par[0], core.par[10])), spy.Function(
            "heyoka_tpoly")(spy.Symbol("par[0]", real=True), spy.Symbol("par[10]", real=True)))

        with self.assertRaises(TypeError) as cm:
            from_sympy(abs(x))
        self.assertTrue(
            "Unable to convert the sympy object" in str(cm.exception))

        # Test caching behaviour.
        foo = hx + hy
        bar = foo / (foo * hz + 1)
        bar_spy = to_sympy(bar)
        self.assertEqual(id(bar_spy.args[1]), id(
            bar_spy.args[0].args[0].args[1].args[1]))

        # pi constant.
        self.assertEqual(to_sympy(pi), spy.pi)
        self.assertEqual(from_sympy(spy.pi), pi)
        self.assertEqual(to_sympy(from_sympy(spy.pi)), spy.pi)


class zero_division_error_test_case(_ut.TestCase):
    def runTest(self):
        from . import make_vars

        x, = make_vars("x")

        with self.assertRaises(ZeroDivisionError) as cm:
            x / 0.
        self.assertTrue(
            "Division by zero" in str(cm.exception))


class expression_test_case(_ut.TestCase):
    def runTest(self):
        self.test_s11n()
        self.test_len()
        self.test_diff()

    def test_diff(self):
        from . import make_vars, sin, cos, diff, par

        x, y = make_vars("x", "y")
        self.assertEqual(diff(cos(x*x-y), "x"), -sin(x*x-y) * (2.*x))
        self.assertEqual(diff(cos(x*x-y), x), -sin(x*x-y) * (2.*x))
        self.assertEqual(
            diff(cos(par[0]*par[0]-y), par[0]), -sin(par[0]*par[0]-y) * (2.*par[0]))

    def test_s11n(self):
        from . import make_vars, expression, with_real128, sin, cos
        from .core import _ppc_arch
        from numpy import longdouble
        import pickle

        x, y = make_vars("x", "y")

        ex = x + 2.*y
        self.assertEqual(ex, pickle.loads(pickle.dumps(ex)))

        # Test dynamic attributes.
        ex.foo = "hello world"
        ex = pickle.loads(pickle.dumps(ex))
        self.assertEqual(ex.foo, "hello world")

        if not _ppc_arch:
            ex = sin(longdouble('1.1')*x) + 2.*y
            self.assertEqual(ex, pickle.loads(pickle.dumps(ex)))

        if not with_real128:
            return

        from mpmath import mpf

        # Quad precision.
        if not _ppc_arch:
            ex = sin(longdouble('1.1')*x) + mpf('1.3')*cos(2.*y)
            self.assertEqual(ex, pickle.loads(pickle.dumps(ex)))

    def test_len(self):
        from . import make_vars

        x, y, z = make_vars("x", "y", "z")

        self.assertEqual(len(x), 1)
        self.assertEqual(len((x - y - z) + (y * z)), 9)


class llvm_state_test_case(_ut.TestCase):
    def runTest(self):
        self.test_s11n()

    def test_s11n(self):
        from . import make_vars, sin, taylor_adaptive
        import pickle
        from sys import getrefcount

        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]

        ta = taylor_adaptive(sys=sys, state=[0., 0.25])

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


def run_test_suite():
    from . import make_nbody_sys, taylor_adaptive, with_real128

    if with_real128:
        from mpmath import mp
        orig_mpmath_prec = mp.prec
        mp.prec = 113

    sys = make_nbody_sys(2, masses=[1.1, 2.1], Gconst=1)
    ta = taylor_adaptive(sys, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    retval = 0

    suite = _ut.TestLoader().loadTestsFromTestCase(taylor_add_jet_test_case)
    suite.addTest(expression_test_case())
    suite.addTest(llvm_state_test_case())
    suite.addTest(expression_test_case())
    suite.addTest(event_classes_test_case())
    suite.addTest(event_detection_test_case())
    suite.addTest(expression_eval_test_case())
    suite.addTest(batch_integrator_test_case())
    suite.addTest(scalar_integrator_test_case())
    suite.addTest(kepE_test_case())
    suite.addTest(sympy_test_case())
    suite.addTest(zero_division_error_test_case())

    test_result = _ut.TextTestRunner(verbosity=2).run(suite)

    if with_real128:
        mp.prec = orig_mpmath_prec

    if len(test_result.failures) > 0 or len(test_result.errors) > 0:
        retval = 1
    if retval != 0:
        raise RuntimeError('One or more tests failed.')
