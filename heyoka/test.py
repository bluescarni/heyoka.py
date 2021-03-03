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
        from . import taylor_add_jet, make_vars, sin, taylor_adaptive, par, time, taylor_adaptive_batch
        from .core import with_real128
        import numpy as np

        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]
        sys_par = [(x, v), (v, -par[0] * sin(x))]
        sys_par_t = [(x, v), (v, -par[0] * sin(x) + time)]

        fp_types = [("double", float), ("long double", np.longdouble)]

        for desc, fp_t in fp_types:
            # Check that the jet is consistent
            # with the Taylor coefficients.
            init_state =  [fp_t(0.05), fp_t(0.025)]
            pars = [fp_t(-9.8)]

            ta = taylor_adaptive(sys, init_state, tol=fp_t(1e-9), fp_type=desc)

            jet = taylor_add_jet(sys, 5, fp_type=desc)
            st = np.full((6, 2), fp_t(0), dtype=fp_t)
            st[0] = init_state

            ta.step(write_tc = True)
            jet(st)

            self.assertTrue(np.all(ta.tc[:,:6].transpose() == st))

            # Try adding an sv_func.
            jet = taylor_add_jet(sys, 5, fp_type=desc, sv_funcs=[x + v])
            st = np.full((6, 3), fp_t(0), dtype=fp_t)
            st[0,:2] = init_state

            jet(st)

            self.assertTrue(np.all(ta.tc[:,:6].transpose() == st[:,:2]))
            self.assertTrue(np.all((ta.tc[0,:6] + ta.tc[1,:6]).transpose() == st[:,2]))

            # An example with params.
            ta_par = taylor_adaptive(sys_par, init_state, tol=fp_t(1e-9), fp_type=desc, pars=pars)

            jet_par = taylor_add_jet(sys_par, 5, fp_type=desc)
            st = np.full((6, 2), fp_t(0), dtype=fp_t)
            st[0] = init_state
            par_arr = np.full((1,), fp_t(-9.8), dtype=fp_t)

            ta_par.step(write_tc = True)
            jet_par(st, pars=par_arr)

            self.assertTrue(np.all(ta_par.tc[:,:6].transpose() == st))

            # Params + time.
            ta_par_t = taylor_adaptive(sys_par_t, init_state, tol=fp_t(1e-9), fp_type=desc, pars=pars)
            ta_par_t.time = fp_t(0.01)

            jet_par_t = taylor_add_jet(sys_par_t, 5, fp_type=desc)
            st = np.full((6, 2), fp_t(0), dtype=fp_t)
            st[0] = init_state
            par_arr = np.full((1,), fp_t(-9.8), dtype=fp_t)
            time_arr = np.full((1,), fp_t(0.01), dtype=fp_t)

            ta_par_t.step(write_tc = True)
            jet_par_t(st, pars=par_arr, time=time_arr)

            self.assertTrue(np.all(ta_par_t.tc[:,:6].transpose() == st))

            # Failure modes.

            # Non-contiguous state.
            with self.assertRaises(ValueError) as cm:
                jet(st[::2])
            self.assertTrue("Invalid state vector passed to a function for the computation of the jet of Taylor derivatives: the NumPy array is not C contiguous" in str(cm.exception))

            # Non-writeable state.
            st.flags.writeable=False
            with self.assertRaises(ValueError) as cm:
                jet(st)
            self.assertTrue("Invalid state vector passed to a function for the computation of the jet of Taylor derivatives: the NumPy array is not writeable" in str(cm.exception))

            # Non-contiguous pars.
            st = np.full((6, 2), fp_t(0), dtype=fp_t)
            par_arr = np.full((5,), fp_t(-9.8), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=par_arr[::2])
            self.assertTrue("Invalid parameters vector passed to a function for the computation of the jet of Taylor derivatives: the NumPy array is not C contiguous" in str(cm.exception))

            # Non-contiguous time.
            time_arr = np.full((5,), fp_t(0.01), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=par_arr, time=time_arr[::2])
            self.assertTrue("Invalid time vector passed to a function for the computation of the jet of Taylor derivatives: the NumPy array is not C contiguous" in str(cm.exception))

            # Overlapping arrays.
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=st, time=time_arr)
            self.assertTrue("Invalid vectors passed to a function for the computation of the jet of Taylor derivatives: the NumPy arrays must all be distinct" in str(cm.exception))
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=par_arr, time=st)
            self.assertTrue("Invalid vectors passed to a function for the computation of the jet of Taylor derivatives: the NumPy arrays must all be distinct" in str(cm.exception))
            with self.assertRaises(ValueError) as cm:
                jet_par(st, pars=st, time=st)
            self.assertTrue("Invalid vectors passed to a function for the computation of the jet of Taylor derivatives: the NumPy arrays must all be distinct" in str(cm.exception))

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
            time_arr = np.full((2,1), fp_t(0.01), dtype=fp_t)
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

        # Batch mode testing.
        fp_types = [("double", float)]

        for desc, fp_t in fp_types:
            batch_size = 4

            # Check that the jet is consistent
            # with the Taylor coefficients.
            init_state =  [[fp_t(0.05), fp_t(0.06), fp_t(0.07), fp_t(0.08)],
                           [fp_t(0.025), fp_t(0.026), fp_t(0.027), fp_t(0.028)]]
            pars = [[fp_t(-9.8), fp_t(-9.7), fp_t(-9.6), fp_t(-9.5)]]

            ta = taylor_adaptive_batch(sys, init_state, tol=fp_t(1e-9), fp_type=desc)

            jet = taylor_add_jet(sys, 5, fp_type=desc, batch_size=batch_size)
            st = np.full((6, 2, batch_size), fp_t(0), dtype=fp_t)
            st[0] = init_state

            ta.step(write_tc = True)
            jet(st)

            self.assertTrue(np.all(ta.tc[:,:6,:].transpose((1,0,2)) == st))

            # Try adding an sv_func.
            jet = taylor_add_jet(sys, 5, fp_type=desc, sv_funcs=[x + v], batch_size=batch_size)
            st = np.full((6, 3, batch_size), fp_t(0), dtype=fp_t)
            st[0,:2] = init_state

            jet(st)

            self.assertTrue(np.all(ta.tc[:,:6,:].transpose((1,0,2)) == st[:,:2]))
            self.assertTrue(np.all((ta.tc[0,:6,:] + ta.tc[1,:6,:]) == st[:,2,:]))

            # An example with params.
            ta_par = taylor_adaptive_batch(sys_par, init_state, tol=fp_t(1e-9), fp_type=desc, pars=pars)

            jet_par = taylor_add_jet(sys_par, 5, fp_type=desc, batch_size=batch_size)
            st = np.full((6, 2, batch_size), fp_t(0), dtype=fp_t)
            st[0] = init_state
            par_arr = np.array(pars)

            ta_par.step(write_tc = True)
            jet_par(st, pars=par_arr)

            self.assertTrue(np.all(ta_par.tc[:,:6,:].transpose((1,0,2)) == st))

            # Params + time.
            ta_par_t = taylor_adaptive_batch(sys_par_t, init_state, tol=fp_t(1e-9), fp_type=desc, pars=pars)
            ta_par_t.time[:] = [fp_t(0.01), fp_t(0.02), fp_t(0.03), fp_t(0.04)]

            jet_par_t = taylor_add_jet(sys_par_t, 5, fp_type=desc, batch_size=batch_size)
            st = np.full((6, 2, batch_size), fp_t(0), dtype=fp_t)
            st[0] = init_state
            par_arr = np.array(pars)
            time_arr = np.array([fp_t(0.01), fp_t(0.02), fp_t(0.03), fp_t(0.04)])

            ta_par_t.step(write_tc = True)
            jet_par_t(st, pars=par_arr, time=time_arr)

            self.assertTrue(np.all(ta_par_t.tc[:,:6,:].transpose((1,0,2)) == st))

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


def run_test_suite():
    from . import make_nbody_sys, taylor_adaptive

    sys = make_nbody_sys(2, masses=[1.1,2.1], Gconst=1)
    ta = taylor_adaptive(sys, [1,2,3,4,5,6,7,8,9,10,11,12])

    retval = 0

    suite = _ut.TestLoader().loadTestsFromTestCase(taylor_add_jet_test_case)

    test_result = _ut.TextTestRunner(verbosity=2).run(suite)

    if len(test_result.failures) > 0 or len(test_result.errors) > 0:
        retval = 1
    if retval != 0:
        raise RuntimeError('One or more tests failed.')
