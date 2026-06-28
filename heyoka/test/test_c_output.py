# Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest
from copy import copy, deepcopy
from .. import (
    make_vars,
    sin,
    taylor_adaptive_batch,
    continuous_output_batch_dbl,
    continuous_output_batch_flt,
    taylor_adaptive,
    continuous_output_dbl,
    continuous_output_flt,
    _core,
)
from pickle import dumps, loads
from sys import getrefcount
import numpy as np


class c_output_test_case(unittest.TestCase):
    def test_batch(self):
        x, v = make_vars("x", "v")

        fp_types = [
            (np.float32, continuous_output_batch_flt),
            (float, continuous_output_batch_dbl),
        ]

        # Use a pendulum for testing purposes.
        sys = [(x, v), (v, -9.8 * sin(x))]

        for fp_t, c_out_t in fp_types:
            # Test the default cted object.
            c_out = c_out_t()

            with self.assertRaises(ValueError) as cm:
                c_out(np.zeros((0,), dtype=fp_t))
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

            self.assertFalse(c_out.llvm_state.ir == "")

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

            self.assertFalse(c_out.llvm_state.ir == "")

            c_out = deepcopy(c_out)

            with self.assertRaises(ValueError) as cm:
                c_out.n_steps
            self.assertTrue(
                "Cannot use a default-constructed continuous_output_batch object"
                in str(cm.exception)
            )

            self.assertEqual(c_out.batch_size, 0)

            self.assertFalse("forward" in repr(c_out))

            self.assertFalse(c_out.llvm_state.ir == "")

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

            self.assertFalse(c_out.llvm_state.ir == "")

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
                    tint.time = fp_t(0)

            final_tm = [fp_t(10), fp_t(10.4), fp_t(10.5), fp_t(11.0)]

            check_tm = [fp_t(0.1), fp_t(1.3), fp_t(5.6), fp_t(9.1)]

            c_out, cb = ta.propagate_until(final_tm)

            self.assertTrue(cb is None)
            self.assertTrue(c_out is None)

            reset()

            c_out, cb = ta.propagate_until(final_tm, c_output=False)

            self.assertTrue(cb is None)
            self.assertTrue(c_out is None)

            reset()

            c_out, cb = ta.propagate_until(final_tm, c_output=True)

            self.assertTrue(cb is None)
            self.assertFalse(c_out is None)

            self.assertTrue(c_out(check_tm).shape == (2, 4))

            with self.assertRaises(ValueError) as cm:
                c_out(check_tm)[0] = 0.5

            rc = getrefcount(c_out)
            tmp_out = c_out(check_tm)
            new_rc = getrefcount(c_out)
            self.assertEqual(new_rc, rc + 1)

            with self.assertRaises(ValueError) as cm:
                c_out(np.zeros((1, 1, 1), dtype=fp_t))
            self.assertTrue(
                "Invalid time array passed to a continuous_output_batch object: the"
                " number of dimensions must be 1 or 2, but it is 3 instead"
                in str(cm.exception)
            )

            # Single batch tests.
            with self.assertRaises(ValueError) as cm:
                c_out(np.zeros((1,), dtype=fp_t))
            self.assertTrue(
                "Invalid time array passed to a continuous_output_batch object: the"
                " length must be 4 but it is 1 instead" in str(cm.exception)
            )
            with self.assertRaises(ValueError) as cm:
                c_out(np.zeros((0,), dtype=fp_t))
            self.assertTrue(
                "Invalid time array passed to a continuous_output_batch object: the"
                " length must be 4 but it is 0 instead" in str(cm.exception)
            )
            with self.assertRaises(ValueError) as cm:
                c_out(np.zeros((5,), dtype=fp_t))
            self.assertTrue(
                "Invalid time array passed to a continuous_output_batch object: the"
                " length must be 4 but it is 5 instead" in str(cm.exception)
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
            nc_check_tm = np.vstack([check_tm, np.zeros((4,), dtype=fp_t)]).T.flatten()[
                ::2
            ]
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
                c_out(np.zeros((5, 3), dtype=fp_t))
            self.assertTrue(
                "Invalid time array passed to a continuous_output_batch object: the"
                " number of columns must be 4 but it is 3 instead" in str(cm.exception)
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
            out_b = c_out(np.zeros((0, 4), dtype=fp_t))
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

            self.assertFalse(c_out.llvm_state.ir == "")

            # Try copies as well.
            c_out = copy(c_out)

            self.assertEqual(c_out.tcs.shape, (c_out.n_steps, 2, ta.order + 1, 4))

            self.assertFalse(c_out.llvm_state.ir == "")

            c_out = deepcopy(c_out)

            self.assertEqual(c_out.tcs.shape, (c_out.n_steps, 2, ta.order + 1, 4))

            # Pickling.
            c_out = loads(dumps(c_out))

            self.assertFalse(c_out.llvm_state.ir == "")

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

            self.assertFalse(c_out.llvm_state.ir == "")

            self.assertEqual(c_out.tcs.shape, (c_out.n_steps, 2, ta.order + 1, 4))

            self.assertEqual(c_out.foo, [])

    def test_scalar(self):
        x, v = make_vars("x", "v")

        if _core._ppc_arch:
            fp_types = [
                (np.float32, continuous_output_flt),
                (float, continuous_output_dbl),
            ]
        else:
            from .. import continuous_output_ldbl

            fp_types = [
                (np.float32, continuous_output_flt),
                (float, continuous_output_dbl),
                (np.longdouble, continuous_output_ldbl),
            ]

        if hasattr(_core, "real128"):
            from .. import continuous_output_f128

            fp_types.append((_core.real128, continuous_output_f128))

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

            self.assertTrue(c_out(time=np.zeros((0,), dtype=fp_t)).shape == (0, 0))
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

            self.assertFalse(c_out.llvm_state.ir == "")

            # Try copies as well.
            c_out = copy(c_out)

            with self.assertRaises(ValueError) as cm:
                c_out.n_steps
            self.assertTrue(
                "Cannot use a default-constructed continuous_output object"
                in str(cm.exception)
            )

            self.assertFalse("forward" in repr(c_out))

            self.assertFalse(c_out.llvm_state.ir == "")

            c_out = deepcopy(c_out)

            with self.assertRaises(ValueError) as cm:
                c_out.n_steps
            self.assertTrue(
                "Cannot use a default-constructed continuous_output object"
                in str(cm.exception)
            )

            self.assertFalse("forward" in repr(c_out))

            self.assertFalse(c_out.llvm_state.ir == "")

            # Pickling.
            c_out = loads(dumps(c_out))

            with self.assertRaises(ValueError) as cm:
                c_out.n_steps
            self.assertTrue(
                "Cannot use a default-constructed continuous_output object"
                in str(cm.exception)
            )

            self.assertFalse("forward" in repr(c_out))

            self.assertFalse(c_out.llvm_state.ir == "")

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

            _, _, _, nsteps, c_out, cb = ta.propagate_until(fp_t(10), c_output=True)

            self.assertTrue(cb is None)
            self.assertFalse(c_out is None)

            self.assertTrue(c_out(fp_t(0.1)).shape == (2,))

            with self.assertRaises(ValueError) as cm:
                c_out(fp_t(0.1))[0] = 0.5

            rc = getrefcount(c_out)
            tmp_out = c_out(fp_t(0.1))
            new_rc = getrefcount(c_out)
            self.assertEqual(new_rc, rc + 1)

            self.assertTrue(c_out(np.zeros((0,), dtype=fp_t)).shape == (0, 2))

            tmp = c_out([fp_t(0), fp_t(1), fp_t(2)])
            self.assertTrue(np.all(c_out(fp_t(0)) == tmp[0]))
            self.assertTrue(np.all(c_out(fp_t(1)) == tmp[1]))
            self.assertTrue(np.all(c_out(fp_t(2)) == tmp[2]))

            # Check wrong shape for the input array.
            with self.assertRaises(ValueError) as cm:
                c_out([[fp_t(0)], [fp_t(1)], [fp_t(2)]])
            self.assertTrue(
                "Invalid time array passed to a continuous_output object: the number of"
                " dimensions must be 1, but it is 2 instead" in str(cm.exception)
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

            self.assertFalse(c_out.llvm_state.ir == "")

            # Try copies as well.
            c_out = copy(c_out)

            self.assertEqual(c_out.tcs.shape, (nsteps, 2, ta.order + 1))
            self.assertTrue(np.all(np.isfinite(c_out.tcs)))
            with self.assertRaises(ValueError) as cm:
                c_out.tcs[0, 0, 0] = 0.5

            self.assertFalse(c_out.llvm_state.ir == "")

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

            self.assertFalse(c_out.llvm_state.ir == "")

            self.assertEqual(c_out.tcs.shape, (nsteps, 2, ta.order + 1))
            self.assertTrue(np.all(np.isfinite(c_out.tcs)))
            with self.assertRaises(ValueError) as cm:
                c_out.tcs[0, 0, 0] = 0.5

            # Pickling with dynattrs.
            c_out.foo = []
            c_out = loads(dumps(c_out))

            self.assertFalse(c_out.llvm_state.ir == "")

            self.assertEqual(c_out.tcs.shape, (nsteps, 2, ta.order + 1))
            self.assertTrue(np.all(np.isfinite(c_out.tcs)))
            with self.assertRaises(ValueError) as cm:
                c_out.tcs[0, 0, 0] = 0.5

            self.assertEqual(c_out.foo, [])
