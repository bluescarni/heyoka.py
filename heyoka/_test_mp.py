# Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class mp_test_case(_ut.TestCase):
    def runTest(self):
        from . import core

        if not hasattr(core, "real"):
            return

        self.test_basic()
        self.test_c_out()
        self.test_events()

    def test_events(self):
        # Basic event testing.
        from . import make_vars, taylor_adaptive, real, t_event

        x, v = make_vars("x", "v")

        prec = 237

        ta = taylor_adaptive(
            [(x, v), (v, -x)],
            [real(0, prec), real(1, prec)],
            fp_type=real,
            t_events = [t_event(v, fp_type=real)]
        )

        ta.propagate_until(real(100))

        self.assertLess(abs(ta.time - real("1.570796326794896619231321691639751442098584699687552910487472296153908199", prec)), 1e-70)
        self.assertLess(abs(ta.state[0] - 1), 1e-70)
        self.assertLess(abs(ta.state[1]), 1e-70)


    def test_c_out(self):
        from . import make_vars, taylor_adaptive, real, core
        import numpy as np
        from copy import deepcopy

        x, v = make_vars("x", "v")

        prec = 237

        ta = taylor_adaptive(
            [(x, v), (v, -x)],
            [real(0, prec), real(1, prec)],
            fp_type=real,
        )

        res = ta.propagate_until(real(10), c_output=True)
        r5 = res[-1](real("5.3", prec + 10))
        self.assertFalse(r5.flags.writeable)
        self.assertFalse(r5.flags.owndata)
        x_cmp = np.sin(real("5.3", prec))
        v_cmp = np.cos(real("5.3", prec))
        self.assertLess(abs((x_cmp - r5[0]) / x_cmp), 1e-70)
        self.assertLess(abs((v_cmp - r5[1]) / v_cmp), 1e-70)

        # Test for the vector time overload.
        r5 = deepcopy(r5)
        r7 = deepcopy(res[-1](real("7.3", prec + 10)))
        vout = res[-1]([real("5.3", prec + 10), real("7.3", prec + 10)])
        self.assertTrue(np.all(vout[0] == r5))
        self.assertTrue(np.all(vout[1] == r7))

        # Test with non-owning array.
        vout = res[-1](
            core._make_no_real_array([real("5.3", prec + 10), real("7.3", prec + 10)])
        )
        self.assertTrue(np.all(vout[0] == r5))
        self.assertTrue(np.all(vout[1] == r7))

        # Test failure modes.
        arr = np.empty((2,), dtype=real)
        with self.assertRaises(ValueError) as cm:
            res[-1](arr)
        self.assertTrue("A non-constructed/invalid real" in str(cm.exception))
        self.assertTrue("0" in str(cm.exception))

        arr[0] = 1
        with self.assertRaises(ValueError) as cm:
            res[-1](arr)
        self.assertTrue("A non-constructed/invalid real" in str(cm.exception))
        self.assertTrue("1" in str(cm.exception))

    def test_basic(self):
        from . import make_vars, taylor_adaptive, sin, real, core
        import numpy as np

        x, v = make_vars("x", "v")

        prec = 237

        # Small test with automatic inference of precision.
        ta = taylor_adaptive(
            [(x, v), (v, -9.8 * sin(x))],
            [real(-1, prec), real(0, prec)],
            fp_type=real,
        )

        self.assertTrue(ta.compact_mode)

        self.assertEqual(ta.prec, 237)

        def compute_energy(sv):
            return (sv[1] * sv[1]) / 2 + 9.8 * (1 - np.cos(sv[0]))

        orig_E = compute_energy(ta.state)

        ta.propagate_until(real(10))

        self.assertLess(abs((orig_E - compute_energy(ta.state)) / orig_E), 1e-71)

        orig_sv = ta.state

        # Precision explicitly given.
        ta = taylor_adaptive(
            [(x, v), (v, -9.8 * sin(x))],
            [real(-1, prec - 1), real(0, prec + 1)],
            fp_type=real,
            prec=prec,
        )

        self.assertEqual(ta.prec, 237)

        # Compact mode disabled.
        ta2 = taylor_adaptive(
            [(x, v), (v, -9.8 * sin(x))],
            [real(-1, prec - 1), real(0, prec + 1)],
            fp_type=real,
            prec=10,
            compact_mode=False,
        )

        self.assertFalse(ta2.compact_mode)

        ta.propagate_until(real(10))
        self.assertTrue(np.all(orig_sv == ta.state))

        # Testing for the pyreal_check_array() helper.
        arr = np.empty((5,), dtype=real)
        with self.assertRaises(ValueError) as cm:
            core._real_check_array(arr)
        self.assertTrue("A non-constructed/invalid real" in str(cm.exception))
        self.assertTrue("0" in str(cm.exception))

        arr[0] = 5
        with self.assertRaises(ValueError) as cm:
            core._real_check_array(arr)
        self.assertTrue("A non-constructed/invalid real" in str(cm.exception))
        self.assertTrue("1" in str(cm.exception))

        core._real_check_array(core._make_no_real_array(arr))

        arr.fill(real(5))
        core._real_check_array(arr)

        arr = np.empty((5, 5), dtype=real)
        with self.assertRaises(ValueError) as cm:
            core._real_check_array(arr)
        self.assertTrue("A non-constructed/invalid real" in str(cm.exception))
        self.assertTrue("0, 0" in str(cm.exception))

        arr[0, 0] = 5
        with self.assertRaises(ValueError) as cm:
            core._real_check_array(arr)
        self.assertTrue("A non-constructed/invalid real" in str(cm.exception))
        self.assertTrue("0, 1" in str(cm.exception))
        arr[0].fill(real(5))
        arr[1, 0] = 6
        arr[1, 1] = 7
        with self.assertRaises(ValueError) as cm:
            core._real_check_array(arr)
        self.assertTrue("A non-constructed/invalid real" in str(cm.exception))
        self.assertTrue("1, 2" in str(cm.exception))

        arr.fill(real(5))
        core._real_check_array(arr)

        arr = np.empty((), dtype=real)
        core._real_check_array(arr)

        arr = np.empty((1, 0, 3), dtype=real)
        core._real_check_array(arr)

        arr = np.empty((1, 2, 3), dtype=real)

        with self.assertRaises(ValueError) as cm:
            core._real_check_array(arr)
        self.assertTrue(
            "Cannot call pyreal_check_array() on an array with 3 dimensions"
            in str(cm.exception)
        )

        # Testing for the pyreal_ensure_array() helper.
        arr = np.empty((5,), dtype=real)
        core._real_ensure_array(arr, 71)
        for val in arr:
            self.assertEqual(val, 0)
            self.assertEqual(val.prec, 71)

        arr = np.empty((5,), dtype=real)
        arr[0] = real(5, 88)
        core._real_ensure_array(arr, 71)
        self.assertEqual(arr[0], 5)
        self.assertEqual(arr[0].prec, 71)
        for val in arr[1:]:
            self.assertEqual(val, 0)
            self.assertEqual(val.prec, 71)

        arr = core._make_no_real_array(np.empty((5,), dtype=real))
        arr[0] = real(5, 88)
        core._real_ensure_array(arr, 71)
        self.assertEqual(arr[0], 5)
        self.assertEqual(arr[0].prec, 71)
        for val in arr[1:]:
            self.assertEqual(val, 0)
            self.assertEqual(val.prec, 71)

        arr = np.full((5,), real("1.1", 11))
        core._real_ensure_array(arr, 71)
        for val in arr:
            self.assertEqual(val, real(real("1.1", 11), 71))
            self.assertEqual(val.prec, 71)

        arr = np.empty((5, 5), dtype=real)
        core._real_ensure_array(arr, 71)
        for r in arr:
            for val in r:
                self.assertEqual(val, 0)
                self.assertEqual(val.prec, 71)

        arr = np.empty((5, 5), dtype=real)
        arr[0, :] = real(5, 88)
        core._real_ensure_array(arr, 71)
        self.assertTrue(np.all(arr[0, :] == np.full((5,), real(5, 88))))
        for r in arr[1:]:
            for val in r:
                self.assertEqual(val, 0)
                self.assertEqual(val.prec, 71)

        arr = np.full((5, 5), real("1.1", 11))
        core._real_ensure_array(arr, 71)
        for r in arr:
            for val in r:
                self.assertEqual(val, real(real("1.1", 11), 71))
                self.assertEqual(val.prec, 71)

        arr = np.empty((), dtype=real)
        core._real_ensure_array(arr, 71)

        arr = np.empty((1, 0, 3), dtype=real)
        self.assertEqual(arr.size, 0)
