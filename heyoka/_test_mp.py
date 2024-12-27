# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class mp_test_case(_ut.TestCase):
    def test_cfunc(self):
        from . import core

        if not hasattr(core, "real"):
            return

        from . import real, cfunc, make_vars, sin, par, time
        import numpy as np

        x, y = make_vars("x", "y")
        func = [sin(x + y), x - par[0], x + y + par[1] + time]

        with self.assertRaises(ValueError) as cm:
            cfunc(func, vars=[y, x], fp_type=real, batch_size=2)
        self.assertTrue(
            "Batch sizes greater than 1 are not supported for this floating-point type"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            cfunc(func, vars=[y, x], fp_type=real, prec=-1)
        self.assertTrue(
            "An invalid precision value of -1 was passed to make_multi_cfunc()"
            in str(cm.exception)
        )

        prec = 237

        fn = cfunc(func, vars=[y, x], fp_type=real, prec=prec)

        # Initial simple test.
        inputs = np.array([real(1, prec), real(2, prec)])
        pars = np.array([real(3, prec), real(4, prec)])
        out = fn(inputs=inputs, pars=pars, time=real("1.1", prec))
        self.assertEqual(out[0], np.sin(inputs[1] + inputs[0]))
        self.assertEqual(out[1], inputs[1] - pars[0])
        self.assertEqual(out[2], inputs[1] + inputs[0] + pars[1] + real("1.1", prec))

        # With provided empty output.
        out = fn(
            inputs=inputs,
            pars=pars,
            outputs=np.empty((3,), dtype=real),
            time=real("1.1", prec),
        )
        self.assertEqual(out[0], np.sin(inputs[1] + inputs[0]))
        self.assertEqual(out[1], inputs[1] - pars[0])
        self.assertEqual(out[2], inputs[1] + inputs[0] + pars[1] + real("1.1", prec))

        # Test with non-owning arrays.
        out = fn(inputs=inputs[:], pars=pars[:], time=real("1.1", prec))
        self.assertEqual(out[0], np.sin(inputs[1] + inputs[0]))
        self.assertEqual(out[1], inputs[1] - pars[0])
        self.assertEqual(out[2], inputs[1] + inputs[0] + pars[1] + real("1.1", prec))

        # Test with overlapping arrays.
        with self.assertRaises(ValueError) as cm:
            fn(inputs=inputs, pars=inputs, time=real("1.1", prec))
        self.assertTrue(
            "Potential memory overlaps detected when attempting to evaluate a compiled"
            " function: please make sure that all input arrays are distinct"
            in str(cm.exception)
        )

        # Test with non-contiguous arrays.
        inputs = np.array([real(1, prec), 0, real(2, prec), 0])
        pars = np.array([real(3, prec), 0, real(4, prec), 0])
        inputs = inputs[::2]
        pars = pars[::2]
        with self.assertRaises(ValueError) as cm:
            fn(inputs=inputs, pars=pars, time=real("1.1", prec))
        self.assertTrue(
            "Invalid inputs array detected: the array is not C-style contiguous, please"
            " consider using numpy.ascontiguousarray() to turn it into one"
            in str(cm.exception)
        )
        with self.assertRaises(ValueError) as cm:
            fn(inputs=np.ascontiguousarray(inputs), pars=pars, time=real("1.1", prec))
        self.assertTrue(
            "Invalid parameters array detected: the array is not C-style contiguous,"
            " please consider using numpy.ascontiguousarray() to turn it into one"
            in str(cm.exception)
        )

        # Test in single eval mode with wrong precision for the time value.
        inputs = np.array([real(1, prec), real(2, prec)])
        pars = np.array([real(3, prec), real(4, prec)])
        with self.assertRaises(ValueError) as cm:
            fn(inputs=inputs, pars=pars, time=real("1.1", prec - 1))
        self.assertTrue(
            "An invalid time value was passed for the evaluation of a compiled"
            " function in multiprecision mode: the time value has a precision of"
            f" {prec-1}, while the expected precision is {prec} instead"
            in str(cm.exception)
        )

        # Test multieval too.
        inputs = np.array(
            [[real(1, prec), real(-1, prec)], [real(2, prec), real(-2, prec)]]
        )
        pars = np.array(
            [[real(3, prec), real(-3, prec)], [real(4, prec), real(-4, prec)]]
        )
        tm = np.array([real("1.1", prec), real("1.3", prec)])
        out = fn(inputs=inputs, pars=pars, time=tm)
        self.assertEqual(out[0, 0], np.sin(inputs[1, 0] + inputs[0, 0]))
        self.assertEqual(out[1, 0], inputs[1, 0] - pars[0, 0])
        self.assertEqual(out[2, 0], inputs[1, 0] + inputs[0, 0] + pars[1, 0] + tm[0])
        self.assertEqual(out[0, 1], np.sin(inputs[1, 1] + inputs[0, 1]))
        self.assertEqual(out[1, 1], inputs[1, 1] - pars[0, 1])
        self.assertEqual(out[2, 1], inputs[1, 1] + inputs[0, 1] + pars[1, 1] + tm[1])

        # With provided empty output.
        out = fn(
            inputs=inputs, pars=pars, outputs=np.empty((3, 2), dtype=real), time=tm
        )
        self.assertEqual(out[0, 0], np.sin(inputs[1, 0] + inputs[0, 0]))
        self.assertEqual(out[1, 0], inputs[1, 0] - pars[0, 0])
        self.assertEqual(out[2, 0], inputs[1, 0] + inputs[0, 0] + pars[1, 0] + tm[0])
        self.assertEqual(out[0, 1], np.sin(inputs[1, 1] + inputs[0, 1]))
        self.assertEqual(out[1, 1], inputs[1, 1] - pars[0, 1])
        self.assertEqual(out[2, 1], inputs[1, 1] + inputs[0, 1] + pars[1, 1] + tm[1])

        # Non-owning.
        out = fn(inputs=inputs[:], pars=pars[:], time=tm[:])
        self.assertEqual(out[0, 0], np.sin(inputs[1, 0] + inputs[0, 0]))
        self.assertEqual(out[1, 0], inputs[1, 0] - pars[0, 0])
        self.assertEqual(out[2, 0], inputs[1, 0] + inputs[0, 0] + pars[1, 0] + tm[0])
        self.assertEqual(out[0, 1], np.sin(inputs[1, 1] + inputs[0, 1]))
        self.assertEqual(out[1, 1], inputs[1, 1] - pars[0, 1])
        self.assertEqual(out[2, 1], inputs[1, 1] + inputs[0, 1] + pars[1, 1] + tm[1])

        # Non-contiguous.
        inputs = np.array(
            [
                [real(1, prec), 0, real(-1, prec), 0],
                [real(2, prec), 0, real(-2, prec), 0],
            ]
        )
        pars = np.array(
            [
                [real(3, prec), 0, real(-3, prec), 0],
                [real(4, prec), 0, real(-4, prec), 0],
            ]
        )
        tm = np.array([real("1.1", prec), 0, real("1.3", prec), 0])

        inputs = inputs[:, ::2]
        pars = pars[:, ::2]
        tm = tm[::2]
        with self.assertRaises(ValueError) as cm:
            fn(inputs=inputs[:], pars=pars[:], time=tm)
        self.assertTrue(
            "Invalid inputs array detected: the array is not C-style contiguous, please"
            " consider using numpy.ascontiguousarray() to turn it into one"
            in str(cm.exception)
        )
        with self.assertRaises(ValueError) as cm:
            fn(inputs=np.ascontiguousarray(inputs[:]), pars=pars[:], time=tm)
        self.assertTrue(
            "Invalid parameters array detected: the array is not C-style contiguous,"
            " please consider using numpy.ascontiguousarray() to turn it into one"
            in str(cm.exception)
        )

        # Error modes.
        inputs = np.array([real(1, prec), real(2, prec)])
        pars = np.array([real(3, prec), real(4, prec - 1)])
        with self.assertRaises(ValueError) as cm:
            fn(inputs=inputs, pars=pars, time=real(42, prec))
        self.assertTrue(
            "A real with precision 236 was detected at the indices" in str(cm.exception)
        )

        inputs = np.array([real(1, prec), real(2, prec)])
        pars = np.array([real(3, prec), real(4, prec)])
        with self.assertRaises(ValueError) as cm:
            fn(inputs=inputs, pars=pars, time=real(42, prec - 2))
        self.assertTrue(
            "An invalid time value was passed for the evaluation of a compiled"
            " function in multiprecision mode: the time value has a precision of"
            f" {prec - 2}, while the expected precision is {prec} instead"
            in str(cm.exception)
        )

        pars = np.array([real(3, prec), real(4, prec)])
        with self.assertRaises(ValueError) as cm:
            fn(inputs=[1, 2], pars=pars, time=real(42, prec))
        self.assertTrue(
            "in an array which should instead contain elements with a precision of 237"
            in str(cm.exception)
        )

    def test_sympy(self):
        from . import core

        if not hasattr(core, "real"):
            return

        try:
            import sympy
            from mpmath import mp, pi, workprec
        except ImportError:
            return

        from . import to_sympy, from_sympy, real, expression, sum as sum_hy

        with workprec(128):
            from_spy_ex = from_sympy(pi() + sympy.Symbol("x"))
            self.assertEqual(
                sum_hy(
                    [
                        expression(
                            real("3.141592653589793238462643383279502884195", 128)
                        ),
                        expression("x"),
                    ]
                ),
                from_spy_ex,
            )

            self.assertEqual(
                to_sympy(from_sympy(pi() + sympy.Symbol("x"))),
                pi() + sympy.Symbol("x", real=True),
            )

        self.assertEqual(
            str(from_sympy(sympy.Integer(2**500 + 1))),
            "3.2733906078961418700131896968275991522166420460430647894832913680961337964046745548832700923259041571508866841275600710092172565458853930533285275893770e+150",
        )

        self.assertEqual(
            str(
                from_sympy(
                    sympy.Rational(sympy.Integer(2**128 + 1), sympy.Integer(2**500))
                )
            ),
            "1.039540976564489921853040458068190636782e-112",
        )

        self.assertEqual(
            from_sympy(sympy.Integer((2 << 128) + 1)),
            expression(real("680564733841876926926749214863536422913.0", 130)),
        )

    def test_expression(self):
        from . import core

        if not hasattr(core, "real"):
            return

        from . import expression as ex, real, kepE, kepF, kepDE, atan2

        self.assertEqual(
            str(ex(real("1.1", 128))), "1.100000000000000000000000000000000000001"
        )

        self.assertEqual(
            str(1.0 + ex(real("1.1", 128))), "2.099999999999999999999999999999999999995"
        )
        self.assertEqual(
            str(ex(real("1.1", 128)) + 1), "2.099999999999999999999999999999999999995"
        )

        self.assertEqual(
            str(1.0 - ex(real("1.1", 128))),
            "-1.000000000000000000000000000000000000012e-1",
        )
        self.assertEqual(
            str(ex(real("1.1", 128)) - 1),
            "1.000000000000000000000000000000000000012e-1",
        )

        self.assertEqual(
            str(1.0 * ex(real("1.1", 128))),
            "1.100000000000000000000000000000000000001",
        )
        self.assertEqual(
            str(ex(real("1.1", 128)) * 1),
            "1.100000000000000000000000000000000000001",
        )

        self.assertEqual(
            str(1.0 / ex(real("1.1", 128))),
            "9.090909090909090909090909090909090909070e-1",
        )
        self.assertEqual(
            str(ex(real("1.1", 128)) / 1),
            "1.100000000000000000000000000000000000001",
        )

        # NOTE: just test execution for the time being,
        # as these currently do not fold.
        ex(1.1) ** real("1.1", 128)

        kepE(ex("x"), real("1.1", 128))
        kepE(real("1.1", 128), ex("x"))

        kepF(ex("x"), real("1.1", 128), ex("y"))
        kepF(real("1.1", 128), ex("y"), ex("x"))

        kepDE(ex("x"), real("1.1", 128), ex("y"))
        kepDE(real("1.1", 128), ex("y"), ex("x"))

        atan2(ex("x"), real("1.1", 128))
        atan2(real("1.1", 128), ex("x"))

    def test_events(self):
        from . import core

        if not hasattr(core, "real"):
            return

        # Basic event testing.
        from . import make_vars, taylor_adaptive, real, t_event

        x, v = make_vars("x", "v")

        prec = 237

        ta = taylor_adaptive(
            [(x, v), (v, -x)],
            [real(0, prec), real(1, prec)],
            fp_type=real,
            t_events=[t_event(v, fp_type=real)],
        )

        ta.propagate_until(real(100))

        self.assertLess(
            abs(
                ta.time
                - real(
                    "1.570796326794896619231321691639751442098584699687552910487472296153908199",
                    prec,
                )
            ),
            1e-70,
        )
        self.assertLess(abs(ta.state[0] - 1), 1e-70)
        self.assertLess(abs(ta.state[1]), 1e-70)

    def test_c_out(self):
        from . import core

        if not hasattr(core, "real"):
            return

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
        self.assertTrue(res[-1] is None)
        r5 = res[-2](real("5.3", prec + 10))
        self.assertFalse(r5.flags.writeable)
        self.assertFalse(r5.flags.owndata)
        x_cmp = np.sin(real("5.3", prec))
        v_cmp = np.cos(real("5.3", prec))
        self.assertLess(abs((x_cmp - r5[0]) / x_cmp), 1e-70)
        self.assertLess(abs((v_cmp - r5[1]) / v_cmp), 1e-70)

        # Test for the vector time overload.
        r5 = deepcopy(r5)
        r7 = deepcopy(res[-2](real("7.3", prec + 10)))
        vout = res[-2]([real("5.3", prec + 10), real("7.3", prec + 10)])
        self.assertTrue(np.all(vout[0] == r5))
        self.assertTrue(np.all(vout[1] == r7))

        # Test with non-owning array.
        vout = res[-2](
            core._make_no_real_array([real("5.3", prec + 10), real("7.3", prec + 10)])
        )
        self.assertTrue(np.all(vout[0] == r5))
        self.assertTrue(np.all(vout[1] == r7))

        # Test failure modes.
        arr = np.empty((2,), dtype=real)
        with self.assertRaises(ValueError) as cm:
            res[-2](arr)
        self.assertTrue("A non-constructed/invalid real" in str(cm.exception))
        self.assertTrue("0" in str(cm.exception))

        arr[0] = 1
        with self.assertRaises(ValueError) as cm:
            res[-2](arr)
        self.assertTrue("A non-constructed/invalid real" in str(cm.exception))
        self.assertTrue("1" in str(cm.exception))

    def test_basic(self):
        from . import core

        if not hasattr(core, "real"):
            return

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

        self.assertEqual(ta.prec, prec)

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

        self.assertEqual(ta.prec, prec)

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

        # Test with bogus precision.
        with self.assertRaises(ValueError) as cm:
            taylor_adaptive(
                [(x, v), (v, -9.8 * sin(x))],
                [real(-1, prec - 1), real(0, prec + 1)],
                fp_type=real,
                prec=-1,
                compact_mode=False,
            )
        self.assertTrue(
            "Cannot set the precision of a real to the value -1" in str(cm.exception)
        )

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

        # Empty init state with explicit precision.
        ta = taylor_adaptive(
            [(x, v), (v, -9.8 * sin(x))],
            fp_type=real,
            prec=prec,
        )

        self.assertEqual(ta.prec, prec)
        self.assertTrue(np.all(ta.state == [0, 0]))

        ta = taylor_adaptive(
            [(x, v), (v, -9.8 * sin(x))],
            [],
            fp_type=real,
            prec=prec,
        )

        self.assertEqual(ta.prec, prec)
        self.assertTrue(np.all(ta.state == [0, 0]))

        # Empty init state without explicit precision.
        with self.assertRaises(ValueError) as cm:
            ta = taylor_adaptive(
                [(x, v), (v, -9.8 * sin(x))],
                fp_type=real,
            )
        self.assertTrue("we cannot deduce the desired precision" in str(cm.exception))
