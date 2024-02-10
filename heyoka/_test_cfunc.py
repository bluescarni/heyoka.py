# Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


import unittest as _ut


class cfunc_test_case(_ut.TestCase):
    def test_basic(self):
        from . import make_cfunc, make_vars, cfunc_dbl, core, par, time
        import pickle
        from copy import copy, deepcopy

        self.assertRaises(ValueError, lambda: make_cfunc([], []))

        x, y, z, s = make_vars("x", "y", "z", "s")
        cf = make_cfunc([y * (x + z)], [x, y, z])

        self.assertFalse(cf.llvm_state_scalar.force_avx512)
        self.assertFalse(cf.llvm_state_scalar.slp_vectorize)

        self.assertEqual(cf([1, 2, 3]), copy(cf)([1, 2, 3]))
        self.assertEqual(cf([1, 2, 3]), deepcopy(cf)([1, 2, 3]))
        self.assertEqual(cf([1, 2, 3]), pickle.loads(pickle.dumps(cf))([1, 2, 3]))

        self.assertEqual(cf.vars, [x, y, z])
        self.assertEqual(cf.fn, [y * (x + z)])
        self.assertEqual(len(cf.dc), 6)
        self.assertNotEqual(len(cf.llvm_state_scalar.get_ir()), 0)
        self.assertEqual(deepcopy(cf).vars, [x, y, z])
        self.assertEqual(deepcopy(cf).fn, [y * (x + z)])
        self.assertEqual(deepcopy(cf).dc, cf.dc)
        self.assertEqual(
            deepcopy(cf).llvm_state_scalar.get_ir(), cf.llvm_state_scalar.get_ir()
        )
        self.assertEqual(
            deepcopy(cf).llvm_state_scalar_s.get_ir(), cf.llvm_state_scalar_s.get_ir()
        )
        self.assertEqual(
            deepcopy(cf).llvm_state_batch_s.get_ir(), cf.llvm_state_batch_s.get_ir()
        )
        self.assertEqual(pickle.loads(pickle.dumps(cf)).vars, [x, y, z])
        self.assertEqual(pickle.loads(pickle.dumps(cf)).fn, [y * (x + z)])
        self.assertEqual(pickle.loads(pickle.dumps(cf)).dc, cf.dc)
        self.assertEqual(
            pickle.loads(pickle.dumps(cf)).llvm_state_scalar.get_ir(),
            cf.llvm_state_scalar.get_ir(),
        )

        cf = make_cfunc(
            [y * (x + z)], vars=[y, z, x], force_avx512=True, slp_vectorize=True
        )
        self.assertEqual(cf.vars, [y, z, x])

        self.assertTrue(cf.llvm_state_scalar.force_avx512)
        self.assertTrue(cf.llvm_state_scalar.slp_vectorize)

        # Tests for correct detection of number of params, time dependency
        # and list of variables.
        cf = make_cfunc([y * (x + z), x], vars=[y, z, x])
        self.assertEqual(cf.nparams, 0)
        cf = make_cfunc([y * (x + z), par[0]], vars=[y, z, x])
        self.assertEqual(cf.nparams, 1)
        cf = make_cfunc([y * (x + z) - par[89], par[0]], vars=[y, z, x])
        self.assertEqual(cf.nparams, 90)

        cf = make_cfunc([y * (x + z), x], vars=[y, z, x])
        self.assertFalse(cf.is_time_dependent)
        cf = make_cfunc([y * (x + z) + time, x], vars=[y, z, x])
        self.assertTrue(cf.is_time_dependent)
        cf = make_cfunc([y * (x + z), x + time], vars=[y, z, x])
        self.assertTrue(cf.is_time_dependent)

        cf = make_cfunc([y * (x + z), x + time], [x, y, z])
        self.assertEqual(cf.vars, [x, y, z])
        cf = make_cfunc([y * (x + z), x + time], vars=[y, z, x])
        self.assertEqual(cf.vars, [y, z, x])
        cf = make_cfunc([y * (x + z), x + time], vars=[y, z, x, s])
        self.assertEqual(cf.vars, [y, z, x, s])
        cf = make_cfunc([y * (x + z), x + time], vars=[s, y, z, x])
        self.assertEqual(cf.vars, [s, y, z, x])

        # NOTE: test for a bug in the multiprecision
        # implementation where the precision is not
        # correctly copied.
        if not hasattr(core, "real"):
            return

        real = core.real

        cf = make_cfunc([y * (x + z)], [x, y, z], fp_type=real, prec=128)
        self.assertEqual(
            cf([real(1, 128), real(2, 128), real(3, 128)]),
            copy(cf)([real(1, 128), real(2, 128), real(3, 128)]),
        )

        self.assertEqual(cf.prec, 128)
        self.assertEqual(deepcopy(cf).prec, 128)
        self.assertEqual(pickle.loads(pickle.dumps(cf)).prec, 128)

    def test_multi(self):
        import numpy as np
        from . import make_cfunc, make_vars, sin, par, expression, core, time
        from .core import _ppc_arch
        from .test import _get_eps, _allclose

        if _ppc_arch:
            fp_types = [np.float32, float]
        else:
            fp_types = [np.float32, float, np.longdouble]

        if hasattr(core, "real128"):
            fp_types.append(core.real128)

        x, y = make_vars("x", "y")
        func = [sin(x + y), x - par[0], x + y + par[1] * time]

        for fp_t in fp_types:
            fn = make_cfunc(func, vars=[y, x], fp_type=fp_t, batch_size=0)
            self.assertTrue(fn.batch_size > 0)

            if fp_t == np.longdouble:
                with self.assertRaises(ValueError) as cm:
                    make_cfunc(func, vars=[y, x], fp_type=fp_t, batch_size=2)
                self.assertTrue(
                    "Batch sizes greater than 1 are not supported for this floating-point type"
                    in str(cm.exception)
                )

            fn = make_cfunc(func, vars=[y, x], fp_type=fp_t)

            with self.assertRaises(ValueError) as cm:
                fn(
                    np.zeros((2, 5), dtype=fp_t),
                    pars=[fp_t(0)],
                    outputs=np.zeros((3, 1), dtype=fp_t),
                    time=np.zeros((5,), dtype=fp_t),
                )
            self.assertTrue(
                "The array of parameters provided for the evaluation of a compiled function has 1 dimension(s), but it must have 2 dimension(s) instead (i.e., the same number of dimensions as the array of inputs)"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                fn(
                    np.zeros((2, 5), dtype=fp_t),
                    pars=np.zeros(
                        (2, 4),
                        dtype=fp_t,
                    ),
                    time=np.zeros((5,), dtype=fp_t),
                )
            self.assertTrue(
                "The array of parameter values provided for the evaluation of a compiled function has 4 column(s), but the expected number of columns deduced from the outputs array is 5"
                in str(cm.exception)
            )

            nw_arr = np.zeros((3, 5), dtype=fp_t)
            nw_arr.setflags(write=False)
            with self.assertRaises(ValueError) as cm:
                fn(
                    np.zeros((2, 5), dtype=fp_t),
                    outputs=nw_arr,
                    pars=np.zeros((2, 5), dtype=fp_t),
                    time=np.zeros((5,), dtype=fp_t),
                )
            self.assertTrue(
                "The array of outputs provided for the evaluation of a compiled function is not writeable"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                fn(
                    np.zeros((2, 5), dtype=fp_t),
                    pars=np.zeros((2, 5), dtype=fp_t),
                )
            self.assertTrue(
                "An array of time values must be provided in order to evaluate a time-dependent function"
                in str(cm.exception)
            )

            with self.assertRaises(TypeError) as cm:
                fn(
                    np.zeros((2,), dtype=fp_t),
                    pars=np.zeros((2,), dtype=fp_t),
                    time=np.zeros((5,), dtype=fp_t),
                )
            self.assertTrue(
                "The time value cannot be an array when evaluating a compiled function over a single set of inputs, it should be a scalar instead"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                fn(
                    np.zeros((2, 5), dtype=fp_t),
                    pars=np.zeros((2, 5), dtype=fp_t),
                    time=np.zeros((5, 5), dtype=fp_t),
                )
            self.assertTrue(
                "The array of times provided for the evaluation of a compiled function has 2 dimension(s), but it must be one-dimensional instead"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                fn(
                    np.zeros((2, 5), dtype=fp_t),
                    pars=np.zeros((2, 5), dtype=fp_t),
                    time=np.zeros((6,), dtype=fp_t),
                )
            self.assertTrue(
                "The array of time values provided for the evaluation of a compiled function has a size of 6, but the expected size deduced from the outputs array is 5"
                in str(cm.exception)
            )

            for nevals in range(0, 10):
                fn = make_cfunc(func, vars=[y, x], fp_type=fp_t)

                # NOTE: deterministic seeding.
                rng = np.random.default_rng(nevals)

                # NOTE: long double rng not supported.
                inputs = rng.random((2, nevals), dtype=float).astype(fp_t)
                pars = rng.random((2, nevals), dtype=float).astype(fp_t)
                tm = rng.random((nevals,), dtype=float).astype(fp_t)

                eval_arr = fn(inputs=inputs, pars=pars, time=tm)
                self.assertTrue(
                    _allclose(
                        eval_arr[0],
                        np.sin(inputs[1, :] + inputs[0, :]),
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )
                self.assertTrue(
                    _allclose(
                        eval_arr[1],
                        inputs[1, :] - pars[0, :],
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )
                self.assertTrue(
                    _allclose(
                        eval_arr[2],
                        inputs[1, :] + inputs[0, :] + pars[1, :] * tm,
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )

                # NOTE: simple function without pars used to test
                # that the null pars span is created with the correct shape
                # in a parameter-less function.
                simple_fn = make_cfunc([x + y], vars=[x, y], fp_type=fp_t)
                eval_arr = simple_fn(inputs=inputs)
                self.assertTrue(
                    _allclose(
                        eval_arr[0],
                        inputs[0, :] + inputs[1, :],
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )

                # Check that eval_arr actually uses the memory
                # provided from the outputs argument.
                out_arr = np.zeros((3, nevals), dtype=fp_t)
                eval_arr = fn(inputs=inputs, pars=pars, outputs=out_arr, time=tm)
                self.assertEqual(id(eval_arr), id(out_arr))
                self.assertTrue(
                    _allclose(
                        eval_arr[0],
                        np.sin(inputs[1, :] + inputs[0, :]),
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )
                self.assertTrue(
                    _allclose(
                        eval_arr[1],
                        inputs[1, :] - pars[0, :],
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )
                self.assertTrue(
                    _allclose(
                        eval_arr[2],
                        inputs[1, :] + inputs[0, :] + pars[1, :] * tm,
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )

                # Test with non-owning arrays.
                eval_arr = fn(inputs=inputs[:], pars=pars, time=tm)
                self.assertTrue(
                    _allclose(
                        eval_arr[0],
                        np.sin(inputs[1, :] + inputs[0, :]),
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )
                self.assertTrue(
                    _allclose(
                        eval_arr[1],
                        inputs[1, :] - pars[0, :],
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )
                self.assertTrue(
                    _allclose(
                        eval_arr[2],
                        inputs[1, :] + inputs[0, :] + pars[1, :] * tm,
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )

                eval_arr = fn(inputs=inputs, pars=pars[:], time=tm)
                self.assertTrue(
                    _allclose(
                        eval_arr[0],
                        np.sin(inputs[1, :] + inputs[0, :]),
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )
                self.assertTrue(
                    _allclose(
                        eval_arr[1],
                        inputs[1, :] - pars[0, :],
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )
                self.assertTrue(
                    _allclose(
                        eval_arr[2],
                        inputs[1, :] + inputs[0, :] + pars[1, :] * tm,
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )

                eval_arr = fn(inputs=inputs, pars=pars, time=tm[:])
                self.assertTrue(
                    _allclose(
                        eval_arr[0],
                        np.sin(inputs[1, :] + inputs[0, :]),
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )
                self.assertTrue(
                    _allclose(
                        eval_arr[1],
                        inputs[1, :] - pars[0, :],
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )
                self.assertTrue(
                    _allclose(
                        eval_arr[2],
                        inputs[1, :] + inputs[0, :] + pars[1, :] * tm,
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )

                # Test with non-distinct arrays.
                if nevals == 0:
                    eval_arr = fn(inputs=inputs, pars=inputs, time=tm)
                    self.assertTrue(
                        _allclose(
                            eval_arr[0],
                            np.sin(inputs[1, :] + inputs[0, :]),
                            rtol=_get_eps(fp_t) * 10,
                            atol=_get_eps(fp_t) * 10,
                        )
                    )
                    self.assertTrue(
                        _allclose(
                            eval_arr[1],
                            inputs[1, :] - inputs[0, :],
                            rtol=_get_eps(fp_t) * 10,
                            atol=_get_eps(fp_t) * 10,
                        )
                    )
                    self.assertTrue(
                        _allclose(
                            eval_arr[2],
                            inputs[1, :] + inputs[0, :] + inputs[1, :] * tm,
                            rtol=_get_eps(fp_t) * 10,
                            atol=_get_eps(fp_t) * 10,
                        )
                    )

                    eval_arr = fn(inputs=inputs, pars=pars, time=pars[0, :])
                    self.assertTrue(
                        _allclose(
                            eval_arr[0],
                            np.sin(inputs[1, :] + inputs[0, :]),
                            rtol=_get_eps(fp_t) * 10,
                            atol=_get_eps(fp_t) * 10,
                        )
                    )
                    self.assertTrue(
                        _allclose(
                            eval_arr[1],
                            inputs[1, :] - pars[0, :],
                            rtol=_get_eps(fp_t) * 10,
                            atol=_get_eps(fp_t) * 10,
                        )
                    )
                    self.assertTrue(
                        _allclose(
                            eval_arr[2],
                            inputs[1, :] + inputs[0, :] + pars[1, :] * pars[0, :],
                            rtol=_get_eps(fp_t) * 10,
                            atol=_get_eps(fp_t) * 10,
                        )
                    )
                else:
                    with self.assertRaises(ValueError) as cm:
                        fn(inputs=inputs, pars=inputs, time=tm)
                    self.assertTrue(
                        "Potential memory overlaps detected when attempting to evaluate a compiled function: please make sure that all input arrays are distinct"
                        in str(cm.exception)
                    )
                    with self.assertRaises(ValueError) as cm:
                        fn(inputs=inputs, pars=pars, time=pars[0, :])
                    self.assertTrue(
                        "Potential memory overlaps detected when attempting to evaluate a compiled function: please make sure that all input arrays are distinct"
                        in str(cm.exception)
                    )

                # Test with arrays which are not C style.
                if nevals == 0:
                    inputs = rng.random((4, nevals), dtype=float).astype(fp_t)
                    inputs = inputs[::2]
                    eval_arr = fn(inputs=inputs, pars=pars, time=tm)
                    self.assertTrue(
                        _allclose(
                            eval_arr[0],
                            np.sin(inputs[1, :] + inputs[0, :]),
                            rtol=_get_eps(fp_t) * 10,
                            atol=_get_eps(fp_t) * 10,
                        )
                    )
                    self.assertTrue(
                        _allclose(
                            eval_arr[1],
                            inputs[1, :] - pars[0, :],
                            rtol=_get_eps(fp_t) * 10,
                            atol=_get_eps(fp_t) * 10,
                        )
                    )
                    self.assertTrue(
                        _allclose(
                            eval_arr[2],
                            inputs[1, :] + inputs[0, :] + pars[1, :] * tm,
                            rtol=_get_eps(fp_t) * 10,
                            atol=_get_eps(fp_t) * 10,
                        )
                    )

                    pars = rng.random((4, nevals), dtype=float).astype(fp_t)
                    pars = pars[::2]
                    eval_arr = fn(inputs=inputs, pars=pars, time=tm)
                    self.assertTrue(
                        _allclose(
                            eval_arr[0],
                            np.sin(inputs[1, :] + inputs[0, :]),
                            rtol=_get_eps(fp_t) * 10,
                            atol=_get_eps(fp_t) * 10,
                        )
                    )
                    self.assertTrue(
                        _allclose(
                            eval_arr[1],
                            inputs[1, :] - pars[0, :],
                            rtol=_get_eps(fp_t) * 10,
                            atol=_get_eps(fp_t) * 10,
                        )
                    )
                    self.assertTrue(
                        _allclose(
                            eval_arr[2],
                            inputs[1, :] + inputs[0, :] + pars[1, :] * tm,
                            rtol=_get_eps(fp_t) * 10,
                            atol=_get_eps(fp_t) * 10,
                        )
                    )
                else:
                    inputs = rng.random((4, nevals), dtype=float).astype(fp_t)
                    inputs = inputs[::2]
                    with self.assertRaises(ValueError) as cm:
                        fn(inputs=inputs, pars=pars, time=tm)
                    self.assertTrue(
                        "Invalid inputs array detected: the array is not C-style contiguous, please consider using numpy.ascontiguousarray() to turn it into one"
                        in str(cm.exception)
                    )

                    pars = rng.random((4, nevals), dtype=float).astype(fp_t)
                    pars = pars[::2]
                    with self.assertRaises(ValueError) as cm:
                        fn(inputs=inputs, pars=pars, time=tm)
                    self.assertTrue(
                        "Invalid inputs array detected: the array is not C-style contiguous, please consider using numpy.ascontiguousarray() to turn it into one"
                        in str(cm.exception)
                    )

                # Tests with no inputs.
                fn = make_cfunc(
                    [expression(fp_t(3)) + par[1], par[0] + time], [], fp_type=fp_t
                )

                inputs = rng.random((0, nevals), dtype=float).astype(fp_t)
                pars = rng.random((2, nevals), dtype=float).astype(fp_t)
                eval_arr = fn(inputs=inputs, pars=pars, time=tm)
                self.assertTrue(
                    _allclose(
                        eval_arr[0],
                        3 + pars[1, :],
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )
                self.assertTrue(
                    _allclose(
                        eval_arr[1],
                        pars[0, :] + tm,
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )

                eval_arr = fn(inputs=inputs[:], pars=pars, time=tm)
                self.assertTrue(
                    _allclose(
                        eval_arr[0],
                        3 + pars[1, :],
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )
                self.assertTrue(
                    _allclose(
                        eval_arr[1],
                        pars[0, :] + tm,
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )

                eval_arr = fn(inputs=inputs, pars=pars[:], time=tm)
                self.assertTrue(
                    _allclose(
                        eval_arr[0],
                        3 + pars[1, :],
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )
                self.assertTrue(
                    _allclose(
                        eval_arr[1],
                        pars[0, :] + tm,
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )

                eval_arr = fn(inputs=inputs, pars=pars, time=tm[:])
                self.assertTrue(
                    _allclose(
                        eval_arr[0],
                        3 + pars[1, :],
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )
                self.assertTrue(
                    _allclose(
                        eval_arr[1],
                        pars[0, :] + tm,
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )

                fn = make_cfunc(
                    [expression(fp_t(3)), expression(fp_t(4))], [], fp_type=fp_t
                )

                inputs = rng.random((0, nevals), dtype=float).astype(fp_t)
                pars = rng.random((0, nevals), dtype=float).astype(fp_t)
                eval_arr = fn(inputs=inputs, pars=pars)
                self.assertTrue(
                    _allclose(
                        eval_arr[0],
                        [3] * nevals,
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )
                self.assertTrue(
                    _allclose(
                        eval_arr[1],
                        [4] * nevals,
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )

                # Test case in which there are no pars but a pars array is provided anyway,
                # with the correct shape.
                fn = make_cfunc([x + y], [x, y], fp_type=fp_t)
                inputs = rng.random((2, nevals), dtype=float).astype(fp_t)
                eval_arr = fn(inputs=inputs, pars=np.zeros((0, nevals), dtype=fp_t))

                self.assertTrue(
                    _allclose(
                        eval_arr[0],
                        inputs[0, :] + inputs[1, :],
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )

                # Test a case without time but a time array is provided anyway, with
                # the correct shape.
                eval_arr = fn(
                    inputs=inputs, pars=np.zeros((0, nevals), dtype=fp_t), time=tm
                )

                self.assertTrue(
                    _allclose(
                        eval_arr[0],
                        inputs[0, :] + inputs[1, :],
                        rtol=_get_eps(fp_t) * 10,
                        atol=_get_eps(fp_t) * 10,
                    )
                )

        # Check throwing behaviour with long double on PPC.
        if _ppc_arch:
            with self.assertRaises(NotImplementedError):
                make_cfunc(func, vars=[y, x], fp_type=np.longdouble)

    def test_single(self):
        import numpy as np
        from . import make_cfunc, make_vars, sin, par, expression, core, time
        from .core import _ppc_arch
        from .test import _get_eps, _allclose

        if _ppc_arch:
            fp_types = [np.float32, float]
        else:
            fp_types = [np.float32, float, np.longdouble]

        if hasattr(core, "real128"):
            fp_types.append(core.real128)

        x, y = make_vars("x", "y")
        func = [sin(x + y), x - par[0], x + y + par[1] + time]

        # NOTE: perhaps in the future we can add
        # some more testing for high_accuracy, compact_mode,
        # etc., once we figure out how to test for them. Perhaps
        # examine the llvm states?
        for fp_t in fp_types:
            fn = make_cfunc(func, [x, y], fp_type=fp_t)

            with self.assertRaises(ValueError) as cm:
                fn([fp_t(1), fp_t(2)])
            self.assertTrue(
                "The array of parameter values provided for the evaluation of a compiled function has 0 element(s), but the number of parameters in the function is 2"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                fn(np.zeros((1, 2, 3), dtype=fp_t), pars=[fp_t(0)], time=fp_t(0))
            self.assertTrue(
                "The array of inputs provided for the evaluation of a compiled function has 3 dimensions, but it must have either 1 or 2 dimensions instead"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                fn([fp_t(0)], pars=[fp_t(0)], time=fp_t(0))
            self.assertTrue(
                "Invalid inputs array passed to a cfunc: the number of function inputs is 2, but the inputs array has a size of 1"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                fn(
                    np.zeros((2,), dtype=fp_t),
                    pars=[fp_t(0)],
                    outputs=np.zeros((2,), dtype=fp_t),
                    time=fp_t(0),
                )
            self.assertTrue(
                "Invalid outputs array passed to a cfunc: the number of function outputs is 3, but the outputs array has a size of 2"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                fn(
                    np.zeros((2,), dtype=fp_t),
                    pars=np.zeros((0,), dtype=fp_t),
                    time=fp_t(0),
                )
            self.assertTrue(
                "The array of parameter values provided for the evaluation of a compiled function has 0 element(s), but the number of parameters in the function is 2"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                fn(
                    np.zeros((2,), dtype=fp_t),
                    pars=np.zeros((2,), dtype=fp_t),
                )
            self.assertTrue(
                "A time value must be provided in order to evaluate a time-dependent function"
                in str(cm.exception)
            )

            eval_arr = fn([fp_t(1), fp_t(2)], pars=[fp_t(-5), fp_t(1)], time=fp_t(3))
            self.assertTrue(
                _allclose(
                    eval_arr,
                    [
                        np.sin(fp_t(1) + fp_t(2)),
                        fp_t(1) - fp_t(-5),
                        fp_t(1) + fp_t(2) + fp_t(1) + fp_t(3),
                    ],
                    rtol=_get_eps(fp_t) * 10,
                    atol=_get_eps(fp_t) * 10,
                )
            )

            # Check that eval_arr actually uses the memory
            # provided from the outputs argument.
            out_arr = np.zeros((3,), dtype=fp_t)
            eval_arr = fn(
                [fp_t(1), fp_t(2)],
                pars=[fp_t(-5), fp_t(1)],
                outputs=out_arr,
                time=fp_t(3),
            )
            self.assertTrue(np.shares_memory(eval_arr, out_arr))
            self.assertEqual(id(eval_arr), id(out_arr))
            self.assertTrue(
                _allclose(
                    eval_arr,
                    [
                        np.sin(fp_t(1) + fp_t(2)),
                        fp_t(1) - fp_t(-5),
                        fp_t(1) + fp_t(2) + fp_t(1) + fp_t(3),
                    ],
                    rtol=_get_eps(fp_t) * 10,
                    atol=_get_eps(fp_t) * 10,
                )
            )

            # Test with non-owning arrays.
            inputs = np.array([fp_t(1), fp_t(2)])
            eval_arr = fn(inputs=inputs[:], pars=[fp_t(-5), fp_t(1)], time=fp_t(3))
            self.assertTrue(
                _allclose(
                    eval_arr,
                    [
                        np.sin(fp_t(1) + fp_t(2)),
                        fp_t(1) - fp_t(-5),
                        fp_t(1) + fp_t(2) + fp_t(1) + fp_t(3),
                    ],
                    rtol=_get_eps(fp_t) * 10,
                    atol=_get_eps(fp_t) * 10,
                )
            )

            pars = np.array([fp_t(-5), fp_t(1)])
            eval_arr = fn(inputs=inputs, pars=pars[:], time=fp_t(3))
            self.assertTrue(
                _allclose(
                    eval_arr,
                    [
                        np.sin(fp_t(1) + fp_t(2)),
                        fp_t(1) - fp_t(-5),
                        fp_t(1) + fp_t(2) + fp_t(1) + fp_t(3),
                    ],
                    rtol=_get_eps(fp_t) * 10,
                    atol=_get_eps(fp_t) * 10,
                )
            )

            # Test with non-distinct arrays.
            with self.assertRaises(ValueError) as cm:
                fn(inputs=inputs, pars=inputs, time=fp_t(3))
            self.assertTrue(
                "Potential memory overlaps detected when attempting to evaluate a compiled function: please make sure that all input arrays are distinct"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                fn(inputs=inputs, pars=pars, time=fp_t(3), outputs=inputs)
            self.assertTrue(
                "Potential memory overlaps detected when attempting to evaluate a compiled function: please make sure that all input arrays are distinct"
                in str(cm.exception)
            )

            # Test with arrays which are not C style.
            inputs = np.zeros((4,), dtype=fp_t)
            inputs[::2] = [fp_t(1), fp_t(2)]
            with self.assertRaises(ValueError) as cm:
                fn(inputs=inputs[::2], pars=pars, time=fp_t(3))
            self.assertTrue(
                "Invalid inputs array detected: the array is not C-style contiguous, please consider using numpy.ascontiguousarray() to turn it into one"
                in str(cm.exception)
            )

            inputs = np.array([fp_t(1), fp_t(2)])
            pars = np.zeros((4,), dtype=fp_t)
            pars[::2] = [fp_t(-5), fp_t(1)]
            with self.assertRaises(ValueError) as cm:
                fn(inputs=inputs, pars=pars[::2], time=fp_t(3))
            self.assertTrue(
                "Invalid parameters array detected: the array is not C-style contiguous, please consider using numpy.ascontiguousarray() to turn it into one"
                in str(cm.exception)
            )

            inputs = np.array([fp_t(1), fp_t(2)])
            pars = np.zeros((4,), dtype=fp_t)
            outputs = np.zeros((6,), dtype=fp_t)
            with self.assertRaises(ValueError) as cm:
                fn(inputs=inputs, pars=pars, outputs=outputs[::2], time=fp_t(3))
            self.assertTrue(
                "Invalid outputs array detected: the array is not C-style contiguous, please consider using numpy.ascontiguousarray() to turn it into one"
                in str(cm.exception)
            )

            # Tests with no inputs.
            fn = make_cfunc(
                [expression(fp_t(3)) + par[1], par[0] + time], [], fp_type=fp_t
            )

            eval_arr = fn(
                inputs=np.zeros((0,), dtype=fp_t), pars=[fp_t(1), fp_t(2)], time=fp_t(3)
            )
            self.assertTrue(
                _allclose(
                    eval_arr,
                    [fp_t(3) + 2, fp_t(1) + fp_t(3)],
                    rtol=_get_eps(fp_t) * 10,
                    atol=_get_eps(fp_t) * 10,
                )
            )

            inputs = np.zeros((0,), dtype=fp_t)
            eval_arr = fn(inputs=inputs[:], pars=[fp_t(1), fp_t(2)], time=fp_t(3))
            self.assertTrue(
                _allclose(
                    eval_arr,
                    [fp_t(3) + 2, fp_t(1) + fp_t(3)],
                    rtol=_get_eps(fp_t) * 10,
                    atol=_get_eps(fp_t) * 10,
                )
            )

            fn = make_cfunc(
                [expression(fp_t(3)), expression(fp_t(4)) + time], [], fp_type=fp_t
            )

            eval_arr = fn(
                inputs=np.zeros((0,), dtype=fp_t),
                pars=np.zeros((0,), dtype=fp_t),
                time=fp_t(3),
            )
            self.assertTrue(
                _allclose(
                    eval_arr,
                    [fp_t(3), fp_t(4) + fp_t(3)],
                    rtol=_get_eps(fp_t) * 10,
                    atol=_get_eps(fp_t) * 10,
                )
            )

            eval_arr = fn(
                inputs=np.zeros((0,), dtype=fp_t),
                pars=np.zeros((0,), dtype=fp_t),
                time=fp_t(3),
            )
            self.assertTrue(
                _allclose(
                    eval_arr,
                    [fp_t(3), fp_t(4) + fp_t(3)],
                    rtol=_get_eps(fp_t) * 10,
                    atol=_get_eps(fp_t) * 10,
                )
            )

            # Test case in which there are no pars but a pars array is provided anyway,
            # with the correct shape.
            fn = make_cfunc([x + y], [x, y], fp_type=fp_t)
            eval_arr = fn(inputs=[fp_t(1), fp_t(2)], pars=np.zeros((0,), dtype=fp_t))

            self.assertEqual(eval_arr[0], 3)

            # Same but with time.
            eval_arr = fn(
                inputs=[fp_t(1), fp_t(2)], pars=np.zeros((0,), dtype=fp_t), time=fp_t(3)
            )

            self.assertEqual(eval_arr[0], 3)
