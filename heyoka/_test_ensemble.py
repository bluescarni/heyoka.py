# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


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
        from .callback import angle_reducer
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

                self.assertTrue(np.all(loc_ret[0](5.0) == ret[i][1](5.0)))

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

                self.assertTrue(np.all(loc_ret[1] == ret[i][2]))

                for j in range(4):
                    self.assertAlmostEqual(ret[i][0].time[j], 20.0)
                self.assertTrue(np.all(ta.state == ret[i][0].state))
                self.assertTrue(np.all(ta.time == ret[i][0].time))
                self.assertEqual(ta.propagate_res, ret[i][0].propagate_res)

        # Check that callbacks are deep-copied in thread-based
        # ensemble propagations.

        class step_cb:
            def __call__(_, ta):
                self.assertNotEqual(id(_), _.orig_id)

                return True

        cb = step_cb()
        cb.orig_id = id(cb)

        def gen(ta, idx):
            ta.set_time(0.0)
            ta.state[:] = ics[idx]

            return ta

        ret = ensemble_propagate_for_batch(
            ta, 20.0, 10, gen, algorithm="thread", max_workers=8, callback=cb
        )

        for r in ret:
            self.assertTrue(isinstance(r[2], step_cb))

        # Test the list overload too.
        ret = ensemble_propagate_for_batch(
            ta,
            20.0,
            10,
            gen,
            algorithm="thread",
            max_workers=8,
            callback=[cb, angle_reducer([x])],
        )

        for r in ret:
            self.assertTrue(isinstance(r[2], list))
            self.assertTrue(isinstance(r[2][0], step_cb))
            self.assertTrue(isinstance(r[2][1], angle_reducer))
            self.assertEqual(len(r[2]), 2)

        # Test s11n machinery in multi-processing situations.
        ret = ensemble_propagate_for_batch(
            ta,
            20.0,
            10,
            gen,
            algorithm="process",
            callback=[cb, angle_reducer([x])],
        )

        for r in ret:
            self.assertTrue(isinstance(r[2], list))
            self.assertTrue(isinstance(r[2][0], step_cb))
            self.assertTrue(isinstance(r[2][1], angle_reducer))
            self.assertEqual(len(r[2]), 2)

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
        from .callback import angle_reducer
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
                self.assertEqual(loc_ret[:-2], ret[i][1:-2])
                self.assertEqual(ta.time, ret[i][0].time)

                self.assertTrue(np.all(loc_ret[-2](5.0) == ret[i][-2](5.0)))

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
            "Cannot perform an ensemble propagate_until/for(): the final epoch/time"
            " interval must be a scalar, not an iterable object"
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            ensemble_propagate_for(ta, [20.0], 10, gen)
        self.assertTrue(
            "Cannot perform an ensemble propagate_until/for(): the final epoch/time"
            " interval must be a scalar, not an iterable object"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            ensemble_propagate_grid(ta, [[20.0, 20.0]], 10, gen)
        self.assertTrue(
            "Cannot perform an ensemble propagate_grid(): the input time grid must be"
            " one-dimensional, but instead it has 2 dimensions"
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            ensemble_propagate_until(ta, 20.0, 10, gen, max_delta_t=[10])
        self.assertTrue(
            'Cannot perform an ensemble propagate_until/for/grid(): the "max_delta_t"'
            " argument must be a scalar, not an iterable object"
            in str(cm.exception)
        )

        # NOTE: check that the chunksize option is not recognised
        # in threaded mode.
        with self.assertRaises(TypeError) as cm:
            ensemble_propagate_until(ta, 20.0, 10, gen, chunksize=1)

        # Check that callbacks are deep-copied in thread-based
        # ensemble propagations.

        class step_cb:
            def __call__(_, ta):
                self.assertNotEqual(id(_), _.orig_id)

                return True

        cb = step_cb()
        cb.orig_id = id(cb)

        def gen(ta, idx):
            ta.time = 0.0
            ta.state[:] = ics[idx]

            return ta

        ret = ensemble_propagate_for(
            ta, 20.0, 10, gen, algorithm="thread", max_workers=8, callback=cb
        )

        for r in ret:
            self.assertTrue(isinstance(r[-1], step_cb))

        # Test the list overload too.
        ret = ensemble_propagate_for(
            ta,
            20.0,
            10,
            gen,
            algorithm="thread",
            max_workers=8,
            callback=[cb, angle_reducer([x])],
        )

        for r in ret:
            self.assertTrue(isinstance(r[-1], list))
            self.assertTrue(isinstance(r[-1][0], step_cb))
            self.assertTrue(isinstance(r[-1][1], angle_reducer))
            self.assertEqual(len(r[-1]), 2)

        # Test s11n machinery in multi-processing situations.
        ret = ensemble_propagate_for(
            ta,
            20.0,
            10,
            gen,
            algorithm="process",
            callback=[cb, angle_reducer([x])],
        )

        for r in ret:
            self.assertTrue(isinstance(r[-1], list))
            self.assertTrue(isinstance(r[-1][0], step_cb))
            self.assertTrue(isinstance(r[-1][1], angle_reducer))
            self.assertEqual(len(r[-1]), 2)
