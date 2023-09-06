# Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class batch_integrator_test_case(_ut.TestCase):
    def test_llvm_state_settings(self):
        # Test to check that the llvm state flags
        # are correctly propagated through the integrator
        # constructor.

        from . import taylor_adaptive_batch
        from .model import pendulum

        ta = taylor_adaptive_batch(pendulum(), [[0.0, 0.0], [0.0, 0.0]])

        self.assertFalse(ta.llvm_state.force_avx512)
        self.assertFalse(ta.llvm_state.slp_vectorize)

        ta = taylor_adaptive_batch(
            pendulum(), [[0.0, 0.0], [0.0, 0.0]], force_avx512=True, slp_vectorize=True
        )

        self.assertTrue(ta.llvm_state.force_avx512)
        self.assertTrue(ta.llvm_state.slp_vectorize)

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

        # Test with a non-callable callback.
        with self.assertRaises(TypeError) as cm:
            ta.set_time(0.0)
            ta.state[:] = ic
            ta.propagate_for(10.0, callback="hello world")
        self.assertTrue(
            "cannot be used as a step callback because it is not callable"
            in str(cm.exception)
        )

        # Broken callback with wrong return type.
        class broken_cb:
            def __call__(self, ta):
                return []

        with self.assertRaises(TypeError) as cm:
            ta.set_time(0.0)
            ta.state[:] = ic
            ta.propagate_for(10.0, callback=broken_cb())
        self.assertTrue(
            "The call operator of a step callback is expected to return a boolean, but a value of type"
            in str(cm.exception)
        )

        # Callback with pre_hook().
        class cb_hook:
            def __call__(_, ta):
                return True

            def pre_hook(self, ta):
                ta.foo = True

        ta.set_time(0.0)
        ta.state[:] = ic
        ta.propagate_for(10.0, callback=cb_hook())
        self.assertTrue(ta.foo)
        delattr(ta, "foo")

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

        # Test with a non-callable callback.
        with self.assertRaises(TypeError) as cm:
            ta.set_time(0.0)
            ta.state[:] = ic
            ta.propagate_until(10.0, callback="hello world")
        self.assertTrue(
            "cannot be used as a step callback because it is not callable"
            in str(cm.exception)
        )

        # Broken callback with wrong return type.
        class broken_cb:
            def __call__(self, ta):
                return []

        with self.assertRaises(TypeError) as cm:
            ta.set_time(0.0)
            ta.state[:] = ic
            ta.propagate_until(10.0, callback=broken_cb())
        self.assertTrue(
            "The call operator of a step callback is expected to return a boolean, but a value of type"
            in str(cm.exception)
        )

        # Callback with pre_hook().
        class cb_hook:
            def __call__(_, ta):
                return True

            def pre_hook(self, ta):
                ta.foo = True

        ta.set_time(0.0)
        ta.state[:] = ic
        ta.propagate_until(10.0, callback=cb_hook())
        self.assertTrue(ta.foo)
        delattr(ta, "foo")

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
        self.assertEqual(ta.state_vars, [x, v])
        self.assertEqual(ta.rhs, [v, -9.8 * sin(x)])

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

        # Test with a non-callable callback.
        with self.assertRaises(TypeError) as cm:
            ta.set_time(0.0)
            ta.state[:] = [x_ic, v_ic]
            ta.propagate_grid(grid, callback="hello world")
        self.assertTrue(
            "cannot be used as a step callback because it is not callable"
            in str(cm.exception)
        )

        # Broken callback with wrong return type.
        class broken_cb:
            def __call__(self, ta):
                return []

        with self.assertRaises(TypeError) as cm:
            ta.set_time(0.0)
            ta.state[:] = [x_ic, v_ic]
            ta.propagate_grid(grid, callback=broken_cb())
        self.assertTrue(
            "The call operator of a step callback is expected to return a boolean, but a value of type"
            in str(cm.exception)
        )

        # Callback with pre_hook().
        class cb_hook:
            def __call__(_, ta):
                return True

            def pre_hook(self, ta):
                ta.foo = True

        ta.set_time(0.0)
        ta.state[:] = [x_ic, v_ic]
        ta.propagate_grid(grid, callback=cb_hook())
        self.assertTrue(ta.foo)
        delattr(ta, "foo")
