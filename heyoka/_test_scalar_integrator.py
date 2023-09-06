# Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class scalar_integrator_test_case(_ut.TestCase):
    def test_llvm_state_settings(self):
        # Test to check that the llvm state flags
        # are correctly propagated through the integrator
        # constructor.

        from . import taylor_adaptive
        from .model import pendulum

        ta = taylor_adaptive(pendulum(), [0.0, 0.0])

        self.assertFalse(ta.llvm_state.force_avx512)
        self.assertFalse(ta.llvm_state.slp_vectorize)

        ta = taylor_adaptive(
            pendulum(), [0.0, 0.0], force_avx512=True, slp_vectorize=True
        )

        self.assertTrue(ta.llvm_state.force_avx512)
        self.assertTrue(ta.llvm_state.slp_vectorize)

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
        self.assertEqual(ta.state_vars, [x, v])
        self.assertEqual(ta.rhs, [v, -9.8 * sin(x)])

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

        # Test with a non-callable callback.
        with self.assertRaises(TypeError) as cm:
            ta.time = 0.0
            ta.propagate_grid([0.0, 1.0, 2.0], callback="hello world")
        self.assertTrue(
            "cannot be used as a step callback because it is not callable"
            in str(cm.exception)
        )

        # Broken callback with wrong return type.
        class broken_cb:
            def __call__(self, ta):
                return []

        with self.assertRaises(TypeError) as cm:
            ta.time = 0.0
            ta.propagate_grid([0.0, 1.0, 2.0], callback=broken_cb())
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

        ta.time = 0.0
        ta.propagate_until(10.0, callback=cb_hook())
        self.assertTrue(ta.foo)
        delattr(ta, "foo")

        ta.time = 0.0
        ta.propagate_for(10.0, callback=cb_hook())
        self.assertTrue(ta.foo)
        delattr(ta, "foo")

        ta.time = 0.0
        ta.propagate_grid([0.0, 1.0, 2.0], callback=cb_hook())
        self.assertTrue(ta.foo)
        delattr(ta, "foo")

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
