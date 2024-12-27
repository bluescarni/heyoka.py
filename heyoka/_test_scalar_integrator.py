# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

        from . import taylor_adaptive, code_model
        from .model import pendulum
        from sys import getrefcount

        ta = taylor_adaptive(pendulum(), [0.0, 0.0])

        # Check correct reference count handling of the
        # llvm_state property.
        rc = getrefcount(ta)
        tmp = ta.llvm_state
        self.assertEqual(getrefcount(ta), rc + 1)

        self.assertFalse(ta.llvm_state.force_avx512)
        self.assertFalse(ta.llvm_state.slp_vectorize)

        ta = taylor_adaptive(
            pendulum(),
            [0.0, 0.0],
            force_avx512=True,
            slp_vectorize=True,
            parjit=True,
            compact_mode=True,
            code_model=code_model.large,
        )

        rc = getrefcount(ta)
        tmp = ta.llvm_state
        self.assertEqual(getrefcount(ta), rc + 1)

        self.assertTrue(ta.llvm_state.force_avx512)
        self.assertTrue(ta.llvm_state.slp_vectorize)
        self.assertTrue(ta.llvm_state.parjit)
        self.assertEqual(ta.llvm_state.code_model, code_model.large)

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
        from .core import _ppc_arch
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

        if _ppc_arch:
            return

        # BUG: the dtime setter used to be hard-coded
        # to double.
        from numpy import longdouble as ld

        ta = taylor_adaptive(sys=sys, state=[ld(0.0), ld(0.25)], fp_type=ld)
        ta.dtime = (ld("1.1"), ld(0))
        self.assertEqual(ta.dtime, (ld("1.1"), ld(0)))

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
        from . import taylor_adaptive, make_vars, t_event, sin, core
        from .core import _ppc_arch
        from .callback import angle_reducer
        import numpy as np

        if _ppc_arch:
            fp_types = [np.float32, float]
        else:
            fp_types = [np.float32, float, np.longdouble]

        if hasattr(core, "real128"):
            fp_types.append(core.real128)

        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]

        for fp_t in fp_types:
            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0.0), fp_t(0.25)],
                t_events=[t_event(v, fp_type=fp_t)],
                fp_type=fp_t,
            )

            self.assertTrue(ta.with_events)
            self.assertFalse(ta.compact_mode)
            self.assertFalse(ta.high_accuracy)
            self.assertEqual(ta.sys, sys)

            # Test init without a state.
            ta = taylor_adaptive(
                sys=sys,
                fp_type=fp_t,
            )

            self.assertTrue(np.all(ta.state == [0, 0]))

            ta = taylor_adaptive(
                sys=sys,
                state=[],
                fp_type=fp_t,
            )

            self.assertTrue(np.all(ta.state == [0, 0]))

            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0.0), fp_t(0.25)],
                compact_mode=True,
                high_accuracy=True,
                fp_type=fp_t,
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
                state=[fp_t(0.0), fp_t(0.25)],
                compact_mode=True,
                high_accuracy=True,
                force_avx512=True,
                fast_math=True,
                opt_level=0,
                fp_type=fp_t,
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

            ta.propagate_until(fp_t(10.0), callback=cb)

            self.assertTrue(ta.counter > 0)
            orig_ct = ta.counter

            ta.propagate_for(fp_t(10.0), callback=cb)

            self.assertTrue(ta.counter > orig_ct)
            orig_ct = ta.counter

            ta.time = fp_t(0.0)
            ta.propagate_grid([fp_t(0.0), fp_t(1.0), fp_t(2.0)], callback=cb)

            self.assertTrue(ta.counter > orig_ct)

            # Test that no copies of the callback are performed.
            class cb:
                def __call__(_, ta):
                    self.assertEqual(id(_), _.orig_id)

                    return True

            cb_inst = cb()
            cb_inst.orig_id = id(cb_inst)

            ta.time = fp_t(0.0)
            res = ta.propagate_until(fp_t(10.0), callback=cb_inst)
            self.assertEqual(id(cb_inst), id(res[-1]))
            res = ta.propagate_for(fp_t(10.0), callback=cb_inst)
            self.assertEqual(id(cb_inst), id(res[-1]))
            ta.time = fp_t(0.0)
            res = ta.propagate_grid([fp_t(0.0), fp_t(1.0), fp_t(2.0)], callback=cb_inst)
            self.assertEqual(id(cb_inst), id(res[-2]))

            # Test with a non-callable callback.
            with self.assertRaises(TypeError) as cm:
                ta.time = fp_t(0.0)
                ta.propagate_grid(
                    [fp_t(0.0), fp_t(1.0), fp_t(2.0)], callback="hello world"
                )
            self.assertTrue(
                "cannot be used as a step callback because it is not callable"
                in str(cm.exception)
            )

            # Broken callback with wrong return type.
            class broken_cb:
                def __call__(self, ta):
                    return []

            with self.assertRaises(TypeError) as cm:
                ta.time = fp_t(0.0)
                ta.propagate_grid(
                    [fp_t(0.0), fp_t(1.0), fp_t(2.0)], callback=broken_cb()
                )
            self.assertTrue(
                "The call operator of a step callback is expected to return a boolean,"
                " but a value of type"
                in str(cm.exception)
            )

            # Callback with pre_hook().
            class cb_hook:
                def __call__(_, ta):
                    return True

                def pre_hook(self, ta):
                    ta.foo = True

            ta.time = fp_t(0.0)
            ta.propagate_until(fp_t(10.0), callback=cb_hook())
            self.assertTrue(ta.foo)
            delattr(ta, "foo")

            ta.time = fp_t(0.0)
            ta.propagate_for(fp_t(10.0), callback=cb_hook())
            self.assertTrue(ta.foo)
            delattr(ta, "foo")

            ta.time = fp_t(0.0)
            ta.propagate_grid([fp_t(0.0), fp_t(1.0), fp_t(2.0)], callback=cb_hook())
            self.assertTrue(ta.foo)
            delattr(ta, "foo")

    def test_events(self):
        from . import nt_event, t_event, make_vars, sin, taylor_adaptive, core
        from .core import _ppc_arch
        import numpy as np

        if _ppc_arch:
            fp_types = [np.float32, float]
        else:
            fp_types = [np.float32, float, np.longdouble]

        if hasattr(core, "real128"):
            fp_types.append(core.real128)

        x, v = make_vars("x", "v")

        # Use a pendulum for testing purposes.
        sys = [(x, v), (v, -9.8 * sin(x))]

        for fp_t in fp_types:

            def cb0(ta, t, d_sgn):
                pass

            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0.0), fp_t(0.25)],
                nt_events=[nt_event(v * v - 1e-6, cb0, fp_type=fp_t)],
                t_events=[t_event(v, fp_type=fp_t)],
                fp_type=fp_t,
            )

            self.assertTrue(ta.with_events)
            self.assertEqual(len(ta.t_events), 1)
            self.assertEqual(len(ta.nt_events), 1)

            oc = ta.propagate_until(fp_t(1e9))[0]
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
            fp_types = [np.float32, float]
        else:
            fp_types = [np.float32, float, np.longdouble]

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

    def test_step_callback(self):
        from . import taylor_adaptive, make_vars, sin, core
        from .core import _ppc_arch
        from .callback import angle_reducer
        import numpy as np

        if _ppc_arch:
            fp_types = [np.float32, float]
        else:
            fp_types = [np.float32, float, np.longdouble]

        if hasattr(core, "real128"):
            fp_types.append(core.real128)

        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]

        # Callback with pre_hook().
        class cb_hook:
            def __call__(_, ta):
                return True

            def pre_hook(self, ta):
                ta.foo = True

        for fp_t in fp_types:
            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0.0), fp_t(10.0)],
                fp_type=fp_t,
            )

            # List overaload.
            cb1 = cb_hook()
            cb2 = cb_hook()
            id_cb1 = id(cb1)
            id_cb2 = id(cb2)
            res = ta.propagate_for(fp_t(10.0), callback=[cb1, cb2])
            self.assertTrue(isinstance(res[-1], list))
            self.assertTrue(isinstance(res[-1][0], cb_hook))
            self.assertTrue(isinstance(res[-1][1], cb_hook))
            self.assertEqual(id(res[-1][0]), id_cb1)
            self.assertEqual(id(res[-1][1]), id_cb2)
            self.assertTrue(hasattr(ta, "foo"))

            # Try with a C++ callback too.
            res = ta.propagate_until(
                fp_t(20.0), callback=[cb1, angle_reducer([x]), cb2]
            )
            self.assertTrue(isinstance(res[-1], list))
            self.assertTrue(isinstance(res[-1][0], cb_hook))
            self.assertTrue(isinstance(res[-1][1], angle_reducer))
            self.assertTrue(isinstance(res[-1][2], cb_hook))
            self.assertEqual(id(res[-1][0]), id_cb1)
            self.assertEqual(id(res[-1][2]), id_cb2)
            self.assertTrue((ta.state[0] >= fp_t(0) and ta.state[0] < fp_t(6.29)))

            # Single callback overload.
            res = ta.propagate_grid([fp_t(20.0), fp_t(30.0)], callback=cb1)
            self.assertTrue(isinstance(res[-2], cb_hook))
            self.assertEqual(id(res[-2]), id_cb1)

            res = ta.propagate_for(fp_t(10.0), callback=angle_reducer([x]))
            self.assertTrue(isinstance(res[-1], angle_reducer))
            self.assertTrue((ta.state[0] >= fp_t(0) and ta.state[0] < fp_t(6.29)))
