# Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import gc
import pickle
import unittest
from copy import copy, deepcopy
from sys import getrefcount

import numpy as np

from .. import (
    _core,
    event_direction,
    make_vars,
    nt_event,
    nt_event_batch,
    sin,
    t_event,
    t_event_batch,
    taylor_adaptive,
    taylor_adaptive_batch,
    taylor_outcome,
)


class event_classes_test_case(unittest.TestCase):
    def test_basic(self):
        x, v = make_vars("x", "v")

        if _core._ppc_arch:
            fp_types = [np.float32, float]
        else:
            fp_types = [np.float32, float, np.longdouble]

        if hasattr(_core, "real128"):
            fp_types.append(_core.real128)

        for fp_t in fp_types:
            # Non-terminal event.
            ev = nt_event(x + v, lambda _: _, fp_type=fp_t)

            self.assertTrue(" non-terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::any" in repr(ev))
            self.assertEqual(ev.expression, x + v)
            self.assertEqual(ev.direction, event_direction.any)
            self.assertFalse(ev.callback is None)

            ev = nt_event(ex=x + v, callback=lambda _: _, fp_type=fp_t)
            self.assertTrue(" non-terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::any" in repr(ev))
            self.assertEqual(ev.expression, x + v)
            self.assertEqual(ev.direction, event_direction.any)
            self.assertFalse(ev.callback is None)

            ev = nt_event(
                ex=x + v,
                callback=lambda _: _,
                direction=event_direction.positive,
                fp_type=fp_t,
            )
            self.assertTrue(" non-terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::positive" in repr(ev))
            self.assertEqual(ev.expression, x + v)
            self.assertEqual(ev.direction, event_direction.positive)
            self.assertFalse(ev.callback is None)

            ev = nt_event(
                ex=x + v,
                callback=lambda _: _,
                direction=event_direction.negative,
                fp_type=fp_t,
            )
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
            ev = nt_event(
                ex=x + v, callback=lcb, direction=event_direction.negative, fp_type=fp_t
            )
            self.assertEqual(ev.callback.n, 0)
            cb = ev.callback
            cb(1, 2, 3)
            cb(1, 2, 3)
            cb(1, 2, 3)
            self.assertEqual(ev.callback.n, 3)
            ev.callback.n = 0
            self.assertEqual(ev.callback.n, 0)
            self.assertNotEqual(id(lcb), id(ev.callback))

            with self.assertRaises(ValueError) as cm:
                nt_event(
                    ex=x + v,
                    callback=lambda _: _,
                    direction=event_direction(10),
                    fp_type=fp_t,
                )
            self.assertTrue("10 is not a valid event_direction" in str(cm.exception))

            with self.assertRaises(TypeError) as cm:
                nt_event(ex=x + v, callback=3, fp_type=fp_t)
            self.assertTrue(
                "An object of type '{}' cannot be used as an event callback because it"
                " is not callable".format(str(type(3)))
                in str(cm.exception)
            )

            with self.assertRaises(TypeError) as cm:
                nt_event(ex=x + v, callback=None, fp_type=fp_t)
            self.assertTrue(
                "An object of type '{}' cannot be used as an event callback because it"
                " is not callable".format(str(type(None)))
                in str(cm.exception)
            )

            ev = nt_event(
                ex=x + v,
                callback=lambda _: _,
                direction=event_direction.negative,
                fp_type=fp_t,
            )
            ev = pickle.loads(pickle.dumps(ev))
            self.assertTrue(" non-terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::negative" in repr(ev))

            # Test dynamic attributes.
            ev.foo = "hello world"
            ev = pickle.loads(pickle.dumps(ev))
            self.assertEqual(ev.foo, "hello world")

            # Test copy semantics.
            class foo:
                pass

            ev.bar = foo()

            self.assertEqual(id(ev.bar), id(copy(ev).bar))
            self.assertNotEqual(id(ev.bar), id(deepcopy(ev).bar))

            # Test to ensure a callback extracted from the event
            # is kept alive and usable when the event is destroyed.
            ev = nt_event(
                ex=x + v,
                callback=local_cb(),
                direction=event_direction.negative,
                fp_type=fp_t,
            )
            out_cb = ev.callback
            del ev
            gc.collect()
            out_cb(1, 2, 3)
            out_cb(1, 2, 3)
            out_cb(1, 2, 3)
            self.assertEqual(out_cb.n, 3)

            # Terminal event.
            ev = t_event(x + v, fp_type=fp_t)

            self.assertTrue(" terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::any" in repr(ev))
            self.assertTrue(": no" in repr(ev))
            self.assertTrue("auto" in repr(ev))
            self.assertEqual(ev.expression, x + v)
            self.assertEqual(ev.direction, event_direction.any)
            self.assertEqual(ev.cooldown, fp_t(-1))
            self.assertTrue(ev.callback is None)

            ev = t_event(
                x + v,
                fp_type=fp_t,
                direction=event_direction.negative,
                cooldown=fp_t(3),
            )

            self.assertTrue(" terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::negative" in repr(ev))
            self.assertTrue(": no" in repr(ev))
            self.assertTrue("3" in repr(ev))
            self.assertEqual(ev.expression, x + v)
            self.assertEqual(ev.direction, event_direction.negative)
            self.assertEqual(ev.cooldown, fp_t(3))
            self.assertTrue(ev.callback is None)

            ev = t_event(
                x + v,
                fp_type=fp_t,
                direction=event_direction.positive,
                cooldown=fp_t(3),
                callback=lambda _: _,
            )

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

                def __call__(self, ta, d_sgn):
                    self.n = self.n + 1

            lcb = local_cb()
            ev = t_event(
                x + v,
                fp_type=fp_t,
                direction=event_direction.positive,
                cooldown=fp_t(3),
                callback=lcb,
            )

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
            cb(1, 2)
            cb(1, 2)
            cb(1, 2)
            self.assertEqual(ev.callback.n, 3)
            ev.callback.n = 0
            self.assertEqual(ev.callback.n, 0)
            self.assertNotEqual(id(lcb), id(ev.callback))

            ev = t_event(
                x + v,
                fp_type=fp_t,
                direction=event_direction.positive,
                cooldown=fp_t(3),
                callback=None,
            )
            self.assertTrue(ev.callback is None)

            with self.assertRaises(ValueError) as cm:
                t_event(
                    x + v,
                    fp_type=fp_t,
                    direction=event_direction(45),
                    cooldown=fp_t(3),
                    callback=lambda _: _,
                )
            self.assertTrue("45 is not a valid event_direction" in str(cm.exception))

            with self.assertRaises(TypeError) as cm:
                t_event(x + v, callback=3, fp_type=fp_t)
            self.assertTrue(
                "An object of type '{}' cannot be used as an event callback because it"
                " is not callable".format(str(type(3)))
                in str(cm.exception)
            )

            ev = t_event(
                x + v,
                fp_type=fp_t,
                direction=event_direction.positive,
                cooldown=fp_t(3),
                callback=lambda _: _,
            )

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

            # Test copy semantics.
            class foo:
                pass

            ev.bar = foo()

            self.assertEqual(id(ev.bar), id(copy(ev).bar))
            self.assertNotEqual(id(ev.bar), id(deepcopy(ev).bar))

            # Test also with empty callback.
            ev = t_event(
                x + v,
                fp_type=fp_t,
                direction=event_direction.positive,
                cooldown=fp_t(3),
            )

            ev = pickle.loads(pickle.dumps(ev))
            self.assertTrue(" terminal" in repr(ev))
            self.assertTrue("(x + v)" in repr(ev))
            self.assertTrue("event_direction::positive" in repr(ev))
            self.assertTrue(": no" in repr(ev))
            self.assertTrue("3" in repr(ev))

            # Test to ensure a callback extracted from the event
            # is kept alive and usable when the event is destroyed.
            ev = t_event(
                ex=x + v,
                callback=local_cb(),
                direction=event_direction.negative,
                fp_type=fp_t,
            )
            out_cb = ev.callback
            del ev
            gc.collect()
            out_cb(1, 2)
            out_cb(1, 2)
            out_cb(1, 2)
            self.assertEqual(out_cb.n, 3)

        # Unsupported fp_type.
        with self.assertRaises(TypeError) as cm:
            nt_event(x + v, lambda _: _, fp_type=str)
        self.assertTrue(
            'The floating-point type "{}" is not recognized/supported'.format(str)
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            t_event(x + v, fp_type=list)
        self.assertTrue(
            'The floating-point type "{}" is not recognized/supported'.format(list)
            in str(cm.exception)
        )

        # Batch events.
        ev = nt_event_batch(x + v, lambda _: _)
        self.assertTrue(" non-terminal" in repr(ev))
        self.assertTrue("(x + v)" in repr(ev))
        self.assertTrue("event_direction::any" in repr(ev))
        self.assertEqual(ev.expression, x + v)
        self.assertEqual(ev.direction, event_direction.any)
        self.assertFalse(ev.callback is None)

        ev = nt_event_batch(ex=x + v, callback=lambda _: _, fp_type=float)
        self.assertTrue(" non-terminal" in repr(ev))
        self.assertTrue("(x + v)" in repr(ev))
        self.assertTrue("event_direction::any" in repr(ev))
        self.assertEqual(ev.expression, x + v)
        self.assertEqual(ev.direction, event_direction.any)
        self.assertFalse(ev.callback is None)

        ev = nt_event_batch(
            ex=x + v, callback=lambda _: _, direction=event_direction.positive
        )
        self.assertTrue(" non-terminal" in repr(ev))
        self.assertTrue("(x + v)" in repr(ev))
        self.assertTrue("event_direction::positive" in repr(ev))
        self.assertEqual(ev.expression, x + v)
        self.assertEqual(ev.direction, event_direction.positive)
        self.assertFalse(ev.callback is None)

        ev = nt_event_batch(
            ex=x + v, callback=lambda _: _, direction=event_direction.negative
        )
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
        ev = nt_event_batch(ex=x + v, callback=lcb, direction=event_direction.negative)
        self.assertEqual(ev.callback.n, 0)
        cb = ev.callback
        cb(1, 2, 3)
        cb(1, 2, 3)
        cb(1, 2, 3)
        self.assertEqual(ev.callback.n, 3)
        ev.callback.n = 0
        self.assertEqual(ev.callback.n, 0)
        self.assertNotEqual(id(lcb), id(ev.callback))

        with self.assertRaises(ValueError) as cm:
            nt_event_batch(
                ex=x + v, callback=lambda _: _, direction=event_direction(10)
            )
        self.assertTrue("10 is not a valid event_direction" in str(cm.exception))

        with self.assertRaises(TypeError) as cm:
            nt_event_batch(ex=x + v, callback=3)
        self.assertTrue(
            "An object of type '{}' cannot be used as an event callback because it is"
            " not callable".format(str(type(3)))
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            nt_event_batch(ex=x + v, callback=None)
        self.assertTrue(
            "An object of type '{}' cannot be used as an event callback because it is"
            " not callable".format(str(type(None)))
            in str(cm.exception)
        )

        ev = nt_event_batch(
            ex=x + v, callback=lambda _: _, direction=event_direction.negative
        )
        ev = pickle.loads(pickle.dumps(ev))
        self.assertTrue(" non-terminal" in repr(ev))
        self.assertTrue("(x + v)" in repr(ev))
        self.assertTrue("event_direction::negative" in repr(ev))

        # Test dynamic attributes.
        ev.foo = "hello world"
        ev = pickle.loads(pickle.dumps(ev))
        self.assertEqual(ev.foo, "hello world")

        # Test copy semantics.
        class foo:
            pass

        ev.bar = foo()

        self.assertEqual(id(ev.bar), id(copy(ev).bar))
        self.assertNotEqual(id(ev.bar), id(deepcopy(ev).bar))

        # Test to ensure a callback extracted from the event
        # is kept alive and usable when the event is destroyed.
        ev = nt_event_batch(
            ex=x + v, callback=local_cb(), direction=event_direction.negative
        )
        out_cb = ev.callback
        del ev
        gc.collect()
        out_cb(1, 2, 3)
        out_cb(1, 2, 3)
        out_cb(1, 2, 3)
        self.assertEqual(out_cb.n, 3)

        # Terminal event.
        fp_t = float
        ev = t_event_batch(x + v)

        self.assertTrue(" terminal" in repr(ev))
        self.assertTrue("(x + v)" in repr(ev))
        self.assertTrue("event_direction::any" in repr(ev))
        self.assertTrue(": no" in repr(ev))
        self.assertTrue("auto" in repr(ev))
        self.assertEqual(ev.expression, x + v)
        self.assertEqual(ev.direction, event_direction.any)
        self.assertEqual(ev.cooldown, fp_t(-1))
        self.assertTrue(ev.callback is None)

        ev = t_event_batch(x + v, direction=event_direction.negative, cooldown=fp_t(3))

        self.assertTrue(" terminal" in repr(ev))
        self.assertTrue("(x + v)" in repr(ev))
        self.assertTrue("event_direction::negative" in repr(ev))
        self.assertTrue(": no" in repr(ev))
        self.assertTrue("3" in repr(ev))
        self.assertEqual(ev.expression, x + v)
        self.assertEqual(ev.direction, event_direction.negative)
        self.assertEqual(ev.cooldown, fp_t(3))
        self.assertTrue(ev.callback is None)

        ev = t_event_batch(
            x + v,
            direction=event_direction.positive,
            cooldown=fp_t(3),
            callback=lambda _: _,
        )

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

            def __call__(self, ta, d_sgn):
                self.n = self.n + 1

        lcb = local_cb()
        ev = t_event_batch(
            x + v, direction=event_direction.positive, cooldown=fp_t(3), callback=lcb
        )

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
        cb(1, 2)
        cb(1, 2)
        cb(1, 2)
        self.assertEqual(ev.callback.n, 3)
        ev.callback.n = 0
        self.assertEqual(ev.callback.n, 0)
        self.assertNotEqual(id(lcb), id(ev.callback))

        ev = t_event_batch(
            x + v, direction=event_direction.positive, cooldown=fp_t(3), callback=None
        )
        self.assertTrue(ev.callback is None)

        with self.assertRaises(ValueError) as cm:
            t_event_batch(
                x + v,
                direction=event_direction(45),
                cooldown=fp_t(3),
                callback=lambda _: _,
            )
        self.assertTrue("45 is not a valid event_direction" in str(cm.exception))

        with self.assertRaises(TypeError) as cm:
            t_event_batch(x + v, callback=3)
        self.assertTrue(
            "An object of type '{}' cannot be used as an event callback because it is"
            " not callable".format(str(type(3)))
            in str(cm.exception)
        )

        ev = t_event_batch(
            x + v,
            direction=event_direction.positive,
            cooldown=fp_t(3),
            callback=lambda _: _,
        )

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

        # Test copy semantics.
        class foo:
            pass

        ev.bar = foo()

        self.assertEqual(id(ev.bar), id(copy(ev).bar))
        self.assertNotEqual(id(ev.bar), id(deepcopy(ev).bar))

        # Test also with empty callback.
        ev = t_event_batch(x + v, direction=event_direction.positive, cooldown=fp_t(3))

        ev = pickle.loads(pickle.dumps(ev))
        self.assertTrue(" terminal" in repr(ev))
        self.assertTrue("(x + v)" in repr(ev))
        self.assertTrue("event_direction::positive" in repr(ev))
        self.assertTrue(": no" in repr(ev))
        self.assertTrue("3" in repr(ev))

        # Test to ensure a callback extracted from the event
        # is kept alive and usable when the event is destroyed.
        ev = t_event_batch(
            ex=x + v, callback=local_cb(), direction=event_direction.negative
        )
        out_cb = ev.callback
        del ev
        gc.collect()
        out_cb(1, 2)
        out_cb(1, 2)
        out_cb(1, 2)
        self.assertEqual(out_cb.n, 3)

        with self.assertRaises(TypeError) as cm:
            nt_event_batch(x + v, lambda _: _, fp_type=str)
        self.assertTrue(
            'The floating-point type "{}" is not recognized/supported'.format(str)
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            t_event_batch(x + v, fp_type=list)
        self.assertTrue(
            'The floating-point type "{}" is not recognized/supported'.format(list)
            in str(cm.exception)
        )


class event_detection_test_case(unittest.TestCase):
    def test_batch(self):
        x, v = make_vars("x", "v")

        # Use a pendulum for testing purposes.
        sys = [(x, v), (v, -9.8 * sin(x))]

        # Non-terminal events.
        counter = [0] * 2
        cur_time = [0.0] * 2

        # Track the memory address of the integrator object
        # in order to make sure that it is passed correctly
        # into the callback.
        ta_id = None

        def cb0(ta, t, d_sgn, bidx):
            nonlocal counter
            nonlocal cur_time
            nonlocal ta_id

            self.assertTrue(t > cur_time[bidx])
            self.assertTrue(counter[bidx] % 3 == 0 or counter[bidx] % 3 == 2)
            self.assertEqual(ta_id, id(ta))

            counter[bidx] = counter[bidx] + 1
            cur_time[bidx] = t

        def cb1(ta, t, d_sgn, bidx):
            nonlocal counter
            nonlocal cur_time
            nonlocal ta_id

            self.assertTrue(t > cur_time[bidx])
            self.assertTrue(counter[bidx] % 3 == 1)
            self.assertEqual(ta_id, id(ta))

            counter[bidx] = counter[bidx] + 1
            cur_time[bidx] = t

        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0.0, 0.001], [0.25, 0.2501]],
            nt_events=[nt_event_batch(v * v - 1e-10, cb0), nt_event_batch(v, cb1)],
        )

        ta_id = id(ta)

        ta.propagate_until([4.0, 4.0])
        self.assertTrue(
            all(_[0] == taylor_outcome.time_limit for _ in ta.propagate_res)
        )

        self.assertEqual(counter[0], 12)
        self.assertEqual(counter[1], 12)

        # Make sure that when accessing events
        # from the integrator property we always
        # get the same object.
        class cb0:
            def __init__(self):
                self.lst = []

            def __call__(self, ta, t, d_sgn, bidx):
                pass

        class cb1:
            def __init__(self):
                self.lst = []

            def __call__(self, ta, t, d_sgn, bidx):
                pass

        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0.0, 0.001], [0.25, 0.2501]],
            nt_events=[
                nt_event_batch(v * v - 1e-10, cb0()),
                nt_event_batch(v, cb1()),
                nt_event_batch(v, cb1()),
            ],
        )

        # Check that the refcount increases by 3
        # (the number of events).
        rc = getrefcount(ta)
        nt_list = ta.nt_events
        new_rc = getrefcount(ta)
        self.assertEqual(new_rc, rc + 3)

        self.assertEqual(id(ta.nt_events[0].callback), id(ta.nt_events[0].callback))
        self.assertEqual(
            id(ta.nt_events[0].callback.lst), id(ta.nt_events[0].callback.lst)
        )
        self.assertEqual(id(ta.nt_events[1].callback), id(ta.nt_events[1].callback))
        self.assertEqual(
            id(ta.nt_events[1].callback.lst), id(ta.nt_events[1].callback.lst)
        )
        self.assertEqual(id(ta.nt_events[2].callback), id(ta.nt_events[2].callback))
        self.assertEqual(
            id(ta.nt_events[2].callback.lst), id(ta.nt_events[2].callback.lst)
        )

        # Ensure a deep copy of the integrator performs
        # a deep copy of the events.
        ta_copy = deepcopy(ta)

        self.assertNotEqual(
            id(ta_copy.nt_events[0].callback), id(ta.nt_events[0].callback)
        )
        self.assertNotEqual(
            id(ta_copy.nt_events[0].callback.lst), id(ta.nt_events[0].callback.lst)
        )
        self.assertNotEqual(
            id(ta_copy.nt_events[1].callback), id(ta.nt_events[1].callback)
        )
        self.assertNotEqual(
            id(ta_copy.nt_events[1].callback.lst), id(ta.nt_events[1].callback.lst)
        )
        self.assertNotEqual(
            id(ta_copy.nt_events[2].callback), id(ta.nt_events[2].callback)
        )
        self.assertNotEqual(
            id(ta_copy.nt_events[2].callback.lst), id(ta.nt_events[2].callback.lst)
        )

        # Callback with wrong signature.
        def cb2(ta, t):
            pass

        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0.0, 0.001], [0.25, 0.2501]],
            nt_events=[nt_event_batch(v * v - 1e-10, cb2)],
        )

        with self.assertRaises(RuntimeError):
            ta.propagate_until([4.0, 4.0])

        # Terminal events.
        counter_t = [0] * 2
        counter_nt = [0] * 2
        cur_time = [0.0] * 2

        def cb0(ta, t, d_sgn, bidx):
            nonlocal counter_nt
            nonlocal cur_time
            nonlocal ta_id

            self.assertTrue(t > cur_time[bidx])
            self.assertEqual(ta_id, id(ta))

            counter_nt[bidx] = counter_nt[bidx] + 1
            cur_time[bidx] = t

        def cb1(ta, d_sgn, bidx):
            nonlocal cur_time
            nonlocal counter_t
            nonlocal ta_id

            self.assertTrue(ta.time[bidx] > cur_time[bidx])
            self.assertEqual(ta_id, id(ta))

            counter_t[bidx] = counter_t[bidx] + 1
            cur_time[bidx] = ta.time[bidx]

            return True

        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0.0, 0.001], [0.25, 0.2501]],
            nt_events=[nt_event_batch(v * v - 1e-10, cb0)],
            t_events=[t_event_batch(v, callback=cb1)],
        )

        ta_id = id(ta)

        while True:
            ta.step()
            if all(_[0] > taylor_outcome.success for _ in ta.step_res):
                break

        self.assertTrue(all(int(_[0]) == 0 for _ in ta.step_res))
        self.assertTrue(all(_ < 1 for _ in ta.time))
        self.assertTrue(all(_ == 1 for _ in counter_nt))
        self.assertTrue(all(_ == 1 for _ in counter_t))

        while True:
            ta.step()
            if all(_[0] > taylor_outcome.success for _ in ta.step_res):
                break

        self.assertTrue(all(int(_[0]) == 0 for _ in ta.step_res))
        self.assertTrue(all(_ > 1 for _ in ta.time))
        self.assertTrue(all(_ == 3 for _ in counter_nt))
        self.assertTrue(all(_ == 2 for _ in counter_t))

        # Make sure that when accessing events
        # from the integrator property we always
        # get the same object.
        class cb0:
            def __init__(self):
                self.lst = []

            def __call__(self, ta, d_sgn, bidx):
                pass

        class cb1:
            def __init__(self):
                self.lst = []

            def __call__(self, ta, d_sgn, bidx):
                pass

        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0.0, 0.001], [0.25, 0.2501]],
            t_events=[
                t_event_batch(v * v - 1e-10, callback=cb0()),
                t_event_batch(v, callback=cb1()),
                t_event_batch(v, callback=cb1()),
            ],
        )

        # Check that the refcount increases by 3
        # (the number of events).
        rc = getrefcount(ta)
        t_list = ta.t_events
        new_rc = getrefcount(ta)
        self.assertEqual(new_rc, rc + 3)

        self.assertEqual(id(ta.t_events[0].callback), id(ta.t_events[0].callback))
        self.assertEqual(
            id(ta.t_events[0].callback.lst), id(ta.t_events[0].callback.lst)
        )
        self.assertEqual(id(ta.t_events[1].callback), id(ta.t_events[1].callback))
        self.assertEqual(
            id(ta.t_events[1].callback.lst), id(ta.t_events[1].callback.lst)
        )
        self.assertEqual(id(ta.t_events[2].callback), id(ta.t_events[2].callback))
        self.assertEqual(
            id(ta.t_events[2].callback.lst), id(ta.t_events[2].callback.lst)
        )

        # Ensure a deep copy of the integrator performs
        # a deep copy of the events.
        ta_copy = deepcopy(ta)

        self.assertNotEqual(
            id(ta_copy.t_events[0].callback), id(ta.t_events[0].callback)
        )
        self.assertNotEqual(
            id(ta_copy.t_events[0].callback.lst), id(ta.t_events[0].callback.lst)
        )
        self.assertNotEqual(
            id(ta_copy.t_events[1].callback), id(ta.t_events[1].callback)
        )
        self.assertNotEqual(
            id(ta_copy.t_events[1].callback.lst), id(ta.t_events[1].callback.lst)
        )
        self.assertNotEqual(
            id(ta_copy.t_events[2].callback), id(ta.t_events[2].callback)
        )
        self.assertNotEqual(
            id(ta_copy.t_events[2].callback.lst), id(ta.t_events[2].callback.lst)
        )

        # Callback with wrong signature.
        def cb2(ta, t):
            pass

        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0.0, 0.001], [0.25, 0.2501]],
            t_events=[t_event_batch(v * v - 1e-10, callback=cb2)],
        )

        with self.assertRaises(RuntimeError):
            ta.propagate_until([4.0, 4.0])

        # Callback with wrong retval.
        def cb3(ta, d_sgn, bidx):
            return "hello"

        ta = taylor_adaptive_batch(
            sys=sys,
            state=[[0.0, 0.001], [0.25, 0.2501]],
            t_events=[t_event_batch(v * v - 1e-10, callback=cb3)],
        )

        with self.assertRaises(RuntimeError) as cm:
            ta.propagate_until([4.0, 4.0])
        self.assertTrue(
            "in the construction of the return value of an event callback"
            in str(cm.exception)
        )

    def test_gil_bug(self):
        # NOTE: this is a test case for a GIL bug involving the invocation of the destructor of
        # an event callback during the construction of a Taylor integrator.
        #
        # The problem was that in case of exceptions being raised during the construction of an
        # integrator, we would end up calling the destructor of the Pythonic callback without holding
        # the GIL.
        class cb0:
            def __call__(self, ta, t, d_sgn):
                pass

        # Use a pendulum for testing purposes.
        x, v = make_vars("x", "v")
        sys = [(x, v), (v, -9.8 * sin(x))]

        # Perform a throwing construction of an integrator.
        with self.assertRaises(ValueError):
            taylor_adaptive(
                sys=sys,
                # NOTE: state size is wrong here.
                state=[0.0],
                nt_events=[
                    nt_event(x, cb0),
                ],
            )

    def test_scalar(self):
        x, v = make_vars("x", "v")

        if _core._ppc_arch:
            fp_types = [np.float32, float]
        else:
            fp_types = [np.float32, float, np.longdouble]

        if hasattr(_core, "real128"):
            fp_types.append(_core.real128)

        # Use a pendulum for testing purposes.
        sys = [(x, v), (v, -9.8 * sin(x))]

        for fp_t in fp_types:
            # Non-terminal events.
            counter = 0
            cur_time = fp_t(0)

            # Track the memory address of the integrator object
            # in order to make sure that it is passed correctly
            # into the callback.
            ta_id = None

            # NOTE: avoid using very small value for single-precision.
            small_delta = 1e-6 if fp_t == np.float32 else 1e-10

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

            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0), fp_t(0.25)],
                fp_type=fp_t,
                nt_events=[
                    nt_event(v * v - small_delta, cb0, fp_type=fp_t),
                    nt_event(v, cb1, fp_type=fp_t),
                ],
            )

            ta_id = id(ta)

            self.assertEqual(ta.propagate_until(fp_t(4))[0], taylor_outcome.time_limit)

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

            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0), fp_t(0.25)],
                fp_type=fp_t,
                nt_events=[
                    nt_event(v * v - small_delta, cb0(), fp_type=fp_t),
                    nt_event(v, cb1(), fp_type=fp_t),
                    nt_event(v, cb1(), fp_type=fp_t),
                ],
            )

            # Check that the refcount increases by 3
            # (the number of events).
            rc = getrefcount(ta)
            nt_list = ta.nt_events
            new_rc = getrefcount(ta)
            self.assertEqual(new_rc, rc + 3)

            self.assertEqual(id(ta.nt_events[0].callback), id(ta.nt_events[0].callback))
            self.assertEqual(
                id(ta.nt_events[0].callback.lst), id(ta.nt_events[0].callback.lst)
            )
            self.assertEqual(id(ta.nt_events[1].callback), id(ta.nt_events[1].callback))
            self.assertEqual(
                id(ta.nt_events[1].callback.lst), id(ta.nt_events[1].callback.lst)
            )
            self.assertEqual(id(ta.nt_events[2].callback), id(ta.nt_events[2].callback))
            self.assertEqual(
                id(ta.nt_events[2].callback.lst), id(ta.nt_events[2].callback.lst)
            )

            # Ensure a deep copy of the integrator performs
            # a deep copy of the events.
            ta_copy = deepcopy(ta)

            self.assertNotEqual(
                id(ta_copy.nt_events[0].callback), id(ta.nt_events[0].callback)
            )
            self.assertNotEqual(
                id(ta_copy.nt_events[0].callback.lst), id(ta.nt_events[0].callback.lst)
            )
            self.assertNotEqual(
                id(ta_copy.nt_events[1].callback), id(ta.nt_events[1].callback)
            )
            self.assertNotEqual(
                id(ta_copy.nt_events[1].callback.lst), id(ta.nt_events[1].callback.lst)
            )
            self.assertNotEqual(
                id(ta_copy.nt_events[2].callback), id(ta.nt_events[2].callback)
            )
            self.assertNotEqual(
                id(ta_copy.nt_events[2].callback.lst), id(ta.nt_events[2].callback.lst)
            )

            # Callback with wrong signature.
            def cb2(ta, t):
                pass

            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0), fp_t(0.25)],
                fp_type=fp_t,
                nt_events=[nt_event(v * v - small_delta, cb2, fp_type=fp_t)],
            )

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

            def cb1(ta, d_sgn):
                nonlocal cur_time
                nonlocal counter_t
                nonlocal ta_id

                self.assertTrue(ta.time > cur_time)
                self.assertEqual(ta_id, id(ta))

                counter_t = counter_t + 1
                cur_time = ta.time

                return True

            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0), fp_t(0.25)],
                fp_type=fp_t,
                nt_events=[nt_event(v * v - small_delta, cb0, fp_type=fp_t)],
                t_events=[t_event(v, callback=cb1, fp_type=fp_t)],
            )

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

                def __call__(self, ta, d_sgn):
                    pass

            class cb1:
                def __init__(self):
                    self.lst = []

                def __call__(self, ta, d_sgn):
                    pass

            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0), fp_t(0.25)],
                fp_type=fp_t,
                t_events=[
                    t_event(v * v - small_delta, callback=cb0(), fp_type=fp_t),
                    t_event(v, callback=cb1(), fp_type=fp_t),
                    t_event(v, callback=cb1(), fp_type=fp_t),
                ],
            )

            # Check that the refcount increases by 3
            # (the number of events).
            rc = getrefcount(ta)
            t_list = ta.t_events
            new_rc = getrefcount(ta)
            self.assertEqual(new_rc, rc + 3)

            self.assertEqual(id(ta.t_events[0].callback), id(ta.t_events[0].callback))
            self.assertEqual(
                id(ta.t_events[0].callback.lst), id(ta.t_events[0].callback.lst)
            )
            self.assertEqual(id(ta.t_events[1].callback), id(ta.t_events[1].callback))
            self.assertEqual(
                id(ta.t_events[1].callback.lst), id(ta.t_events[1].callback.lst)
            )
            self.assertEqual(id(ta.t_events[2].callback), id(ta.t_events[2].callback))
            self.assertEqual(
                id(ta.t_events[2].callback.lst), id(ta.t_events[2].callback.lst)
            )

            # Ensure a deep copy of the integrator performs
            # a deep copy of the events.
            ta_copy = deepcopy(ta)

            self.assertNotEqual(
                id(ta_copy.t_events[0].callback), id(ta.t_events[0].callback)
            )
            self.assertNotEqual(
                id(ta_copy.t_events[0].callback.lst), id(ta.t_events[0].callback.lst)
            )
            self.assertNotEqual(
                id(ta_copy.t_events[1].callback), id(ta.t_events[1].callback)
            )
            self.assertNotEqual(
                id(ta_copy.t_events[1].callback.lst), id(ta.t_events[1].callback.lst)
            )
            self.assertNotEqual(
                id(ta_copy.t_events[2].callback), id(ta.t_events[2].callback)
            )
            self.assertNotEqual(
                id(ta_copy.t_events[2].callback.lst), id(ta.t_events[2].callback.lst)
            )

            # Callback with wrong signature.
            def cb2(ta, t, tut):
                pass

            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0), fp_t(0.25)],
                fp_type=fp_t,
                t_events=[t_event(v * v - small_delta, callback=cb2, fp_type=fp_t)],
            )

            with self.assertRaises(TypeError):
                ta.propagate_until(fp_t(4))

            # Callback with wrong retval.
            def cb3(ta, d_sgn):
                return "hello"

            ta = taylor_adaptive(
                sys=sys,
                state=[fp_t(0), fp_t(0.25)],
                fp_type=fp_t,
                t_events=[t_event(v * v - 1e-10, callback=cb3, fp_type=fp_t)],
            )

            with self.assertRaises(TypeError) as cm:
                ta.propagate_until(fp_t(4))
            self.assertTrue(
                "in the construction of the return value of an event callback"
                in str(cm.exception)
            )
