# Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pickle
import unittest
from copy import copy, deepcopy
from sys import getrefcount

from .. import make_vars, sin, taylor_adaptive


class llvm_state_test_case(unittest.TestCase):
    def test_copy(self):
        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]

        ta = taylor_adaptive(sys=sys, state=[0.0, 0.25])

        ls = ta.llvm_state

        class foo:
            pass

        ls.bar = foo()

        self.assertEqual(id(ls.bar), id(copy(ls).bar))
        self.assertNotEqual(id(ls.bar), id(deepcopy(ls).bar))
        self.assertEqual(ls.ir, copy(ls).ir)
        self.assertEqual(ls.ir, deepcopy(ls).ir)

    def test_s11n(self):
        x, v = make_vars("x", "v")

        sys = [(x, v), (v, -9.8 * sin(x))]

        ta = taylor_adaptive(sys=sys, state=[0.0, 0.25])

        # Verify that the reference count of ta
        # is increased when we fetch the llvm_state.
        rc = getrefcount(ta)
        ls = ta.llvm_state
        self.assertEqual(getrefcount(ta), rc + 1)

        self.assertEqual(ls.ir, pickle.loads(pickle.dumps(ls)).ir)

        # Test dynamic attributes.
        ls.foo = "hello world"
        ls = pickle.loads(pickle.dumps(ls))
        self.assertEqual(ls.foo, "hello world")
