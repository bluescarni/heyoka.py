# Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest
from .. import set_serialization_backend, get_serialization_backend
import cloudpickle as cp
import pickle as pk


class s11n_backend_test_case(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(get_serialization_backend(), cp)
        set_serialization_backend("pickle")
        self.assertEqual(get_serialization_backend(), pk)
        set_serialization_backend("cloudpickle")
        self.assertEqual(get_serialization_backend(), cp)

        with self.assertRaises(TypeError) as cm:
            set_serialization_backend(1)
        self.assertTrue(
            "The serialization backend must be specified as a string"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            set_serialization_backend("pippo")
        self.assertTrue(
            "The serialization backend 'pippo' is not valid. The valid backends are:"
            in str(cm.exception)
        )

        self.assertEqual(get_serialization_backend(), cp)
