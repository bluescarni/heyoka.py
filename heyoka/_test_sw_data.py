# Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


import unittest as _ut


class sw_data_test_case(_ut.TestCase):
    def test_basic(self):
        from . import sw_data, sw_data_row
        import numpy as np
        from sys import getrefcount
        import pickle
        from copy import copy, deepcopy

        self.assertTrue(isinstance(sw_data_row, np.dtype))

        # Check access to the data table.
        data = sw_data()
        rc = getrefcount(data)
        tbl = data.table
        self.assertEqual(getrefcount(data), rc + 1)

        # Check that we cannot write to the table.
        with self.assertRaises(ValueError) as cm:
            tbl[:] = tbl[:]
        self.assertTrue("read-only" in str(cm.exception))

        self.assertEqual(tbl.dtype, sw_data_row)

        self.assertGreater(len(data.timestamp), 0)
        self.assertGreater(len(data.identifier), 0)
        self.assertGreater(len(repr(data)), 0)

        # Pickling.
        new_data = pickle.loads(pickle.dumps(data))
        self.assertEqual(new_data.timestamp, data.timestamp)
        self.assertEqual(new_data.identifier, data.identifier)
        self.assertTrue(np.all(new_data.table == tbl))

        # Copy/deepcopy.
        new_data = copy(data)
        self.assertEqual(new_data.timestamp, data.timestamp)
        self.assertEqual(new_data.identifier, data.identifier)
        self.assertTrue(np.all(new_data.table == tbl))

        new_data = deepcopy(data)
        self.assertEqual(new_data.timestamp, data.timestamp)
        self.assertEqual(new_data.identifier, data.identifier)
        self.assertTrue(np.all(new_data.table == tbl))
