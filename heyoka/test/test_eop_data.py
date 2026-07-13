# Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


import unittest
from .. import eop_data, eop_data_row
import numpy as np
from sys import getrefcount
import pickle
from copy import copy, deepcopy


class eop_data_test_case(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(isinstance(eop_data_row, np.dtype))

        # Check access to the data table.
        data = eop_data()
        rc = getrefcount(data)
        tbl = data.table
        self.assertEqual(getrefcount(data), rc + 1)

        # Check that we cannot write to the table.
        with self.assertRaises(ValueError) as cm:
            tbl[:] = tbl[:]
        self.assertTrue("read-only" in str(cm.exception))

        self.assertEqual(tbl.dtype, eop_data_row)

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

    def test_custom_data(self):
        data = eop_data()
        tbl = data.table

        # Round-trip: build a custom dataset from the builtin table.
        # NOTE: we must supply our own identifier, as the builtin one uses a
        # reserved prefix.
        d2 = eop_data(data=tbl, timestamp="20260101", identifier="custom")
        self.assertEqual(d2.timestamp, "20260101")
        self.assertEqual(d2.identifier, "custom")
        self.assertEqual(d2.table.dtype, eop_data_row)
        self.assertTrue(np.all(d2.table == tbl))

        # A non-contiguous input must be accepted (copied into a contiguous buffer).
        nc = tbl[::2]
        self.assertFalse(nc.flags["C_CONTIGUOUS"])
        d3 = eop_data(data=nc, timestamp="ts", identifier="custom")
        self.assertTrue(np.all(d3.table == nc))

        # A contiguous but misaligned input must be accepted (copied into an
        # aligned buffer). We build one by viewing a byte buffer at an odd offset.
        sub = tbl[:50]
        itemsize = eop_data_row.itemsize
        raw = np.zeros(itemsize * len(sub) + 8, dtype=np.uint8)
        mis = raw[1 : 1 + itemsize * len(sub)].view(eop_data_row)
        mis[:] = sub
        self.assertTrue(mis.flags["C_CONTIGUOUS"])
        self.assertFalse(mis.flags["ALIGNED"])
        d4 = eop_data(data=mis, timestamp="ts", identifier="custom")
        self.assertTrue(np.all(d4.table == sub))

        # A multidimensional data array is rejected.
        md = tbl[:6].reshape(2, 3)
        with self.assertRaises(ValueError) as cm:
            eop_data(data=md, timestamp="ts", identifier="custom")
        self.assertTrue("1 dimension" in str(cm.exception))

        # An input array whose dtype does not match the expected one is rejected.
        for bad in (
            np.zeros(5, dtype=np.float64),
            np.zeros(5, dtype=np.dtype([("mjd", "f8"), ("foo", "f8")], align=True)),
        ):
            with self.assertRaises(TypeError) as cm:
                eop_data(data=bad, timestamp="ts", identifier="custom")
            self.assertTrue("dtype" in str(cm.exception))

        # A dtype that is layout-compatible but *unaligned* (numpy alignment 1) is still accepted:
        # numpy dtype equality ignores alignment, so it passes the dtype check, and the ctor re-aligns
        # the buffer (numpy.require against the aligned dtype) before reinterpreting it as C++ data.
        packed = np.dtype(
            {
                "names": list(eop_data_row.names),
                "formats": ["<f8"] * len(eop_data_row.names),
                "offsets": [eop_data_row.fields[n][1] for n in eop_data_row.names],
                "itemsize": eop_data_row.itemsize,
            }
        )
        self.assertEqual(packed, eop_data_row)
        self.assertEqual(packed.alignment, 1)
        raw = np.zeros(packed.itemsize * 50 + 8, dtype=np.uint8)
        misp = raw[1 : 1 + packed.itemsize * 50].view(packed)
        misp[:] = tbl[:50]
        self.assertEqual(misp.ctypes.data % 8, 1)
        d5 = eop_data(data=misp, timestamp="ts", identifier="custom")
        self.assertTrue(np.all(d5.table == tbl[:50]))

        # All-or-nothing: providing only some of the three arguments is an error.
        for kwargs in (
            {"data": tbl},
            {"timestamp": "ts"},
            {"identifier": "custom"},
            {"data": tbl, "timestamp": "ts"},
            {"data": tbl, "identifier": "custom"},
            {"timestamp": "ts", "identifier": "custom"},
        ):
            with self.assertRaises(TypeError) as cm:
                eop_data(**kwargs)
            self.assertTrue("none or all" in str(cm.exception))
