# Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest
from .. import recommended_simd_size
import numpy as np


class recommended_simd_size_test_case(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(recommended_simd_size() >= 1)
        self.assertTrue(recommended_simd_size(fp_type=np.float32) >= 1)
        self.assertEqual(recommended_simd_size(), recommended_simd_size(fp_type=float))
        self.assertEqual(
            recommended_simd_size(), recommended_simd_size(fp_type=np.float64)
        )
