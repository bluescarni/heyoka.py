# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


import unittest as _ut


class memcache_test_case(_ut.TestCase):
    def test_basic(self):
        from . import llvm_state

        self.assertTrue(llvm_state.memcache_limit != 0)
        self.assertTrue(llvm_state.memcache_size != 0)
        llvm_state.clear_memcache()
        self.assertTrue(llvm_state.memcache_size == 0)
        llvm_state.memcache_limit = 1024 * 1024 * 1024
        self.assertTrue(llvm_state.memcache_limit == 1024 * 1024 * 1024)
