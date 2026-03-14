# Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


import unittest as _ut


class diskcache_test_case(_ut.TestCase):
    def test_basic(self):
        import tempfile
        import pathlib
        from . import llvm_state, make_vars, cfunc
        import numpy as np

        # Save original state.
        orig_path = llvm_state.get_diskcache_path()
        orig_enabled = llvm_state.get_diskcache_enabled()

        try:
            with tempfile.TemporaryDirectory(prefix="heyoka_diskcache_test_") as tmp:
                tmp_dir = pathlib.Path(tmp)

                llvm_state.set_diskcache_path(tmp_dir)
                self.assertEqual(llvm_state.get_diskcache_path(), tmp_dir)

                llvm_state.set_diskcache_enabled(True)
                self.assertTrue(llvm_state.get_diskcache_enabled())

                # Fresh cache should be empty.
                self.assertEqual(llvm_state.get_diskcache_size(), 0)

                # Default limit should be positive.
                orig_limit = llvm_state.get_diskcache_limit()
                self.assertGreater(orig_limit, 0)

                # Set and restore limit.
                llvm_state.set_diskcache_limit(1000000)
                self.assertEqual(llvm_state.get_diskcache_limit(), 1000000)
                llvm_state.set_diskcache_limit(orig_limit)

                # Compile something - should populate disk cache.
                llvm_state.clear_memcache()
                x, y = make_vars("x", "y")
                _cf = cfunc([x + y], [x, y])
                self.assertGreater(llvm_state.get_diskcache_size(), 0)
                disk_size = llvm_state.get_diskcache_size()

                # Clear memcache, recompile - should hit disk cache.
                llvm_state.clear_memcache()
                cf2 = cfunc([x + y], [x, y])
                self.assertEqual(llvm_state.get_diskcache_size(), disk_size)

                # Verify correctness.
                out = np.zeros(1)
                cf2(outputs=out, inputs=[1.0, 2.0])
                self.assertEqual(out[0], 3.0)

                # Clear disk cache.
                llvm_state.clear_diskcache()
                self.assertEqual(llvm_state.get_diskcache_size(), 0)

                # Disable disk cache.
                llvm_state.set_diskcache_enabled(False)
                self.assertFalse(llvm_state.get_diskcache_enabled())
                llvm_state.clear_memcache()
                _cf3 = cfunc([x * y], [x, y])
                self.assertEqual(llvm_state.get_diskcache_size(), 0)

                # NOTE: restore the original state before leaving the 'with' block, so that the SQLite connection
                # is closed before TemporaryDirectory cleanup attempts to delete the files (would fail on Windows).
                llvm_state.set_diskcache_path(orig_path)
                llvm_state.set_diskcache_enabled(orig_enabled)

        finally:
            # Safety net: ensure restore happens even if an exception was thrown before the restore above.
            llvm_state.set_diskcache_path(orig_path)
            llvm_state.set_diskcache_enabled(orig_enabled)
