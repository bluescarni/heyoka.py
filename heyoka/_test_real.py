# Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class real_test_case(_ut.TestCase):
    def runTest(self):
        from . import core

        if not hasattr(core, "real"):
            return

        self.test_ctor()
        self.test_unary()
        self.test_binary()
        self.test_conversions()
        self.test_comparisons()
        self.test_numpy_basic()
        self.test_numpy_add()
        self.test_numpy_sub()
        self.test_numpy_mul()
        self.test_numpy_div()
        self.test_numpy_square()

    def test_numpy_square(self):
        from . import real
        from . import core
        import numpy as np

        make_no = core._make_no_real_array

        # Basic.
        arr1 = np.array([1, 2, 3], dtype=real)
        arr2 = np.square(arr1)
        self.assertEqual(arr2[0], real(1) * real(1))
        self.assertEqual(arr2[1], real(2) * real(2))
        self.assertEqual(arr2[2], real(3) * real(3))

        # Holes in the input args.
        arr1 = np.zeros((3,), dtype=real)
        arr1[0] = real(2)
        arr2 = np.square(arr1)
        self.assertEqual(arr2[0], real(4))
        self.assertEqual(arr2[1], real(0))
        self.assertEqual(arr2[2], real(0))

        # Non-contiguous input.
        arr1 = np.array([1, 0, 2, 0, 3, 0], dtype=real)
        arr2 = np.square(arr1[::2])
        self.assertEqual(arr2[0], real(1) * real(1))
        self.assertEqual(arr2[1], real(2) * real(2))
        self.assertEqual(arr2[2], real(3) * real(3))

        # With provided output.
        arr1 = np.array([1, 2, 3], dtype=real)
        arr2 = np.zeros((3,), dtype=real)
        arr2[0] = real("1.1", 128)
        np.square(arr1, out=arr2)
        self.assertEqual(arr2[0], real(1) * real(1))
        self.assertEqual(arr2[0].prec, real(1).prec)
        self.assertEqual(arr2[1], real(2) * real(2))
        self.assertEqual(arr2[2], real(3) * real(3))

        # Test with non-owning.
        arr2 = np.zeros((3,), dtype=real)
        arr2[0] = real("1.1", 128)
        arr2 = make_no(arr2)
        np.square(make_no(arr1), out=arr2)
        self.assertEqual(arr2[0], real(1) * real(1))
        self.assertEqual(arr2[0].prec, real(1).prec)
        self.assertEqual(arr2[1], real(2) * real(2))
        self.assertEqual(arr2[2], real(3) * real(3))

        # Test with overlapping arguments.
        arr2 = np.array([1, 2, 3], dtype=real)
        np.square(arr2, out=arr2)
        self.assertEqual(arr2[0], real(1) * real(1))
        self.assertEqual(arr2[1], real(2) * real(2))
        self.assertEqual(arr2[2], real(3) * real(3))

    def test_numpy_div(self):
        from . import real
        from . import core
        import numpy as np

        make_no = core._make_no_real_array

        # Basic.
        arr1 = np.array([1, 2, 3], dtype=real)
        arr2 = np.array([4, 5, 6], dtype=real)
        arr3 = arr1 / arr2
        self.assertEqual(arr3[0], real(1) / real(4))
        self.assertEqual(arr3[1], real(2) / real(5))
        self.assertEqual(arr3[2], real(3) / real(6))

        # Holes in the input args.
        arr1 = np.zeros((3,), dtype=real)
        arr1[0] = real(1)
        arr1[1] = 1
        arr2 = np.zeros((3,), dtype=real)
        arr2[0] = 1
        arr2[1] = real(5)
        arr2[2] = real(1)
        arr3 = arr1 / arr2
        self.assertEqual(arr3[0], real(1))
        self.assertEqual(arr3[1], real(1) / real(5))
        self.assertEqual(arr3[2], real(0))

        # Non-contiguous inputs.
        arr1 = np.array([1, 0, 2, 0, 3, 0], dtype=real)
        arr2 = np.array([4, 5, 6], dtype=real)
        arr3 = arr1[::2] / arr2
        self.assertEqual(arr3[0], real(1) / real(4))
        self.assertEqual(arr3[1], real(2) / real(5))
        self.assertEqual(arr3[2], real(3) / real(6))

        # With provided output.
        arr1 = np.array([1, 2, 3], dtype=real)
        arr2 = np.array([4, 5, 6], dtype=real)
        arr3 = np.zeros((3,), dtype=real)
        arr3[0] = real("1.1", 128)
        np.divide(arr1, arr2, out=arr3)
        self.assertEqual(arr3[0], real(1) / real(4))
        self.assertEqual(arr3[0].prec, real(1).prec)
        self.assertEqual(arr3[1], real(2) / real(5))
        self.assertEqual(arr3[2], real(3) / real(6))

        # Test with non-owning.
        arr3 = np.zeros((3,), dtype=real)
        arr3[0] = real("1.1", 128)
        arr3 = make_no(arr3)
        np.divide(make_no(arr1), make_no(arr2), out=arr3)
        self.assertEqual(arr3[0], real(1) / real(4))
        self.assertEqual(arr3[0].prec, real(1).prec)
        self.assertEqual(arr3[1], real(2) / real(5))
        self.assertEqual(arr3[2], real(3) / real(6))

        # Test with overlapping arguments.
        arr3 = np.array([1, 2, 3], dtype=real)
        np.divide(arr3, arr3, out=arr3)
        self.assertEqual(arr3[0], 1)
        self.assertEqual(arr3[1], 1)
        self.assertEqual(arr3[2], 1)

    def test_numpy_mul(self):
        from . import real
        from . import core
        import numpy as np

        make_no = core._make_no_real_array

        # Basic.
        arr1 = np.array([1, 2, 3], dtype=real)
        arr2 = np.array([4, 5, 6], dtype=real)
        arr3 = arr1 * arr2
        self.assertEqual(arr3[0], real(1) * real(4))
        self.assertEqual(arr3[1], real(2) * real(5))
        self.assertEqual(arr3[2], real(3) * real(6))

        # Holes in the input args.
        arr1 = np.zeros((3,), dtype=real)
        arr1[0] = real(1)
        arr1[1] = 1
        arr2 = np.zeros((3,), dtype=real)
        arr2[0] = 1
        arr2[1] = real(5)
        arr3 = arr1 * arr2
        self.assertEqual(arr3[0], real(1))
        self.assertEqual(arr3[1], real(5))
        self.assertEqual(arr3[2], real(0))

        # Non-contiguous inputs.
        arr1 = np.array([1, 0, 2, 0, 3, 0], dtype=real)
        arr2 = np.array([4, 5, 6], dtype=real)
        arr3 = arr1[::2] * arr2
        self.assertEqual(arr3[0], real(1) * real(4))
        self.assertEqual(arr3[1], real(2) * real(5))
        self.assertEqual(arr3[2], real(3) * real(6))

        # With provided output.
        arr1 = np.array([1, 2, 3], dtype=real)
        arr2 = np.array([4, 5, 6], dtype=real)
        arr3 = np.zeros((3,), dtype=real)
        arr3[0] = real("1.1", 128)
        np.multiply(arr1, arr2, out=arr3)
        self.assertEqual(arr3[0], real(1) * real(4))
        self.assertEqual(arr3[0].prec, real(1).prec)
        self.assertEqual(arr3[1], real(2) * real(5))
        self.assertEqual(arr3[2], real(3) * real(6))

        # Test with non-owning.
        arr3 = np.zeros((3,), dtype=real)
        arr3[0] = real("1.1", 128)
        arr3 = make_no(arr3)
        np.multiply(make_no(arr1), make_no(arr2), out=arr3)
        self.assertEqual(arr3[0], real(1) * real(4))
        self.assertEqual(arr3[0].prec, real(1).prec)
        self.assertEqual(arr3[1], real(2) * real(5))
        self.assertEqual(arr3[2], real(3) * real(6))

        # Test with overlapping arguments.
        arr3 = np.zeros((3,), dtype=real)
        np.multiply(arr3, arr3, out=arr3)
        self.assertEqual(arr3[0], 0)
        self.assertEqual(arr3[1], 0)
        self.assertEqual(arr3[2], 0)

    def test_numpy_add(self):
        from . import real
        from . import core
        import numpy as np

        make_no = core._make_no_real_array

        # Basic.
        arr1 = np.array([1, 2, 3], dtype=real)
        arr2 = np.array([4, 5, 6], dtype=real)
        arr3 = arr1 + arr2
        self.assertEqual(arr3[0], real(1) + real(4))
        self.assertEqual(arr3[1], real(2) + real(5))
        self.assertEqual(arr3[2], real(3) + real(6))

        # Holes in the input args.
        arr1 = np.zeros((3,), dtype=real)
        arr1[0] = real(1)
        arr2 = np.zeros((3,), dtype=real)
        arr2[1] = real(5)
        arr3 = arr1 + arr2
        self.assertEqual(arr3[0], real(1))
        self.assertEqual(arr3[1], real(5))
        self.assertEqual(arr3[2], real(0))

        # Non-contiguous inputs.
        arr1 = np.array([1, 0, 2, 0, 3, 0], dtype=real)
        arr2 = np.array([4, 5, 6], dtype=real)
        arr3 = arr1[::2] + arr2
        self.assertEqual(arr3[0], real(1) + real(4))
        self.assertEqual(arr3[1], real(2) + real(5))
        self.assertEqual(arr3[2], real(3) + real(6))

        # With provided output.
        arr1 = np.array([1, 2, 3], dtype=real)
        arr2 = np.array([4, 5, 6], dtype=real)
        arr3 = np.zeros((3,), dtype=real)
        arr3[0] = real("1.1", 128)
        np.add(arr1, arr2, out=arr3)
        self.assertEqual(arr3[0], real(1) + real(4))
        self.assertEqual(arr3[0].prec, real(1).prec)
        self.assertEqual(arr3[1], real(2) + real(5))
        self.assertEqual(arr3[2], real(3) + real(6))

        # Test with non-owning.
        arr3 = np.zeros((3,), dtype=real)
        arr3[0] = real("1.1", 128)
        arr3 = make_no(arr3)
        np.add(make_no(arr1), make_no(arr2), out=arr3)
        self.assertEqual(arr3[0], real(1) + real(4))
        self.assertEqual(arr3[0].prec, real(1).prec)
        self.assertEqual(arr3[1], real(2) + real(5))
        self.assertEqual(arr3[2], real(3) + real(6))

        # Test with overlapping arguments.
        arr3 = np.zeros((3,), dtype=real)
        np.add(arr3, arr3, out=arr3)
        self.assertEqual(arr3[0], 0)
        self.assertEqual(arr3[1], 0)
        self.assertEqual(arr3[2], 0)

    def test_numpy_sub(self):
        from . import real
        from . import core
        import numpy as np

        make_no = core._make_no_real_array

        # Basic.
        arr1 = np.array([1, 2, 3], dtype=real)
        arr2 = np.array([4, 5, 6], dtype=real)
        arr3 = arr1 - arr2
        self.assertEqual(arr3[0], real(1) - real(4))
        self.assertEqual(arr3[1], real(2) - real(5))
        self.assertEqual(arr3[2], real(3) - real(6))

        # Holes in the input args.
        arr1 = np.zeros((3,), dtype=real)
        arr1[0] = real(1)
        arr2 = np.zeros((3,), dtype=real)
        arr2[1] = real(5)
        arr3 = arr1 - arr2
        self.assertEqual(arr3[0], real(1))
        self.assertEqual(arr3[1], real(-5))
        self.assertEqual(arr3[2], real(0))

        # Non-contiguous inputs.
        arr1 = np.array([1, 0, 2, 0, 3, 0], dtype=real)
        arr2 = np.array([4, 5, 6], dtype=real)
        arr3 = arr1[::2] - arr2
        self.assertEqual(arr3[0], real(1) - real(4))
        self.assertEqual(arr3[1], real(2) - real(5))
        self.assertEqual(arr3[2], real(3) - real(6))

        # With provided output.
        arr1 = np.array([1, 2, 3], dtype=real)
        arr2 = np.array([4, 5, 6], dtype=real)
        arr3 = np.zeros((3,), dtype=real)
        arr3[0] = real("1.1", 128)
        np.subtract(arr1, arr2, out=arr3)
        self.assertEqual(arr3[0], real(1) - real(4))
        self.assertEqual(arr3[0].prec, real(1).prec)
        self.assertEqual(arr3[1], real(2) - real(5))
        self.assertEqual(arr3[2], real(3) - real(6))

        # Test with non-owning.
        arr3 = np.zeros((3,), dtype=real)
        arr3[0] = real("1.1", 128)
        arr3 = make_no(arr3)
        np.subtract(make_no(arr1), make_no(arr2), out=arr3)
        self.assertEqual(arr3[0], real(1) - real(4))
        self.assertEqual(arr3[0].prec, real(1).prec)
        self.assertEqual(arr3[1], real(2) - real(5))
        self.assertEqual(arr3[2], real(3) - real(6))

        # Test with overlapping arguments.
        arr3 = np.zeros((3,), dtype=real)
        np.subtract(arr3, arr3, out=arr3)
        self.assertEqual(arr3[0], 0)
        self.assertEqual(arr3[1], 0)
        self.assertEqual(arr3[2], 0)

    def test_numpy_basic(self):
        from . import real
        from . import core
        import numpy as np

        ld = np.longdouble

        make_no = core._make_no_real_array

        arr = np.empty((5,), dtype=real)
        for val in arr:
            self.assertEqual(val, 0)
            self.assertEqual(val.prec, real().prec)

        self.assertFalse(make_no(arr).flags.owndata)

        for val in make_no(arr):
            self.assertEqual(val, 0)
            self.assertEqual(val.prec, real().prec)

        # NOTE: zeros seems to just zero out the memory,
        # so that it should be equivalent to empty() for real.
        arr = np.zeros((5,), dtype=real)
        for val in arr:
            self.assertEqual(val, 0)
            self.assertEqual(val.prec, real().prec)

        for val in make_no(arr):
            self.assertEqual(val, 0)
            self.assertEqual(val.prec, real().prec)

        arr[3] = real("1.1", 128)
        self.assertEqual(arr[3], real("1.1", 128))
        self.assertEqual(arr[3].prec, 128)

        no_arr = make_no(arr)
        no_arr[3] = real("1.1", 128)
        self.assertEqual(no_arr[3], real("1.1", 128))
        self.assertEqual(no_arr[3].prec, 128)

        # Setitem witn non-real types.
        arr[1] = 1
        self.assertEqual(arr[1], real(1))
        self.assertEqual(arr[1].prec, real(1).prec)
        arr[1] = 1.0
        self.assertEqual(arr[1], real(1.0))
        self.assertEqual(arr[1].prec, real(1.0).prec)

        if hasattr(core, "real128"):
            real128 = core.real128
            arr[1] = real128(1.0)
            self.assertEqual(arr[1], real(real128(1.0)))
            self.assertEqual(arr[1].prec, real(real128(1.0)).prec)

        with self.assertRaises(TypeError) as cm:
            arr[1] = []
        self.assertTrue(
            "Cannot invoke __setitem__() on a real array with an input value of type"
            in str(cm.exception)
        )

        # NOTE: this invokes the copyswap primitive.
        arr2 = np.copy(arr)

        for _ in range(5):
            self.assertEqual(arr[_], arr2[_])
            self.assertEqual(arr[_].prec, arr2[_].prec)

        no_arr2 = np.copy(no_arr)
        for _ in range(5):
            self.assertEqual(no_arr[_], no_arr2[_])
            self.assertEqual(no_arr[_].prec, no_arr2[_].prec)

        arr2[3] = real()
        self.assertNotEqual(arr2[3], arr[3])

        no_arr2[3] = real()
        self.assertNotEqual(no_arr2[3], no_arr[3])

        # Test that byteswapping is forbidden.
        with self.assertRaises(SystemError) as cm:
            arr2.byteswap()

        with self.assertRaises(SystemError) as cm:
            no_arr2.byteswap()

        # Nonzero.
        arr = np.zeros((5,), dtype=real)
        nz = arr.nonzero()
        self.assertEqual(len(nz[0]), 0)
        arr[3] = real(1.1)
        nz = arr.nonzero()
        self.assertTrue(np.all(nz[0] == np.array([3])))
        arr[0] = real(float("nan"))
        nz = arr.nonzero()
        self.assertTrue(np.all(nz[0] == np.array([0, 3])))

        no_arr = make_no(np.zeros((5,), dtype=real))
        nz = no_arr.nonzero()
        self.assertEqual(len(nz[0]), 0)
        no_arr[3] = real(1.1)
        nz = no_arr.nonzero()
        self.assertTrue(np.all(nz[0] == np.array([3])))
        no_arr[0] = real(float("nan"))
        nz = no_arr.nonzero()
        self.assertTrue(np.all(nz[0] == np.array([0, 3])))

        # Sorting (via the compare primitive).
        arr1 = np.array([real(3), real(1), real(2)])
        arr1_sorted = np.sort(arr1)
        self.assertEqual(arr1_sorted[0], 1)
        self.assertEqual(arr1_sorted[1], 2)
        self.assertEqual(arr1_sorted[2], 3)

        arr1 = make_no(np.array([real(3), real(1), real(2)]))
        arr1_sorted = make_no(np.sort(arr1))
        self.assertEqual(arr1_sorted[0], 1)
        self.assertEqual(arr1_sorted[1], 2)
        self.assertEqual(arr1_sorted[2], 3)

        # TODO implement once we have isnan()
        # Check NaN handling.
        # arr1 = np.array([real128("nan"), real128("1"), real128("nan")])
        # self.assertTrue(all(np.isnan(arr1) == [True, False, True]))
        # arr1_sorted = np.sort(arr1)
        # self.assertEqual(arr1_sorted[0], 1)
        # self.assertFalse(np.isnan(arr1_sorted[0]))
        # self.assertTrue(np.isnan(arr1_sorted[1]))
        # self.assertTrue(np.isnan(arr1_sorted[2]))

        # Argmin/argmax.
        arr = np.array(
            [real(1), real(321), real(54), real(6), real(2), real(6), real(-6)],
            dtype=real,
        )
        self.assertEqual(np.argmin(arr), 6)
        self.assertEqual(np.argmax(arr), 1)
        arr = np.array([], dtype=real)
        with self.assertRaises(ValueError) as cm:
            np.argmin(arr)
        with self.assertRaises(ValueError) as cm:
            np.argmax(arr)

        arr = make_no(
            np.array(
                [real(1), real(321), real(54), real(6), real(2), real(6), real(-6)],
                dtype=real,
            )
        )
        self.assertEqual(np.argmin(arr), 6)
        self.assertEqual(np.argmax(arr), 1)
        arr = make_no(np.array([], dtype=real))
        with self.assertRaises(ValueError) as cm:
            np.argmin(arr)
        with self.assertRaises(ValueError) as cm:
            np.argmax(arr)

        # Try array with uninited values.
        arr = np.empty((7,), dtype=real)
        arr[1] = real(321)
        arr[-1] = real(-6)
        self.assertEqual(np.argmin(arr), 6)
        self.assertEqual(np.argmax(arr), 1)

        arr = np.empty((7,), dtype=real)
        self.assertEqual(np.argmin(arr), 0)
        self.assertEqual(np.argmax(arr), 0)

        # arange().
        # TODO remove the explicit types for 0 and 1, check the dtype
        # after we implement conversions. Check also that this returns
        # an array with dtype real (currently it's object).
        arr = np.arange(0, 1, real("0.3", 128))
        # self.assertTrue(
        #     np.all(
        #         arr
        #         == np.array(
        #             [
        #                 0,
        #                 real("0.29999999999999999999999999999999999"),
        #                 real("0.599999999999999999999999999999999981"),
        #                 real("0.899999999999999999999999999999999923"),
        #             ],
        #             dtype=real,
        #         )
        #     )
        # )

        # full().
        arr = np.full((5,), 1.1, dtype=real)
        for val in arr:
            self.assertEqual(val, 1.1)
            self.assertEqual(val.prec, real(1.1).prec)

        arr = np.full((5,), real("1.1", 128), dtype=real)
        for val in arr:
            self.assertEqual(val, real("1.1", 128))
            self.assertEqual(val.prec, 128)

        # fill().
        arr = np.empty((5,), dtype=real)
        arr.fill(1.1)
        for val in arr:
            self.assertEqual(val, 1.1)
            self.assertEqual(val.prec, real(1.1).prec)

        arr = np.empty((5,), dtype=real)
        arr.fill(real("1.1", 128))
        for val in arr:
            self.assertEqual(val, real("1.1", 128))
            self.assertEqual(val.prec, 128)

        arr = np.empty((5,), dtype=real)[::2]
        arr.fill(real("1.1", 128))
        for val in arr:
            self.assertEqual(val, real("1.1", 128))
            self.assertEqual(val.prec, 128)

        arr = make_no(np.array([1, 2, 3, 4, 5], dtype=real))
        arr.fill(real("1.1", 128))
        for val in arr:
            self.assertEqual(val, real("1.1", 128))
            self.assertEqual(val.prec, 128)

        # linspace().
        arr = np.linspace(0, 1, 10, dtype=real)
        self.assertEqual(arr.dtype, real)
        self.assertEqual(arr[0].prec, real(1.0).prec)

        arr = np.linspace(real(0), real(1), 10, dtype=real)
        self.assertEqual(arr.dtype, real)
        self.assertEqual(arr[0].prec, real(1).prec)

        arr = np.linspace(real(0, 128), real(1, 128), 10, dtype=real)
        self.assertEqual(arr.dtype, real)
        self.assertEqual(arr[0].prec, 128)

    def test_comparisons(self):
        from . import real
        from . import core
        import numpy as np

        ld = np.longdouble

        # Equality
        self.assertEqual(real(2), real(2))

        self.assertEqual(real(2), 2)
        self.assertEqual(2, real(2))

        self.assertEqual(real(2), 2.0)
        self.assertEqual(2.0, real(2))

        self.assertEqual(real(2), ld(2.0))
        self.assertEqual(ld(2.0), real(2))

        if hasattr(core, "real128"):
            real128 = core.real128

            self.assertEqual(real(2), real128(2))
            self.assertEqual(real128(2), real(2))

        # Inquality
        self.assertNotEqual(real(3), real(2))

        self.assertNotEqual(real(3), 2)
        self.assertNotEqual(3, real(2))

        self.assertNotEqual(real(3), 2.0)
        self.assertNotEqual(3.0, real(2))

        self.assertNotEqual(real(3), ld(2.0))
        self.assertNotEqual(ld(3.0), real(2))

        if hasattr(core, "real128"):
            real128 = core.real128

            self.assertNotEqual(real(3), real128(2))
            self.assertNotEqual(real128(2), real(3))

        # Less than.
        self.assertLess(real(1), real(2))

        self.assertLess(real(1), 2)
        self.assertLess(1, real(2))

        self.assertLess(real(1), 2.0)
        self.assertLess(1.0, real(2))

        self.assertLess(real(1), ld(2.0))
        self.assertLess(ld(1.0), real(2))

        if hasattr(core, "real128"):
            real128 = core.real128

            self.assertLess(real(1), real128(2))
            self.assertLess(real128(1), real(3))

        # Less than/equal.
        self.assertLessEqual(real(1), real(2))

        self.assertLessEqual(real(2), 2)
        self.assertLessEqual(1, real(2))

        self.assertLessEqual(real(1), 2.0)
        self.assertLessEqual(2.0, real(2))

        self.assertLessEqual(real(1), ld(2.0))
        self.assertLessEqual(ld(2.0), real(2))

        if hasattr(core, "real128"):
            real128 = core.real128

            self.assertLessEqual(real(1), real128(2))
            self.assertLessEqual(real128(3), real(3))

        # Greater than.
        self.assertGreater(real(3), real(2))

        self.assertGreater(real(3), 2)
        self.assertGreater(3, real(2))

        self.assertGreater(real(3), 2.0)
        self.assertGreater(3.0, real(2))

        self.assertGreater(real(3), ld(2.0))
        self.assertGreater(ld(3.0), real(2))

        if hasattr(core, "real128"):
            real128 = core.real128

            self.assertGreater(real(3), real128(2))
            self.assertGreater(real128(3), real(2))

        # Greater than/equal.
        self.assertGreaterEqual(real(3), real(2))

        self.assertGreaterEqual(real(3), 3)
        self.assertGreaterEqual(3, real(2))

        self.assertGreaterEqual(real(3), 3.0)
        self.assertGreaterEqual(3.0, real(2))

        self.assertGreaterEqual(real(3), ld(2.0))
        self.assertGreaterEqual(ld(3.0), real(3))

        if hasattr(core, "real128"):
            real128 = core.real128

            self.assertGreaterEqual(real(3), real128(2))
            self.assertGreaterEqual(real128(3), real(3))

    def test_conversions(self):
        from . import real

        self.assertTrue(bool(real(1)))
        self.assertFalse(bool(real(0)))
        self.assertTrue(bool(real("nan", 32)))

        self.assertEqual(1.1, float(real(1.1)))

        self.assertEqual(int(real("1.23", 64)), 1)
        self.assertEqual(int(real("-1.23", 64)), -1)
        self.assertEqual(
            int(real("1.1e100", prec=233)),
            11000000000000000000000000000000000000000000000000000000000000000000000633825300114114700748351602688,
        )

        with self.assertRaises(ValueError) as cm:
            int(real("nan", 32))
        self.assertTrue("Cannot convert real NaN to integer" in str(cm.exception))

        with self.assertRaises(OverflowError) as cm:
            int(real("inf", 32))
        self.assertTrue("Cannot convert real infinity to integer" in str(cm.exception))

    def test_binary(self):
        from . import real
        from . import core
        import numpy as np

        ld = np.longdouble

        # Plus.
        x = real(1, 128)
        y = real(-2, 128)
        self.assertEqual(str(x + y), str(real(-1, 128)))

        self.assertEqual(str(x + -2), str(real(-1, 128)))
        self.assertEqual(str(1 + y), str(real(-1, 128)))

        self.assertEqual(str(x + -2.0), str(real(-1, 128)))
        self.assertEqual(str(1.0 + y), str(real(-1, 128)))

        self.assertEqual(str(x + ld(-2.0)), str(real(-1, 128)))
        self.assertEqual(str(ld(1.0) + y), str(real(-1, 128)))

        # real128.
        if hasattr(core, "real128"):
            real128 = core.real128

            self.assertEqual(str(x + real128(-2.0)), str(real(-1, 128)))
            self.assertEqual(str(real128(1.0) + y), str(real(-1, 128)))

        with self.assertRaises(TypeError) as cm:
            x + []
        with self.assertRaises(TypeError) as cm:
            [] + x

        # Minus.
        x = real(1, 128)
        y = real(-2, 128)
        self.assertEqual(str(x - y), str(real(3, 128)))

        self.assertEqual(str(x - -2), str(real(3, 128)))
        self.assertEqual(str(1 - y), str(real(3, 128)))

        self.assertEqual(str(x - -2.0), str(real(3, 128)))
        self.assertEqual(str(1.0 - y), str(real(3, 128)))

        self.assertEqual(str(x - ld(-2.0)), str(real(3, 128)))
        self.assertEqual(str(ld(1.0) - y), str(real(3, 128)))

        # real128.
        if hasattr(core, "real128"):
            real128 = core.real128

            self.assertEqual(str(x - real128(-2.0)), str(real(3, 128)))
            self.assertEqual(str(real128(1.0) - y), str(real(3, 128)))

        with self.assertRaises(TypeError) as cm:
            x - []
        with self.assertRaises(TypeError) as cm:
            [] - x

        # Times.
        x = real(1, 128)
        y = real(-2, 128)
        self.assertEqual(str(x * y), str(real(-2, 128)))

        self.assertEqual(str(x * -2), str(real(-2, 128)))
        self.assertEqual(str(1 * y), str(real(-2, 128)))

        self.assertEqual(str(x * -2.0), str(real(-2, 128)))
        self.assertEqual(str(1.0 * y), str(real(-2, 128)))

        self.assertEqual(str(x * ld(-2.0)), str(real(-2, 128)))
        self.assertEqual(str(ld(1.0) * y), str(real(-2, 128)))

        # real128.
        if hasattr(core, "real128"):
            real128 = core.real128

            self.assertEqual(str(x * real128(-2.0)), str(real(-2, 128)))
            self.assertEqual(str(real128(1.0) * y), str(real(-2, 128)))

        with self.assertRaises(TypeError) as cm:
            x * []
        with self.assertRaises(TypeError) as cm:
            [] * x

        # Div.
        x = real(1, 128)
        y = real(-2, 128)
        self.assertEqual(str(x / y), str(real(-0.5, 128)))

        self.assertEqual(str(x / -2), str(real(-0.5, 128)))
        self.assertEqual(str(1 / y), str(real(-0.5, 128)))

        self.assertEqual(str(x / -2.0), str(real(-0.5, 128)))
        self.assertEqual(str(1.0 / y), str(real(-0.5, 128)))

        self.assertEqual(str(x / ld(-2.0)), str(real(-0.5, 128)))
        self.assertEqual(str(ld(1.0) / y), str(real(-0.5, 128)))

        # real128.
        if hasattr(core, "real128"):
            real128 = core.real128

            self.assertEqual(str(x / real128(-2.0)), str(real(-0.5, 128)))
            self.assertEqual(str(real128(1.0) / y), str(real(-0.5, 128)))

        with self.assertRaises(TypeError) as cm:
            x / []
        with self.assertRaises(TypeError) as cm:
            [] / x

        # Floor divide.
        self.assertEqual(str(real(2.1) // real(1.1)), str(real(1.0)))

        self.assertEqual(str(real(2.1, 128) // 1), str(real(2.0, 128)))
        self.assertEqual(str(1 // real(2.1, 128)), str(real(0.0, 128)))

        self.assertEqual(str(real(2.1, 128) // 1.0), str(real(2.0, 128)))
        self.assertEqual(str(1.0 // real(2.1, 128)), str(real(0.0, 128)))

        self.assertEqual(str(real(2.1, 128) // ld(1.0)), str(real(2.0, 128)))
        self.assertEqual(str(ld(1.0) // real(2.1, 128)), str(real(0.0, 128)))

        # real128.
        if hasattr(core, "real128"):
            real128 = core.real128

            self.assertEqual(str(real(2.1, 128) // real128(1)), str(real(2.0, 128)))
            self.assertEqual(str(real128(1) // real(2.1, 128)), str(real(0.0, 128)))

        with self.assertRaises(TypeError) as cm:
            x // []
        with self.assertRaises(TypeError) as cm:
            [] // x

        # pow.
        self.assertEqual(str(real(2.0) ** real(3.0)), str(real(8.0)))

        self.assertEqual(str(real(2.0, 128) ** 3), str(real(8.0, 128)))
        self.assertEqual(str(2 ** real(3.0, 128)), str(real(8.0, 128)))

        self.assertEqual(str(real(2.0, 128) ** 3.0), str(real(8.0, 128)))
        self.assertEqual(str(2.0 ** real(3.0, 128)), str(real(8.0, 128)))

        self.assertEqual(str(real(2.0, 128) ** ld(3.0)), str(real(8.0, 128)))
        self.assertEqual(str(ld(2.0) ** real(3.0, 128)), str(real(8.0, 128)))

        # real128.
        if hasattr(core, "real128"):
            real128 = core.real128

            self.assertEqual(str(real(2.0, 128) ** real128(3.0)), str(real(8.0, 128)))
            self.assertEqual(str(real128(2.0) ** real(3.0, 128)), str(real(8.0, 128)))

        with self.assertRaises(TypeError) as cm:
            x ** []
        with self.assertRaises(TypeError) as cm:
            [] ** x

        with self.assertRaises(ValueError) as cm:
            pow(real(2.0), real(3.0), mod=real(1.0))
        self.assertTrue(
            "Modular exponentiation is not supported for real" in str(cm.exception)
        )

    def test_unary(self):
        from . import real

        x = real("1.1", 512)
        self.assertEqual(str(x), str(+x))

        xm = real("-1.1", 512)
        self.assertEqual(str(xm), str(-x))
        self.assertEqual(str(abs(xm)), str(x))

    def test_ctor(self):
        from . import real
        from . import core
        import numpy as np

        ld = np.longdouble

        # Default ctor.
        self.assertEqual(str(real()), "0.0")
        self.assertEqual(real().prec, 2)

        # Check error if only the precision argument
        # is present.
        with self.assertRaises(ValueError) as cm:
            real(prec=32)
        self.assertTrue(
            "Cannot construct a real from a precision only - a value must also be supplied"
            in str(cm.exception)
        )

        # Check that the precision argument must be of the correct type.
        with self.assertRaises(TypeError) as cm:
            real(1, prec=[])

        # Constructor from float.
        x = real(1.1)
        self.assertEqual(str(x), "1.1000000000000001")
        self.assertEqual(x.prec, np.finfo(float).nmant + 1)
        x = real(-1.1)
        self.assertEqual(str(x), "-1.1000000000000001")

        x = real(1.1, 23)
        self.assertEqual(str(x), "1.0999999")
        self.assertEqual(x.prec, 23)
        x = real(-1.1, 23)
        self.assertEqual(str(x), "-1.0999999")

        with self.assertRaises(ValueError) as cm:
            real(1.1, prec=-1)
        self.assertTrue(
            "Cannot set the precision of a real to the value -1" in str(cm.exception)
        )

        # Constructor from int.

        # Small int, no precision.
        x = real(1)
        self.assertTrue("1.000" in str(x))

        # Larger int, no precision.
        x = real(-(1 << 50))
        self.assertTrue("125899906842624" in str(x))

        # Int that does not fit in long/long long, no precision.
        x = real(-(1 << 128))
        self.assertTrue("40282366920938463463374607431768211456" in str(x))

        # Small int, explicit precision.
        x = real(1, 12)
        self.assertEqual("1.0000", str(x))
        self.assertEqual(x.prec, 12)

        x = real(-1, 23)
        self.assertEqual("-1.0000000", str(x))
        self.assertEqual(x.prec, 23)

        # Large int, explicit precision, non-exact.
        x = real(1 << 345, 47)
        self.assertEqual("7.167183174968973e+103", str(x))
        self.assertEqual(x.prec, 47)

        # Large int, explicit precision, exact.
        x = real(-(1 << 100), 128)
        self.assertEqual("-1.267650600228229401496703205376000000000e+30", str(x))
        self.assertEqual(x.prec, 128)

        with self.assertRaises(ValueError) as cm:
            real(1, prec=-1)
        self.assertTrue(
            "Cannot init a real with a precision of -1" in str(cm.exception)
        )

        # Long double.
        x = real(ld("1.1"))
        self.assertTrue("1.1000000" in str(x))
        self.assertEqual(x.prec, np.finfo(ld).nmant + 1)
        x = real(ld("-1.1"))
        self.assertTrue("-1.100000" in str(x))

        x = real(ld("1.1"), 53)
        self.assertEqual(str(x), "1.1000000000000001")
        self.assertEqual(x.prec, 53)
        x = real(-ld("1.1"), 53)
        self.assertEqual(str(x), "-1.1000000000000001")
        self.assertEqual(x.prec, 53)

        with self.assertRaises(ValueError) as cm:
            real(ld("1.1"), prec=-1)
        self.assertTrue(
            "Cannot set the precision of a real to the value -1" in str(cm.exception)
        )

        # real128.
        if hasattr(core, "real128"):
            real128 = core.real128

            x = real(real128("1.1"))
            self.assertEqual(str(x), "1.10000000000000000000000000000000008")
            self.assertEqual(x.prec, 113)
            x = real(real128("-1.1"))
            self.assertEqual(str(x), "-1.10000000000000000000000000000000008")

            x = real(real128("1.1"), 23)
            self.assertEqual(str(x), "1.0999999")
            self.assertEqual(x.prec, 23)
            x = real(real128("-1.1"), 23)
            self.assertEqual(str(x), "-1.0999999")

            with self.assertRaises(ValueError) as cm:
                real(real128("1.1"), prec=-1)
            self.assertTrue(
                "Cannot set the precision of a real to the value -1"
                in str(cm.exception)
            )

        # From real.
        x = real(real(1.1))
        self.assertEqual(str(x), "1.1000000000000001")
        self.assertEqual(x.prec, np.finfo(float).nmant + 1)
        x = real(real(-1.1))
        self.assertEqual(str(x), "-1.1000000000000001")

        x = real(real(1.1, 23))
        self.assertEqual(str(x), str(real(1.1, 23)))
        self.assertEqual(x.prec, 23)
        x = real(real(-1.1, 23))
        self.assertEqual(str(x), str(real(-1.1, 23)))
        self.assertEqual(x.prec, 23)

        with self.assertRaises(ValueError) as cm:
            real(real(1.1), prec=-1)
        self.assertTrue(
            "Cannot set the precision of a real to the value -1" in str(cm.exception)
        )

        # Construction from string.
        x = real("1.1", 123)
        self.assertEqual("1.09999999999999999999999999999999999992", str(x))
        x = real("-1.1", 123)
        self.assertEqual("-1.09999999999999999999999999999999999992", str(x))

        with self.assertRaises(ValueError) as cm:
            real("1.1")
        self.assertTrue(
            "Cannot construct a real from a string without a precision value"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            real("hello world", prec=233)
        self.assertTrue(
            "The string 'hello world' cannot be interpreted as a floating-point value in base 10"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            real("hello world", prec=-1)
        self.assertTrue(
            "Cannot set the precision of a real to the value -1" in str(cm.exception)
        )

        # Construction from unsupported type.
        with self.assertRaises(TypeError) as cm:
            real([])
        self.assertTrue(
            'Cannot construct a real from an object of type "list"' in str(cm.exception)
        )

        # The limb address getter.
        x = real("1.1", 123)
        y = real("1.1", 123)
        self.assertNotEqual(x._limb_address, y._limb_address)
