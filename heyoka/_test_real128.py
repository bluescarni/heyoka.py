# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class real128_test_case(_ut.TestCase):
    def test_scalar(self):
        from . import core

        if not hasattr(core, "real128"):
            return

        import random
        from . import real128
        from .core import _ppc_arch

        if _ppc_arch:
            ld = float
        else:
            from numpy import longdouble as ld

        from numpy import float32 as f32

        # Make random testing deterministic.
        random.seed(42)

        # Constructors.
        self.assertEqual(str(real128()), "0")

        # Small ints.
        self.assertEqual(str(real128(1)), "1")
        self.assertEqual(str(real128(-1)), "-1")
        self.assertEqual(str(real128(42)), "42")
        self.assertEqual(str(real128(-42)), "-42")

        # Bool.
        self.assertEqual(real128(True), 1)
        self.assertEqual(real128(False), 0)

        # Large ints, still exactly representable.
        for _ in range(100):
            n = random.randint(-(2**101), 2**101)
            self.assertEqual(str(real128(n)), str(n))

        # Ints with bit width around the mantissa bit width.
        self.assertEqual(str(real128(2**113)), str(2**113))
        self.assertEqual(str(real128(-(2**113))), str(-(2**113)))
        self.assertEqual(str(real128(2**113 + 1)), str(2**113))
        self.assertEqual(str(real128(-(2**113) - 1)), str(-(2**113)))
        self.assertEqual(str(real128(2**113 - 1)), str(2**113 - 1))
        self.assertEqual(str(real128(-(2**113) + 1)), str(-(2**113) + 1))

        # Very large ints.
        for _ in range(100):
            n = random.randint(-(2**1001), 2**1001)
            self.assertEqual(str(real128(n)), str(real128(str(n))))

        # Construction from floats.
        self.assertEqual(str(real128(-0.0)), "-0")
        self.assertEqual(str(real128(42.0)), "42")
        self.assertEqual(str(real128(-float("inf"))), "-inf")
        self.assertEqual(str(real128(float("nan"))), "nan")

        # Construction from long double.
        self.assertEqual(str(real128(ld(-0.0))), "-0")
        self.assertEqual(str(real128(ld(42.0))), "42")
        self.assertEqual(str(real128(ld(-float("inf")))), "-inf")
        self.assertEqual(str(real128(ld(float("nan")))), "nan")

        # Construction from f32.
        self.assertEqual(str(real128(f32(-0.0))), "-0")
        self.assertEqual(str(real128(f32(42.0))), "42")
        self.assertEqual(str(real128(f32(-float("inf")))), "-inf")
        self.assertEqual(str(real128(f32(float("nan")))), "nan")
        self.assertEqual(str(real128(f32("1.1"))), "1.10000002384185791015625")

        # Construction from string.
        self.assertEqual(str(real128("-0")), "-0")
        self.assertEqual(str(real128("-inf")), "-inf")
        self.assertEqual(str(real128("42")), "42")
        self.assertEqual(str(real128("-123")), "-123")

        # Construction from real.
        if hasattr(core, "real"):
            real = core.real

            self.assertEqual(real128(real("1.1", 113)), real128("1.1"))
            self.assertNotEqual(real128(real("1.1", 100)), real128("1.1"))

        with self.assertRaises(ValueError) as cm:
            real128("hello world")
        self.assertTrue(
            "The string 'hello world' does not represent a valid quadruple-precision"
            " floating-point value"
            in str(cm.exception)
        )

        with self.assertRaises(TypeError) as cm:
            real128([])
        self.assertTrue(
            'Cannot construct a real128 from an object of type "list"'
            in str(cm.exception)
        )

        # Check that the constructor takes exactly 1 argument.
        with self.assertRaises(TypeError) as cm:
            real128(1, prec=7)
        self.assertTrue("function takes at most 1 argument" in str(cm.exception))

        with self.assertRaises(TypeError) as cm:
            real128(1, 2)
        self.assertTrue("function takes at most 1 argument" in str(cm.exception))

        with self.assertRaises(TypeError) as cm:
            real128(prec=7)
        self.assertTrue(
            "'prec' is an invalid keyword argument for this function"
            in str(cm.exception)
        )

        # Conversion to bool.
        self.assertTrue(bool(real128(1)))
        self.assertTrue(bool(real128(-1)))
        self.assertFalse(bool(real128(0)))
        self.assertTrue(bool(real128("inf")))
        self.assertTrue(bool(real128("nan")))
        self.assertTrue(bool(real128("-inf")))

        # Unary math operations.
        self.assertEqual(repr(+real128(1)), "1")
        self.assertEqual(repr(+real128(-1)), "-1")
        self.assertEqual(repr(-real128(42)), "-42")
        self.assertEqual(repr(-real128(-42)), "42")
        self.assertEqual(repr(abs(real128(-42))), "42")

        # Binary math ops.
        self.assertEqual(repr(real128(42) + real128(1)), "43")
        self.assertEqual(repr(real128(42) + 1), "43")
        self.assertEqual(repr(real128(42) + 1.0), "43")
        # self.assertEqual(repr(real128(42) + ld(1)), "43")
        self.assertEqual(repr(real128(42) + f32(1)), "43")
        self.assertEqual(repr(1 + real128(42)), "43")
        self.assertEqual(repr(1.0 + real128(42)), "43")
        # self.assertEqual(repr(ld(1) + real128(42)), "43")
        self.assertEqual(repr(f32(1) + real128(42)), "43")
        if hasattr(core, "real"):
            real = core.real
            self.assertTrue(isinstance(real128() + real(), real))
            self.assertTrue(isinstance(real() + real128(), real))
            self.assertEqual(
                real128("1.1") + real("1.1", 100), real("1.1", 113) + real("1.1", 100)
            )
            self.assertEqual(
                real("1.1", 100) + real128("1.1"), real("1.1", 113) + real("1.1", 100)
            )
        with self.assertRaises(TypeError) as cm:
            real128(1) + []
        with self.assertRaises(TypeError) as cm:
            [] + real128(1)

        # A small test to check that the numpy operators
        # do not interfere with our own.
        self.assertEqual(str(real128(0) + f32("1.1")), "1.10000002384185791015625")
        self.assertEqual(str(f32("1.1") + real128(0)), "1.10000002384185791015625")

        self.assertEqual(repr(real128(42) - real128(1)), "41")
        self.assertEqual(repr(real128(42) - 1), "41")
        self.assertEqual(repr(real128(42) - 1.0), "41")
        # self.assertEqual(repr(real128(42) - ld(1)), "41")
        self.assertEqual(repr(real128(42) - f32(1)), "41")
        self.assertEqual(repr(1 - real128(42)), "-41")
        self.assertEqual(repr(1.0 - real128(42)), "-41")
        # self.assertEqual(repr(ld(1) - real128(42)), "-41")
        self.assertEqual(repr(f32(1) - real128(42)), "-41")
        if hasattr(core, "real"):
            real = core.real
            self.assertTrue(isinstance(real128() - real(), real))
            self.assertTrue(isinstance(real() - real128(), real))
            self.assertEqual(
                real128("1.1") - real("1.1", 100), real("1.1", 113) - real("1.1", 100)
            )
            self.assertEqual(
                real("1.1", 100) - real128("1.1"), real("1.1", 100) - real("1.1", 113)
            )
        with self.assertRaises(TypeError) as cm:
            real128(1) - []
        with self.assertRaises(TypeError) as cm:
            [] - real128(1)

        self.assertEqual(repr(real128(42) * real128(2)), "84")
        self.assertEqual(repr(real128(42) * 2), "84")
        self.assertEqual(repr(real128(42) * 2.0), "84")
        # self.assertEqual(repr(real128(42) * ld(2)), "84")
        self.assertEqual(repr(real128(42) * f32(2)), "84")
        self.assertEqual(repr(2 * real128(42)), "84")
        self.assertEqual(repr(2.0 * real128(42)), "84")
        # self.assertEqual(repr(ld(2) * real128(42)), "84")
        self.assertEqual(repr(f32(2) * real128(42)), "84")
        if hasattr(core, "real"):
            real = core.real
            self.assertTrue(isinstance(real128() * real(), real))
            self.assertTrue(isinstance(real() * real128(), real))
            self.assertEqual(
                real128("1.1") * real("1.1", 100), real("1.1", 113) * real("1.1", 100)
            )
            self.assertEqual(
                real("1.1", 100) * real128("1.1"), real("1.1", 113) * real("1.1", 100)
            )
        with self.assertRaises(TypeError) as cm:
            real128(1) * []
        with self.assertRaises(TypeError) as cm:
            [] * real128(1)

        self.assertEqual(repr(real128(42) // real128(9)), "4")
        self.assertEqual(repr(real128(42) // 9), "4")
        self.assertEqual(repr(real128(42) // 9.0), "4")
        # self.assertEqual(repr(real128(42) // ld(9)), "4")
        self.assertEqual(repr(real128(42) // f32(9)), "4")
        self.assertEqual(repr(-42 // real128(9)), "-5")
        self.assertEqual(repr(-42.0 // real128(9)), "-5")
        # self.assertEqual(repr(ld(-42) // real128(9)), "-5")
        self.assertEqual(repr(f32(-42) // real128(9)), "-5")
        if hasattr(core, "real"):
            real = core.real
            self.assertTrue(isinstance(real128(1) // real(1), real))
            self.assertTrue(isinstance(real(1) // real128(1), real))
            self.assertEqual(
                real128("1.1") // real("1.1", 100), real("1.1", 113) // real("1.1", 100)
            )
            self.assertEqual(
                real("1.1", 100) // real128("1.1"), real("1.1", 100) // real("1.1", 113)
            )
        with self.assertRaises(TypeError) as cm:
            real128(1) // []
        with self.assertRaises(TypeError) as cm:
            [] // real128(1)

        self.assertEqual(repr(real128(42) ** real128(2)), "1764")
        self.assertEqual(repr(real128(42) ** 2), "1764")
        self.assertEqual(repr(real128(42) ** 2.0), "1764")
        # self.assertEqual(repr(real128(42) ** ld(2)), "1764")
        self.assertEqual(repr(real128(42) ** f32(2)), "1764")
        self.assertEqual(repr(42 ** real128(2)), "1764")
        self.assertEqual(repr(42.0 ** real128(2)), "1764")
        # self.assertEqual(repr(ld(42) ** real128(2)), "1764")
        self.assertEqual(repr(f32(42) ** real128(2)), "1764")
        with self.assertRaises(TypeError) as cm:
            real128(1) ** []
        with self.assertRaises(TypeError) as cm:
            [] ** real128(1)
        with self.assertRaises(ValueError) as cm:
            pow(real128(1), 3, mod=4)
        self.assertTrue(
            "Modular exponentiation is not supported for real128" in str(cm.exception)
        )

        # Comparisons.
        self.assertTrue(real128(1) < real128(2))
        self.assertTrue(real128(1) < 2)
        self.assertTrue(real128(1) < 2.0)
        # self.assertTrue(real128(1) < ld(2))
        self.assertTrue(real128(1) < f32(2))
        self.assertTrue(1 < real128(2))
        self.assertTrue(1.0 < real128(2))
        # self.assertTrue(ld(1) < real128(2))
        self.assertTrue(f32(1) < real128(2))
        self.assertFalse(real128("nan") < 2)
        self.assertFalse(2 < real128("nan"))
        self.assertFalse(real128("nan") < real128("nan"))
        if hasattr(core, "real"):
            real = core.real

            self.assertTrue(real(1) < real128(2))
            self.assertTrue(real128(1) < real(2))
            self.assertFalse(real(1) > real128(2))
            self.assertFalse(real128(1) > real(2))
        with self.assertRaises(TypeError) as cm:
            real128(1) < []
        with self.assertRaises(TypeError) as cm:
            [] < real128(1)

        # The codepath for the other comparisons is identical,
        # let's limit to some light testing.
        self.assertTrue(real128(1) <= real128(2))
        self.assertTrue(real128(2) <= real128(2))
        self.assertFalse(real128(3) <= real128(2))
        self.assertTrue(real128(2) == real128(2))
        self.assertFalse(real128(3) == real128(2))
        self.assertFalse(real128(3) == real128("NaN"))
        self.assertFalse(real128(2) != real128(2))
        self.assertTrue(real128(3) != real128(2))
        self.assertTrue(real128(3) != real128("NaN"))
        self.assertTrue(real128(3) > real128(2))
        self.assertFalse(real128(2) > real128(3))
        self.assertFalse(real128(1) >= real128(2))
        self.assertTrue(real128(2) >= real128(2))
        self.assertTrue(real128(3) >= real128(2))

        # Copy/deepcopy.
        from copy import copy, deepcopy

        x = real128("1.1")
        y = copy(x)
        self.assertEqual(x, y)
        y = deepcopy(x)
        self.assertEqual(x, y)

        # Pickling.
        # NOTE: this is provided for free by the
        # NumPy inheritance.
        import pickle

        x = real128("1.1")
        y = pickle.loads(pickle.dumps(x))
        self.assertEqual(x, y)

    def test_numpy(self):
        from . import core

        if not hasattr(core, "real128"):
            return

        import numpy as np
        from copy import copy, deepcopy
        from . import real128
        from pickle import dumps, loads

        # Basic creation/getitem/setitem.
        arr = np.array([1, 2, 3], dtype=real128)
        self.assertEqual(type(arr[0]), real128)
        self.assertEqual(arr[0], 1)
        self.assertEqual(arr[1], 2)
        self.assertEqual(arr[2], 3)

        arr = np.array([real128("1.1")])
        self.assertEqual(arr.dtype, real128)
        self.assertEqual(type(arr[0]), real128)
        self.assertEqual(arr[0], real128("1.1"))
        arr[0] = real128("1.3")
        self.assertEqual(arr[0], real128("1.3"))
        arr[0] = real128(123)
        self.assertEqual(arr[0], 123)
        arr[0] = real128(1.1)
        self.assertEqual(arr[0], real128(1.1))

        with self.assertRaises(TypeError) as cm:
            arr = np.array(["1.1"], dtype=np.dtype(real128))
        self.assertTrue(
            "Cannot invoke __setitem__() on a real128 array with an input value of type"
            ' "str"'
            in str(cm.exception)
        )

        # Copying primitives.
        arr = np.array([real128("1.1"), real128("1.3")])
        arr_copy = copy(arr)
        self.assertTrue(arr_copy[0] == real128("1.1"))
        self.assertTrue(arr_copy[1] == real128("1.3"))
        arr_copy = deepcopy(arr)
        self.assertTrue(arr_copy[0] == real128("1.1"))
        self.assertTrue(arr_copy[1] == real128("1.3"))

        # Nonzero.
        arr = np.array([1, 0, 3, real128("nan")], dtype=real128)
        self.assertTrue(np.all([0, 2, 3] == np.nonzero(arr)[0]))

        # Fill.
        arr = np.zeros((10,), dtype=real128)
        arr.fill(real128("1.1"))
        for _ in arr:
            self.assertEqual(_, real128("1.1"))

        # Argmin/argmax.
        arr = np.array([1, 321, 54, 6, 2, 6, -6], dtype=real128)
        self.assertEqual(np.argmin(arr), 6)
        self.assertEqual(np.argmax(arr), 1)
        arr = np.array([], dtype=real128)
        with self.assertRaises(ValueError) as cm:
            np.argmin(arr)
        with self.assertRaises(ValueError) as cm:
            np.argmax(arr)

        # arange() and linspace().
        arr = np.arange(0, 1, real128("0.3"))
        self.assertTrue(
            np.all(
                arr
                == np.array(
                    [
                        0,
                        real128("0.29999999999999999999999999999999999"),
                        real128("0.599999999999999999999999999999999981"),
                        real128("0.899999999999999999999999999999999923"),
                    ],
                    dtype=real128,
                )
            )
        )
        self.assertEqual(arr.dtype, real128)
        arr = np.linspace(real128(0), 1, 4, dtype=real128)
        self.assertTrue(
            np.all(
                arr
                == np.array(
                    [
                        0,
                        real128("0.333333333333333333333333333333333317"),
                        real128("0.666666666666666666666666666666666635"),
                        1,
                    ],
                    dtype=real128,
                )
            )
        )
        self.assertEqual(arr.dtype, real128)

        # zeros, ones, full.
        arr = np.zeros((2, 2), dtype=real128)
        self.assertTrue(np.all(arr == np.array([[0, 0], [0, 0]], dtype=real128)))
        arr = np.ones((2, 2), dtype=real128)
        self.assertTrue(np.all(arr == np.array([[1, 1], [1, 1]], dtype=real128)))
        arr = np.full((2, 2), real128("1.1"))
        self.assertTrue(
            np.all(
                arr
                == np.array(
                    [[real128("1.1"), real128("1.1")], [real128("1.1"), real128("1.1")]]
                )
            )
        )

        # dot product.
        arr1 = np.array([real128("1.1"), real128("1.3")])
        arr2 = np.array([real128("2.1"), real128("2.3")])
        self.assertEqual(
            real128("1.1") * real128("2.1") + real128("1.3") * real128("2.3"),
            np.dot(arr1, arr2),
        )
        arr1 = np.array([], dtype=real128)
        arr2 = np.array([], dtype=real128)
        self.assertEqual(0, np.dot(arr1, arr2))

        # Matrix multiplication.
        mat = np.array(
            [[real128("1.1"), real128("1.3")], [real128("2.1"), real128("2.3")]]
        )
        self.assertTrue(
            np.all(
                mat @ mat
                == np.array(
                    [
                        [
                            real128("3.94000000000000000000000000000000034"),
                            real128("4.41999999999999999999999999999999994"),
                        ],
                        [
                            real128("7.14000000000000000000000000000000049"),
                            real128("8.01999999999999999999999999999999963"),
                        ],
                    ]
                )
            )
        )

        # Conversions.
        arr = np.array([real128("1.1")])
        self.assertEqual(arr.astype(float)[0], 1.1)
        self.assertEqual(arr.astype(np.int32)[0], 1)
        self.assertEqual(arr.astype(bool)[0], True)
        arr = np.array([1], dtype=np.int32)
        self.assertEqual(arr.astype(real128)[0], real128(1))
        self.assertEqual(arr.astype(real128, casting="safe")[0], real128(1))
        arr = np.array([1.1], dtype=float)
        self.assertEqual(arr.astype(real128)[0], real128(1.1))
        self.assertEqual(arr.astype(real128, casting="safe")[0], real128(1.1))
        arr = np.array([real128("1.1")])
        with self.assertRaises(TypeError) as cm:
            arr.astype(float, casting="safe")
        with self.assertRaises(TypeError) as cm:
            arr.astype(np.int32, casting="safe")

        # Arithmetic.
        a = real128("1.1")
        b = real128("1.3")
        c = real128("-.7")
        intval = 2
        arr1 = np.array([a])
        arr2 = np.array([b])
        arr3 = np.array([c])
        arrint = np.array([2], dtype=np.int64)

        self.assertEqual((arr1 + arr2)[0], a + b)
        self.assertEqual((arr1 + arrint)[0], a + intval)
        self.assertEqual((arrint + arr2)[0], intval + b)

        self.assertEqual((arr1 - arr2)[0], a - b)
        self.assertEqual((arr1 - arrint)[0], a - intval)
        self.assertEqual((arrint - arr2)[0], intval - b)

        self.assertEqual((arr1 * arr2)[0], a * b)
        self.assertEqual((arr1 * arrint)[0], a * intval)
        self.assertEqual((arrint * arr2)[0], intval * b)

        self.assertEqual((arr1 / arr2)[0], a / b)
        self.assertEqual((arr1 / arrint)[0], a / intval)
        self.assertEqual((arrint / arr2)[0], intval / b)

        self.assertEqual(np.square(arr1)[0], a * a)

        self.assertEqual((arr1 // arr2)[0], a // b)
        self.assertEqual((arr1 // arrint)[0], a // intval)
        self.assertEqual((arrint // arr2)[0], intval // b)

        self.assertEqual((+arr1)[0], a)
        self.assertEqual((-arr1)[0], -a)

        self.assertEqual(abs(-arr1)[0], a)
        self.assertEqual(np.abs(-arr1)[0], a)
        self.assertEqual(np.fabs(-arr1)[0], a)

        # Power/roots.
        self.assertEqual((arr1**arr2)[0], a**b)
        self.assertEqual((arr1**arrint)[0], a**intval)
        self.assertEqual((arrint**arr2)[0], intval**b)

        self.assertEqual(
            np.sqrt(arr1)[0], real128("1.04880884817015154699145351367993759")
        )

        self.assertEqual(
            np.cbrt(arr1)[0], real128("1.03228011545636715921358522500970152")
        )

        # Trigonometry.
        self.assertEqual(
            np.sin(arr1)[0], real128("0.891207360061435339951802577871703605")
        )

        self.assertEqual(
            np.cos(arr1)[0], real128("0.453596121425577387771370051784716053")
        )

        self.assertEqual(
            np.tan(arr1)[0], real128("1.96475965724865195093092278177937863")
        )

        self.assertEqual(
            np.arcsin(arr3)[0], real128("-0.775397496610753063740353352714987079")
        )

        self.assertEqual(
            np.arccos(arr3)[0], real128("2.34619382340564968297167504435473857")
        )

        self.assertEqual(
            np.arctan(arr3)[0], real128("-0.610725964389208616543758876490236037")
        )

        self.assertEqual(
            np.arctan2(arr1, arr2)[0], real128("0.702256931509007079704992531169094469")
        )

        self.assertEqual(
            np.arctan2(arr1, arrint)[0],
            real128("0.502843210927860827330882029245277632"),
        )
        self.assertEqual(
            np.arctan2(arrint, arr2)[0],
            real128("0.994421106203712939003751213245981381"),
        )

        self.assertEqual(
            np.sinh(arr1)[0], real128("1.33564747012417677938478052357867843")
        )

        self.assertEqual(
            np.cosh(arr1)[0], real128("1.6685185538222563326736274300099942")
        )

        self.assertEqual(
            np.tanh(arr1)[0], real128("0.800499021760629706011461330600696557")
        )

        self.assertEqual(
            np.arcsinh(arr3)[0], real128("-0.652666566082355786808686344109675833")
        )

        self.assertEqual(
            np.arccosh(arr1)[0], real128("0.443568254385115189132911066352498299")
        )

        self.assertEqual(
            np.arctanh(arr3)[0], real128("-0.867300527694053194427144690475300431")
        )

        self.assertEqual(
            np.deg2rad(np.array([1], dtype=real128))[0],
            real128(
                "0.01745329251994329576923690768488612713442871888541725456097191440171005"
            ),
        )

        self.assertEqual(
            np.radians(np.array([1], dtype=real128))[0],
            real128(
                "0.01745329251994329576923690768488612713442871888541725456097191440171005"
            ),
        )

        self.assertEqual(
            np.rad2deg(np.array([1], dtype=real128))[0],
            real128(
                "57.295779513082320876798154814105170332405472466564321549160243861202985"
            ),
        )

        self.assertEqual(
            np.degrees(np.array([1], dtype=real128))[0],
            real128(
                "57.295779513082320876798154814105170332405472466564321549160243861202985"
            ),
        )

        # Exponentials and logarithms.
        self.assertEqual(
            np.exp(arr1)[0], real128("3.00416602394643311205840795358867282")
        )

        self.assertEqual(
            np.expm1(arr1)[0], real128("2.00416602394643311205840795358867243")
        )

        self.assertEqual(
            np.log(arr1)[0], real128("0.0953101798043248600439521232807651556")
        )

        self.assertEqual(
            np.log2(arr1)[0], real128("0.137503523749934908329043617236402896")
        )

        self.assertEqual(
            np.log10(arr1)[0], real128("0.0413926851582250407501999712430242757")
        )

        self.assertEqual(
            np.log1p(arr1)[0], real128("0.741937344729377312482606525681341284")
        )

        # Comparisons.
        self.assertEqual((arr1 < arr2).dtype, bool)
        self.assertTrue((arr1 < arr2)[0])
        self.assertEqual((arr1 < arrint).dtype, bool)
        self.assertTrue((arr1 < arrint)[0])

        self.assertEqual((arr1 <= arr1).dtype, bool)
        self.assertTrue((arr1 <= arr2)[0])
        self.assertEqual((arr1 <= arrint).dtype, bool)
        self.assertTrue((arr1 <= arrint)[0])

        self.assertEqual((arr1 == arr1).dtype, bool)
        self.assertFalse((arr1 == arr2)[0])
        self.assertEqual((arr1 == arrint).dtype, bool)
        self.assertFalse((arr1 == arrint)[0])

        self.assertEqual((arr1 != arr1).dtype, bool)
        self.assertTrue((arr1 != arr2)[0])
        self.assertEqual((arr1 != arrint).dtype, bool)
        self.assertTrue((arr1 != arrint)[0])

        self.assertEqual((arr1 > arr2).dtype, bool)
        self.assertTrue((arr2 > arr1)[0])
        self.assertEqual((arrint > arr1).dtype, bool)
        self.assertTrue((arrint > arr1)[0])

        self.assertEqual((arr1 >= arr1).dtype, bool)
        self.assertTrue((arr2 >= arr1)[0])
        self.assertEqual((arrint >= arr1).dtype, bool)
        self.assertTrue((arrint >= arr1)[0])

        # isfinite().
        self.assertEqual((np.isfinite(arr1)).dtype, bool)
        self.assertTrue((np.isfinite(arr1))[0])
        self.assertTrue((np.isfinite([real128("1"), real128("nan")]))[0])
        self.assertFalse((np.isfinite([real128("1"), real128("nan")]))[1])
        self.assertFalse((np.isfinite([real128("1"), real128("+inf")]))[1])

        # isinf().
        self.assertEqual((np.isinf(arr1)).dtype, bool)
        self.assertFalse((np.isinf(arr1))[0])
        self.assertFalse((np.isinf([real128("1"), real128("nan")]))[0])
        self.assertTrue((np.isinf([real128("1"), real128("-inf")]))[1])
        self.assertTrue((np.isinf([real128("1"), real128("+inf")]))[1])

        # sign.
        self.assertEqual(np.sign(-real128("1.1")), -1)
        self.assertEqual(np.sign(real128("0")), 0)
        self.assertEqual(np.sign(real128("1.1")), 1)
        self.assertEqual((np.sign(arr1)).dtype, real128)
        self.assertTrue(
            np.all(np.sign([real128("10"), real128("0"), real128("-10")]) == [1, 0, -1])
        )

        # maximum/minimum.
        self.assertTrue(
            np.all(
                np.minimum([10, real128("1.1"), 20], [1, real128("1.11"), 25])
                == [1, real128("1.1"), 20]
            )
        )
        self.assertTrue(
            np.all(
                np.maximum([10, real128("1.1"), 20], [1, real128("1.11"), 25])
                == [10, real128("1.11"), 25]
            )
        )

        # Pickling.
        tmp = np.array([real128("1.1"), real128("1.11"), real128("1.13")])
        self.assertTrue(np.all(tmp == loads(dumps(tmp))))

        # Use hstack to check the copyswap implementation.
        arr1 = np.array([real128("1.1"), real128("1.11"), real128("1.13")])
        arr2 = np.array([real128("2.1"), real128("2.11"), real128("2.13")])
        arr3 = np.hstack([arr1, arr2])

        self.assertTrue(np.all(arr3[:3] == arr1))
        self.assertTrue(np.all(arr3[3:] == arr2))

        # Sorting (via the compare primitive).
        arr1 = np.array([real128("3"), real128("1"), real128("2")])
        arr1_sorted = np.sort(arr1)
        self.assertTrue(
            np.all(arr1_sorted == [real128("1"), real128("2"), real128("3")])
        )

        # Check NaN handling.
        arr1 = np.array([real128("nan"), real128("1"), real128("nan")])
        self.assertTrue(all(np.isnan(arr1) == [True, False, True]))
        arr1_sorted = np.sort(arr1)
        self.assertEqual(arr1_sorted[0], 1)
        self.assertFalse(np.isnan(arr1_sorted[0]))
        self.assertTrue(np.isnan(arr1_sorted[1]))
        self.assertTrue(np.isnan(arr1_sorted[2]))

        # Setitem from real.
        if hasattr(core, "real"):
            real = core.real

            arr1 = np.array([1, 2, 3], dtype=real128)
            arr1[1] = real("1.1", 113)
            self.assertEqual(arr1[1], real128("1.1"))
