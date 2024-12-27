# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class real_test_case(_ut.TestCase):
    def test_numpy_realloc(self):
        from . import core

        if not hasattr(core, "real"):
            return

        from . import real
        import numpy as np

        # Start with a full array,
        arr = np.full((2,), real("1.1", 128))
        arr.resize((5,))
        self.assertTrue(np.all(arr == [real("1.1", 128)] * 2 + [real()] * 3))
        arr.resize((1,))
        self.assertTrue(np.all(arr == [real("1.1", 128)]))
        arr.resize((5,))
        self.assertTrue(np.all(arr == [real("1.1", 128)] * 1 + [real()] * 4))
        arr.resize((0,))
        self.assertEqual(arr.shape, (0,))
        arr.resize((5,))
        self.assertTrue(np.all(arr == [real()] * 5))

        # Completely empty.
        arr = np.empty((2,), dtype=real)
        arr.resize((5,))
        self.assertTrue(np.all(arr == [real()] * 5))
        arr.resize((1,))
        self.assertTrue(np.all(arr == [real()] * 1))
        arr.resize((10,))
        self.assertTrue(np.all(arr == [real()] * 10))
        arr.resize((0,))
        self.assertEqual(arr.shape, (0,))
        arr.resize((5,))
        self.assertTrue(np.all(arr == [real()] * 5))

        # Some element inited, some not.
        arr = np.empty((5,), dtype=real)
        arr[::2] = real("1.1", 128)
        arr.resize((10,))
        self.assertTrue(np.all(arr[:5:2] == [real("1.1", 128)] * 3))
        self.assertTrue(np.all(arr[1:5:2] == [real()] * 2))
        self.assertTrue(np.all(arr[5:] == [real()] * 5))
        arr.resize((3,))
        self.assertTrue(np.all(arr[::2] == [real("1.1", 128)] * 2))
        arr.resize((0,))
        self.assertEqual(arr.shape, (0,))
        arr.resize((5,))
        self.assertTrue(np.all(arr == [real()] * 5))

        # Resize to same.
        arr = np.empty((5,), dtype=real)
        arr[::2] = real("1.1", 128)
        arr.resize((5,))
        self.assertTrue(np.all(arr[::2] == [real("1.1", 128)] * 3))
        self.assertTrue(np.all(arr[1::2] == [real()] * 2))

        # Test with POD types.
        arr = np.full((2,), 1.1)
        arr.resize((5,))
        self.assertTrue(np.all(arr[:2] == [1.1] * 2))
        arr.resize((1,))
        self.assertTrue(np.all(arr[:1] == [1.1]))
        arr.resize((0,))
        self.assertEqual(arr.shape, (0,))

    def test_numpy_matmul(self):
        from . import core

        if not hasattr(core, "real"):
            return

        from . import real
        import numpy as np

        # Matrix multiplication.
        mat = np.array(
            [[real("1.1", 113), real("1.3", 113)], [real("2.1", 113), real("2.3", 113)]]
        )
        self.assertTrue(
            np.all(
                mat @ mat
                == np.array(
                    [
                        [
                            real("3.94000000000000000000000000000000034", 113),
                            real("4.41999999999999999999999999999999994", 113),
                        ],
                        [
                            real("7.14000000000000000000000000000000049", 113),
                            real("8.01999999999999999999999999999999963", 113),
                        ],
                    ]
                )
            )
        )

        # With output.
        ret = np.empty((2, 2), dtype=real)
        np.matmul(mat, mat, out=ret)
        self.assertTrue(
            np.all(
                ret
                == np.array(
                    [
                        [
                            real("3.94000000000000000000000000000000034", 113),
                            real("4.41999999999999999999999999999999994", 113),
                        ],
                        [
                            real("7.14000000000000000000000000000000049", 113),
                            real("8.01999999999999999999999999999999963", 113),
                        ],
                    ]
                )
            )
        )

        # With skipping.
        mat = np.array(
            [
                [real("1.1", 113), 0, real("1.3", 113), 0],
                [real()] * 4,
                [real("2.1", 113), 0, real("2.3", 113), 0],
                [real()] * 4,
            ]
        )
        mat = mat[::2, ::2]
        self.assertTrue(
            np.all(
                mat @ mat
                == np.array(
                    [
                        [
                            real("3.94000000000000000000000000000000034", 113),
                            real("4.41999999999999999999999999999999994", 113),
                        ],
                        [
                            real("7.14000000000000000000000000000000049", 113),
                            real("8.01999999999999999999999999999999963", 113),
                        ],
                    ]
                )
            )
        )

        # With uninited values.
        mat = np.empty((2, 2), dtype=real)
        mat[0, 0] = real("1.1", 113)
        mat[1, 1] = real("2.3", 113)
        self.assertTrue(
            np.all(
                mat @ mat
                == np.array(
                    [
                        [
                            real("1.1", 113) * real("1.1", 113),
                            0,
                        ],
                        [
                            0,
                            real("2.3", 113) * real("2.3", 113),
                        ],
                    ]
                )
            )
        )

    def test_numpy_comparisons(self):
        from . import core

        if not hasattr(core, "real"):
            return

        from . import real
        import numpy as np
        from . import core

        make_no = core._make_no_real_array

        # Basic.
        arr = np.array([1, 2, 3, 4, 5], dtype=real)
        self.assertTrue(np.all(arr == arr))
        self.assertEqual((arr == arr).dtype, bool)
        arr2 = np.array([1, -2, 3, -4, 5], dtype=real)
        self.assertTrue(np.all((arr == arr2) == [True, False, True, False, True]))

        # With uninited values.
        arr2 = np.empty((5,), dtype=real)
        arr2[0] = 1
        self.assertTrue(np.all((arr == arr2) == [True, False, False, False, False]))

        # With holes.
        arr2 = np.array(
            [1, 0, -2, 0, 3, 0, -4, 0, 5, 0],
            dtype=real,
        )
        self.assertTrue(np.all((arr == arr2[::2]) == [True, False, True, False, True]))

        # Non-owning.
        self.assertTrue(
            np.all(
                (make_no(arr) == make_no(arr2)[::2]) == [True, False, True, False, True]
            )
        )

        # Testing for the other comparison operators.
        self.assertTrue(
            np.all((make_no(arr) != arr2[::2]) == [False, True, False, True, False])
        )
        self.assertTrue(
            np.all(
                (np.array([1, 2, 3], dtype=real) < np.array([1, 3, -1], dtype=real))
                == [False, True, False]
            )
        )
        self.assertTrue(
            np.all(
                (np.array([1, 2, 3], dtype=real) <= np.array([1, 3, -1], dtype=real))
                == [True, True, False]
            )
        )
        self.assertTrue(
            np.all(
                (np.array([1, 2, 3], dtype=real) > np.array([1, 3, -1], dtype=real))
                == [False, False, True]
            )
        )
        self.assertTrue(
            np.all(
                (np.array([1, 2, 3], dtype=real) >= np.array([1, 3, -1], dtype=real))
                == [True, False, True]
            )
        )

        # Unary comparisons.
        arr = np.array([1, float("nan"), 3, float("nan"), 5], dtype=real)
        self.assertTrue(np.all((np.isnan(arr)) == [False, True, False, True, False]))
        self.assertEqual(np.isnan(arr).dtype, bool)

        # With uninited values.
        arr = np.empty((5,), dtype=real)
        arr[1] = float("nan")
        arr[3] = float("nan")
        self.assertTrue(np.all((np.isnan(arr)) == [False, True, False, True, False]))

        # With holes.
        arr = np.array([1, 0, float("nan"), 0, 3, 0, float("nan"), 0, 5, 0], dtype=real)
        self.assertTrue(
            np.all((np.isnan(arr[::2])) == [False, True, False, True, False])
        )
        self.assertEqual(np.isnan(arr[::2]).dtype, bool)

        # Other unary comparisons.
        arr = np.array([1, float("inf"), 3, float("nan"), 5], dtype=real)
        self.assertTrue(np.all((np.isinf(arr)) == [False, True, False, False, False]))
        arr = np.array([1, float("inf"), 3, float("nan"), 5], dtype=real)
        self.assertTrue(np.all((np.isfinite(arr)) == [True, False, True, False, True]))

        # Try comparison wrt scalars too.
        arr = np.array([1], dtype=real)
        self.assertTrue(np.all(arr == 1))
        self.assertTrue(np.all(arr != 0))
        self.assertTrue(np.all(arr < real(2)))
        self.assertTrue(np.all(arr > 0))
        # self.assertTrue(np.all(arr <= 3.0))
        if hasattr(core, "real128"):
            real128 = core.real128

            self.assertTrue(np.all(arr >= real128(-2.0)))

    def test_numpy_binary(self):
        from . import core

        if not hasattr(core, "real"):
            return

        from . import real
        import numpy as np

        arr1 = np.full((10,), real("1.1", 128), dtype=real)
        arr2 = np.full((10,), real("1", 128), dtype=real)
        arr3 = arr1 // arr2
        self.assertTrue(np.all(arr3 == np.full((10,), real("1", 128), dtype=real)))
        ret = np.empty((10,), dtype=real)
        np.floor_divide(arr1, arr2, out=ret)
        self.assertTrue(np.all(arr3 == ret))

        arr1 = np.full((10,), real("1.1", 128), dtype=real)
        arr2 = np.full((10,), real("-1", 128), dtype=real)
        arr3 = arr1 // arr2
        self.assertTrue(np.all(arr3 == np.full((10,), real("-2", 128), dtype=real)))
        ret = np.empty((10,), dtype=real)
        np.floor_divide(arr1, arr2, out=ret)
        self.assertTrue(np.all(arr3 == ret))

        arr1 = np.full((10,), real("1.1", 128), dtype=real)
        arr2 = np.full((10,), real("-1", 128), dtype=real)
        arr3 = arr1**arr2
        self.assertTrue(
            np.all(
                arr3
                == np.full(
                    (10,),
                    real("0.90909090909090909090909090909090909090695", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.power(arr1, arr2, out=ret)
        self.assertTrue(np.all(arr3 == ret))

        arr1 = np.full((10,), real("1.1", 128), dtype=real)
        arr2 = np.full((10,), real("-1", 128), dtype=real)
        arr3 = np.arctan2(arr1, arr2)
        self.assertTrue(
            np.all(
                arr3
                == np.full(
                    (10,),
                    real("2.3086113869153615330449498214431416456855", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.arctan2(arr1, arr2, out=ret)
        self.assertTrue(np.all(arr3 == ret))

        arr1 = np.full((10,), real("1.1", 128), dtype=real)
        arr2 = np.full((10,), real("-1", 128), dtype=real)
        arr3 = np.minimum(arr1, arr2)
        self.assertTrue(np.all(arr3 == np.full((10,), real("-1", 128), dtype=real)))
        ret = np.empty((10,), dtype=real)
        np.minimum(arr1, arr2, out=ret)
        self.assertTrue(np.all(arr3 == ret))

        arr1 = np.full((10,), real("1.1", 128), dtype=real)
        arr2 = np.full((10,), real("-1", 128), dtype=real)
        arr3 = np.maximum(arr1, arr2)
        self.assertTrue(np.all(arr3 == np.full((10,), real("1.1", 128), dtype=real)))
        ret = np.empty((10,), dtype=real)
        np.maximum(arr1, arr2, out=ret)
        self.assertTrue(np.all(arr3 == ret))

    def test_numpy_unary(self):
        from . import core

        if not hasattr(core, "real"):
            return

        from . import real
        import numpy as np

        arr1 = np.full((10,), real("-1.1", 128), dtype=real)
        arr2 = np.absolute(arr1)
        self.assertTrue(np.all(arr2 == np.full((10,), real("1.1", 128), dtype=real)))
        arr2 = np.fabs(arr1)
        self.assertTrue(np.all(arr2 == np.full((10,), real("1.1", 128), dtype=real)))
        ret = np.empty((10,), dtype=real)
        np.absolute(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))
        np.fabs(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("-1.1", 128), dtype=real)
        arr2 = +arr1
        self.assertTrue(np.all(arr2 == np.full((10,), real("-1.1", 128), dtype=real)))
        ret = np.empty((10,), dtype=real)
        np.positive(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("-1.1", 128), dtype=real)
        arr2 = -arr1
        self.assertTrue(np.all(arr2 == np.full((10,), real("1.1", 128), dtype=real)))
        ret = np.empty((10,), dtype=real)
        np.negative(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("1.1", 128), dtype=real)
        arr2 = np.sqrt(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("1.0488088481701515469914535136799375984778", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.sqrt(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("1.1", 128), dtype=real)
        arr2 = np.cbrt(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("1.0322801154563671592135852250097016117302", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.cbrt(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("1.1", 128), dtype=real)
        arr2 = np.sin(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("0.8912073600614353399518025778717035383202", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.sin(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("1.1", 128), dtype=real)
        arr2 = np.cos(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("0.45359612142557738777137005178471612212109", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.cos(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("1.1", 128), dtype=real)
        arr2 = np.tan(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("1.9647596572486519509309227817793782437226", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.tan(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("0.1", 128), dtype=real)
        arr2 = np.arcsin(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("0.10016742116155979634552317945269331856891", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.arcsin(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("0.1", 128), dtype=real)
        arr2 = np.arccos(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("1.4706289056333368228857985121870581235309", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.arccos(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("0.1", 128), dtype=real)
        arr2 = np.arctan(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("0.099668652491162027378446119878020590243274", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.arctan(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("1.1", 128), dtype=real)
        arr2 = np.sinh(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("1.3356474701241767793847805235786784358374", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.sinh(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("1.1", 128), dtype=real)
        arr2 = np.cosh(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("1.6685185538222563326736274300099939574508", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.cosh(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("1.1", 128), dtype=real)
        arr2 = np.tanh(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("0.80049902176062970601146133060069645804746", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.tanh(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("0.1", 128), dtype=real)
        arr2 = np.arcsinh(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("0.099834078899207563327303124704769443267764", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.arcsinh(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("1.1", 128), dtype=real)
        arr2 = np.arccosh(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("0.44356825438511518913291106635249808665203", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.arccosh(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("0.1", 128), dtype=real)
        arr2 = np.arctanh(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("0.1003353477310755806357265520600389452634", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.arctanh(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("0.1", 128), dtype=real)
        arr2 = np.deg2rad(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("0.001745329251994329576923690768488612713446", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.radians(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("0.1", 128), dtype=real)
        arr2 = np.rad2deg(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("5.7295779513082320876798154814105170332414", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.degrees(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("0.1", 128), dtype=real)
        arr2 = np.exp(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("1.1051709180756476248117078264902466682234", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.exp(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("0.1", 128), dtype=real)
        arr2 = np.exp2(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("1.0717734625362931642130063250233420229082", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.exp2(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("0.1", 128), dtype=real)
        arr2 = np.expm1(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("0.10517091807564762481170782649024666822453", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.expm1(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("0.1", 128), dtype=real)
        arr2 = np.log(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("-2.3025850929940456840179914546843642076025", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.log(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("0.1", 128), dtype=real)
        arr2 = np.log2(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("-3.3219280948873623478703194294893901758666", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.log2(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("0.1", 128), dtype=real)
        arr2 = np.log10(arr1)
        self.assertTrue(np.all(arr2 == np.full((10,), real("-1", 128), dtype=real)))
        ret = np.empty((10,), dtype=real)
        np.log10(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("0.1", 128), dtype=real)
        arr2 = np.log1p(arr1)
        self.assertTrue(
            np.all(
                arr2
                == np.full(
                    (10,),
                    real("0.095310179804324860043952123280765092220842", 128),
                    dtype=real,
                )
            )
        )
        ret = np.empty((10,), dtype=real)
        np.log1p(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))

        arr1 = np.full((10,), real("0.1", 128), dtype=real)
        arr2 = np.sign(arr1)
        self.assertTrue(np.all(arr2 == np.full((10,), real("1", 128), dtype=real)))
        self.assertTrue(all([_.prec == 128 for _ in arr2]))
        ret = np.empty((10,), dtype=real)
        np.sign(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))
        self.assertTrue(all([_.prec == 128 for _ in ret]))

        arr1 = np.full((10,), real("-0.1", 128), dtype=real)
        arr2 = np.sign(arr1)
        self.assertTrue(np.all(arr2 == np.full((10,), real("-1", 128), dtype=real)))
        self.assertTrue(all([_.prec == 128 for _ in arr2]))
        ret = np.empty((10,), dtype=real)
        np.sign(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))
        self.assertTrue(all([_.prec == 128 for _ in ret]))

        arr1 = np.full((10,), real("-0.", 128), dtype=real)
        arr2 = np.sign(arr1)
        self.assertTrue(np.all(arr2 == np.full((10,), real("0", 128), dtype=real)))
        self.assertTrue(all([_.prec == 128 for _ in arr2]))
        ret = np.empty((10,), dtype=real)
        np.sign(arr1, out=ret)
        self.assertTrue(np.all(arr2 == ret))
        self.assertTrue(all([_.prec == 128 for _ in ret]))

        arr1 = np.full((10,), real("nan", 128), dtype=real)
        arr2 = np.sign(arr1)
        self.assertTrue(np.all(np.isnan(arr2)))
        self.assertTrue(all([_.prec == 128 for _ in arr2]))
        ret = np.empty((10,), dtype=real)
        np.sign(arr1, out=ret)
        self.assertTrue(np.all(np.isnan(ret)))
        self.assertTrue(all([_.prec == 128 for _ in ret]))

    def test_numpy_pickle(self):
        from . import core

        if not hasattr(core, "real"):
            return

        # Pickle support.
        import pickle
        import numpy as np
        from . import real

        arr = np.full((10,), real("1.1", 128), dtype=real)
        arr2 = pickle.loads(pickle.dumps(arr))
        self.assertEqual(arr.shape, arr2.shape)
        for i in range(len(arr)):
            self.assertEqual(arr[i], arr2[i])
            self.assertEqual(arr2[i].prec, 128)

    def test_numpy_conversions(self):
        from . import core

        if not hasattr(core, "real"):
            return

        from . import real
        from . import core
        import numpy as np

        make_no = core._make_no_real_array

        # real -> other.

        # Basic testing.
        arr = np.array([real("1.1", 128)])
        self.assertEqual(arr.astype(float)[0], 1.1)
        self.assertEqual(arr.astype(np.int32)[0], 1)
        self.assertEqual(arr.astype(bool)[0], True)

        # With uninited values.
        arr = np.empty((2,), dtype=real)
        arr[0] = real("1.1", 128)
        self.assertEqual(arr.astype(float)[0], 1.1)
        self.assertEqual(arr.astype(float)[1], 0.0)
        self.assertEqual(arr.astype(np.int32)[0], 1)
        self.assertEqual(arr.astype(np.int32)[1], 0)
        self.assertEqual(arr.astype(bool)[0], True)
        self.assertEqual(arr.astype(bool)[1], False)

        # With holes.
        arr = np.empty((6,), dtype=real)
        arr[0] = real("1.1", 128)
        arr = arr[::3]
        self.assertEqual(arr.astype(float)[0], 1.1)
        self.assertEqual(arr.astype(float)[1], 0.0)
        self.assertEqual(arr.astype(np.int32)[0], 1)
        self.assertEqual(arr.astype(np.int32)[1], 0)
        self.assertEqual(arr.astype(bool)[0], True)
        self.assertEqual(arr.astype(bool)[1], False)

        # With non-owning.
        arr = np.empty((2,), dtype=real)
        arr[0] = real("1.1", 128)
        arr = make_no(arr)
        self.assertEqual(arr.astype(float)[0], 1.1)
        self.assertEqual(arr.astype(float)[1], 0.0)
        self.assertEqual(arr.astype(np.int32)[0], 1)
        self.assertEqual(arr.astype(np.int32)[1], 0)
        self.assertEqual(arr.astype(bool)[0], True)
        self.assertEqual(arr.astype(bool)[1], False)

        # other -> real.
        arr = np.array([1], dtype=np.int32)
        self.assertEqual(arr.astype(real)[0], real(1))
        self.assertEqual(arr.astype(real)[0].prec, 32)
        self.assertEqual(arr.astype(real, casting="safe")[0], real(1))
        self.assertEqual(arr.astype(real, casting="safe")[0].prec, 32)
        arr = np.array([1.1], dtype=float)
        self.assertEqual(arr.astype(real)[0], real(1.1))
        self.assertEqual(arr.astype(real)[0].prec, real(1.1).prec)
        self.assertEqual(arr.astype(real, casting="safe")[0], real(1.1))
        self.assertEqual(arr.astype(real, casting="safe")[0].prec, real(1.1).prec)
        arr = np.array([real("1.1", 128)])
        with self.assertRaises(TypeError) as cm:
            arr.astype(float, casting="safe")
        with self.assertRaises(TypeError) as cm:
            arr.astype(np.int32, casting="safe")

        # real128 interop.
        if hasattr(core, "real128"):
            real128 = core.real128

            arr = np.array([1, 2, 3, 4, 5], dtype=real128)
            self.assertTrue(np.all(arr == arr.astype(real)))
            self.assertTrue(np.all(arr == arr.astype(real, casting="safe")))

            arr = np.array([1, 2, 3, 4, 5], dtype=real)
            self.assertTrue(np.all(arr == arr.astype(real128)))
            with self.assertRaises(TypeError) as cm:
                self.assertTrue(np.all(arr == arr.astype(real128, casting="safe")))

    def test_numpy_square(self):
        from . import core

        if not hasattr(core, "real"):
            return

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
        from . import core

        if not hasattr(core, "real"):
            return

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
        from . import core

        if not hasattr(core, "real"):
            return

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
        from . import core

        if not hasattr(core, "real"):
            return

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
        from . import core

        if not hasattr(core, "real"):
            return

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
        from . import core

        if not hasattr(core, "real"):
            return

        from . import real
        from . import core
        import numpy as np

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

        # Check NaN handling.
        arr1 = np.array([real("nan", 128), real("1", 128), real("nan", 128)])
        self.assertTrue(all(np.isnan(arr1) == [True, False, True]))
        arr1_sorted = np.sort(arr1)
        self.assertEqual(arr1_sorted[0], 1)
        self.assertFalse(np.isnan(arr1_sorted[0]))
        self.assertTrue(np.isnan(arr1_sorted[1]))
        self.assertTrue(np.isnan(arr1_sorted[2]))

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
        arr = np.arange(0, 1, real("0.3", 113))
        self.assertTrue(
            np.all(
                arr
                == np.array(
                    [
                        0,
                        real("0.29999999999999999999999999999999999", 113),
                        real("0.599999999999999999999999999999999981", 113),
                        real("0.899999999999999999999999999999999923", 113),
                    ],
                    dtype=real,
                )
            )
        )
        self.assertEqual(arr.dtype, real)

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

        arr = np.linspace(real(0, 128), real(1, 128), 10, dtype=real)
        self.assertEqual(arr.dtype, real)
        self.assertEqual(arr[0].prec, 128)

        # Dot product.
        arr1 = np.array([real("1.1", 113), real("1.3", 113)])
        arr2 = np.array([real("2.1", 113), real("2.3", 113)])
        self.assertEqual(
            real("1.1", 113) * real("2.1", 113) + real("1.3", 113) * real("2.3", 113),
            np.dot(arr1, arr2),
        )
        arr1 = np.array([], dtype=real)
        arr2 = np.array([], dtype=real)
        self.assertEqual(0, np.dot(arr1, arr2))

        # A handful of mixed scalar/array operations.
        arr = np.array([2], dtype=real)
        arr2 = real("1.1", 128) * arr
        self.assertEqual(arr2[0], real("1.1", 128) * 2)
        self.assertEqual(arr2[0].prec, 128)
        arr2 = arr * real("1.1", 128)
        self.assertEqual(arr2[0], real("1.1", 128) * 2)
        self.assertEqual(arr2[0].prec, 128)

        arr = np.array([real("1.1", 128)])
        arr2 = 2 * arr
        self.assertEqual(arr2[0], real("1.1", 128) * 2)
        self.assertEqual(arr2[0].prec, 128)
        arr2 = arr * 2
        self.assertEqual(arr2[0], real("1.1", 128) * 2)
        self.assertEqual(arr2[0].prec, 128)

        # Use hstack to check the copyswap implementation.
        arr1 = np.array([real("1.1", 128), real("1.11", 128), real("1.13", 128)])
        arr2 = np.array([real("2.1", 128), real("2.11", 128), real("2.13", 128)])
        arr3 = np.hstack([arr1, arr2])

        self.assertTrue(np.all(arr3[:3] == arr1))
        self.assertTrue(np.all(arr3[3:] == arr2))

    def test_comparisons(self):
        from . import core

        if not hasattr(core, "real"):
            return

        from . import real
        from . import core
        import numpy as np

        ld = np.longdouble
        f32 = np.float32

        # Equality
        self.assertEqual(real(2), real(2))

        self.assertEqual(real(2), 2)
        self.assertEqual(2, real(2))

        self.assertEqual(real(2), 2.0)
        self.assertEqual(2.0, real(2))

        # self.assertEqual(real(2), ld(2.0))
        # self.assertEqual(ld(2.0), real(2))

        self.assertEqual(real(2), f32(2.0))
        self.assertEqual(f32(2.0), real(2))

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

        # self.assertNotEqual(real(3), ld(2.0))
        # self.assertNotEqual(ld(3.0), real(2))

        self.assertNotEqual(real(3), f32(2.0))
        self.assertNotEqual(f32(3.0), real(2))

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

        # self.assertLess(real(1), ld(2.0))
        # self.assertLess(ld(1.0), real(2))

        self.assertLess(real(1), f32(2.0))
        self.assertLess(f32(1.0), real(2))

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

        # self.assertLessEqual(real(1), ld(2.0))
        # self.assertLessEqual(ld(2.0), real(2))

        self.assertLessEqual(real(1), f32(2.0))
        self.assertLessEqual(f32(2.0), real(2))

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

        # self.assertGreater(real(3), ld(2.0))
        # self.assertGreater(ld(3.0), real(2))

        self.assertGreater(real(3), f32(2.0))
        self.assertGreater(f32(3.0), real(2))

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

        # self.assertGreaterEqual(real(3), ld(2.0))
        # self.assertGreaterEqual(ld(3.0), real(3))

        self.assertGreaterEqual(real(3), f32(2.0))
        self.assertGreaterEqual(f32(3.0), real(3))

        if hasattr(core, "real128"):
            real128 = core.real128

            self.assertGreaterEqual(real(3), real128(2))
            self.assertGreaterEqual(real128(3), real(3))

    def test_conversions(self):
        from . import core

        if not hasattr(core, "real"):
            return

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
        from . import core

        if not hasattr(core, "real"):
            return

        from . import real
        from . import core
        import numpy as np

        ld = np.longdouble
        f32 = np.float32

        # Plus.
        x = real(1, 128)
        y = real(-2, 128)
        self.assertEqual(str(x + y), str(real(-1, 128)))

        self.assertEqual(str(x + -2), str(real(-1, 128)))
        self.assertEqual(str(1 + y), str(real(-1, 128)))

        self.assertEqual(str(x + -2.0), str(real(-1, 128)))
        self.assertEqual(str(1.0 + y), str(real(-1, 128)))

        # self.assertEqual(str(x + ld(-2.0)), str(real(-1, 128)))
        # self.assertEqual(str(ld(1.0) + y), str(real(-1, 128)))

        self.assertEqual(str(real(1, 20) + f32("1.1")), "2.09999990")
        self.assertEqual(str(f32("1.1") + real(1, 20)), "2.09999990")

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

        # self.assertEqual(str(x - ld(-2.0)), str(real(3, 128)))
        # self.assertEqual(str(ld(1.0) - y), str(real(3, 128)))

        self.assertEqual(str(real(1, 20) - f32("1.1")), "-1.00000024e-1")
        self.assertEqual(str(f32("1.1") - real(1, 20)), "1.00000024e-1")

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

        # self.assertEqual(str(x * ld(-2.0)), str(real(-2, 128)))
        # self.assertEqual(str(ld(1.0) * y), str(real(-2, 128)))

        self.assertEqual(str(real(1, 20) * f32("1.1")), "1.10000002")
        self.assertEqual(str(f32("1.1") * real(1, 20)), "1.10000002")

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

        # self.assertEqual(str(x / ld(-2.0)), str(real(-0.5, 128)))
        # self.assertEqual(str(ld(1.0) / y), str(real(-0.5, 128)))

        self.assertEqual(str(real(1, 20) / f32("1.1")), "9.09090877e-1")
        self.assertEqual(str(f32("1.1") / real(1, 20)), "1.10000002")

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

        # self.assertEqual(str(real(2.1, 128) // ld(1.0)), str(real(2.0, 128)))
        # self.assertEqual(str(ld(1.0) // real(2.1, 128)), str(real(0.0, 128)))

        self.assertEqual(str(real(1, 20) // f32("1.1")), "0.00000000")
        self.assertEqual(str(f32("1.1") // real(1, 20)), "1.00000000")

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

        # self.assertEqual(str(real(2.0, 128) ** ld(3.0)), str(real(8.0, 128)))
        # self.assertEqual(str(ld(2.0) ** real(3.0, 128)), str(real(8.0, 128)))

        self.assertEqual(str(real(1, 20) ** f32("1.1")), "1.00000000")
        self.assertEqual(str(f32("1.1") ** real(1, 20)), "1.10000002")

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

        # Comparisons.
        self.assertTrue(real(1) < real(2))
        self.assertTrue(real(1) < 2)
        self.assertTrue(real(1) < 2.0)
        # self.assertTrue(real(1) < ld(2))
        self.assertTrue(real(1) < f32(2))
        self.assertTrue(1 < real(2))
        self.assertTrue(1.0 < real(2))
        # self.assertTrue(ld(1) < real(2))
        self.assertTrue(f32(1) < real(2))
        self.assertFalse(real("nan", 10) < 2)
        self.assertFalse(2 < real("nan", 10))
        self.assertFalse(real("nan", 10) < real("nan", 10))
        if hasattr(core, "real128"):
            real128 = core.real128

            self.assertTrue(real(1) < real128(2))
            self.assertTrue(real128(1) < real(2))
            self.assertFalse(real(1) > real128(2))
            self.assertFalse(real128(1) > real(2))
        with self.assertRaises(TypeError) as cm:
            real(1) < []
        with self.assertRaises(TypeError) as cm:
            [] < real(1)

        # The codepath for the other comparisons is identical,
        # let's limit to some light testing.
        self.assertTrue(real(1) <= real(2))
        self.assertTrue(real(2) <= real(2))
        self.assertFalse(real(3) <= real(2))
        self.assertTrue(real(2) == real(2))
        self.assertFalse(real(3) == real(2))
        self.assertFalse(real(3) == real("NaN", 10))
        self.assertFalse(real(2) != real(2))
        self.assertTrue(real(3) != real(2))
        self.assertTrue(real(3) != real("NaN", 10))
        self.assertTrue(real(3) > real(2))
        self.assertFalse(real(2) > real(3))
        self.assertFalse(real(1) >= real(2))
        self.assertTrue(real(2) >= real(2))
        self.assertTrue(real(3) >= real(2))

    def test_unary(self):
        from . import core

        if not hasattr(core, "real"):
            return

        from . import real

        x = real("1.1", 512)
        self.assertEqual(str(x), str(+x))

        xm = real("-1.1", 512)
        self.assertEqual(str(xm), str(-x))
        self.assertEqual(str(abs(xm)), str(x))

    def test_basic(self):
        from . import core

        if not hasattr(core, "real"):
            return

        from . import real
        from . import core, real_prec_min, real_prec_max
        import numpy as np

        ld = np.longdouble
        f32 = np.float32

        self.assertGreater(real_prec_max(), real_prec_min())

        # Default ctor.
        self.assertEqual(str(real()), "0.0")
        self.assertEqual(real().prec, 2)

        # Check error if only the precision argument
        # is present.
        with self.assertRaises(ValueError) as cm:
            real(prec=32)
        self.assertTrue(
            "Cannot construct a real from a precision only - a value must also be"
            " supplied"
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
        self.assertTrue(str(x).startswith("-"))

        x = real(1 << 50)
        self.assertTrue("125899906842624" in str(x))
        self.assertFalse(str(x).startswith("-"))

        # Int that does not fit in long/long long, no precision.
        x = real(-(1 << 128))
        self.assertTrue("40282366920938463463374607431768211456" in str(x))
        self.assertTrue(str(x).startswith("-"))

        x = real(1 << 128)
        self.assertTrue("40282366920938463463374607431768211456" in str(x))
        self.assertFalse(str(x).startswith("-"))

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

        # Construction from bool.
        self.assertEqual(real(True), 1)
        self.assertEqual(real(False), 0)

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

        # float32.
        x = real(f32("1.1"))
        self.assertEqual(x.prec, np.finfo(f32).nmant + 1)

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
            "The string 'hello world' cannot be interpreted as a floating-point value"
            " in base 10"
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

        # set_prec().
        x = real("1.1", 123)
        x.set_prec(10)
        self.assertEqual(x.prec, 10)
        self.assertEqual(str(x), "nan")
        with self.assertRaises(ValueError) as cm:
            x.set_prec(0)
        self.assertTrue(
            "Cannot set the precision of a real to the value 0" in str(cm.exception)
        )
        with self.assertRaises(OverflowError) as cm:
            x.set_prec(1 << 512)

        # prec_round().
        x = real("1.1", 123)
        x.prec_round(200)
        self.assertEqual(x.prec, 200)
        self.assertEqual(x, real(real("1.1", 123), 200))
        x.prec_round(123)
        self.assertEqual(x, real("1.1", 123))
        self.assertEqual(x.prec, 123)
        with self.assertRaises(ValueError) as cm:
            x.prec_round(0)
        self.assertTrue(
            "Cannot set the precision of a real to the value 0" in str(cm.exception)
        )
        with self.assertRaises(OverflowError) as cm:
            x.prec_round(1 << 512)

        # copy/deepcopy.
        from copy import copy, deepcopy

        x = real("1.1", 123)
        y = copy(x)
        self.assertEqual(x, y)
        self.assertEqual(y.prec, 123)
        self.assertNotEqual(x._limb_address, y._limb_address)
        with self.assertRaises(TypeError) as cm:
            x.__copy__(4)
        with self.assertRaises(TypeError) as cm:
            x.__copy__(foo=4)

        y = deepcopy(x)
        self.assertEqual(x, y)
        self.assertEqual(y.prec, 123)
        self.assertNotEqual(x._limb_address, y._limb_address)

        # Check direct calls to the method.
        x.__deepcopy__(4)
        x.__deepcopy__(memo=4)

        with self.assertRaises(TypeError) as cm:
            x.__deepcopy__(mamo=4)
        self.assertTrue("mamo" in str(cm.exception))

        # Pickle support.
        import pickle

        x = real("1.1", 128)
        y = pickle.loads(pickle.dumps(x))
        self.assertEqual(x, y)
        self.assertEqual(y.prec, 128)

        # Custom caster.
        y = core._copy_real(x)
        self.assertEqual(x, y)
        self.assertEqual(y.prec, 128)

        # Try to invoke all methods of real.
        # We run this check to make sure that we have
        # implemented all mandatory methods (e.g., failing
        # to implement the __int__() method would result
        # in a segmentation fault).
        # NOTE: disable this test for the moment due to the
        # potential missing memset() issue in NumPy.
        # for s in dir(real):
        #     try:
        #         x = real()
        #         if not s in ["byteswap", "imag"]:
        #             print("Getting attr: {}".format(s))
        #             getattr(x, s)()
        #             print("Gotten attr: {}".format(s))
        #     except:
        #         pass
