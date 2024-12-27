# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class sgp4_propagator_test_case(_ut.TestCase):
    # Couple of TLEs downloaded from SpaceTrack.
    s1 = "1 00045U 60007A   24187.45810325  .00000504  00000-0  14841-3 0  9992"
    t1 = "2 00045  66.6943  81.3521 0257384 317.3173  40.8180 14.34783636277898"

    s2 = "1 00046U 60007B   24187.58585517  .00002292  00000-0  46268-3 0  9993"
    t2 = "2 00046  66.6912   4.9995 0194928 108.0242 254.2215 14.52078523353568"

    def test_basics(self):
        try:
            from sgp4.api import Satrec
        except ImportError:
            return

        from .model import sgp4_propagator
        from pickle import loads, dumps
        from copy import copy, deepcopy
        from . import make_vars, code_model
        import numpy as np

        s1 = sgp4_propagator_test_case.s1
        t1 = sgp4_propagator_test_case.t1

        s2 = sgp4_propagator_test_case.s2
        t2 = sgp4_propagator_test_case.t2

        sat1 = Satrec.twoline2rv(s1, t1)
        sat2 = Satrec.twoline2rv(s2, t2)

        with self.assertRaises(TypeError) as cm:
            sgp4_propagator([sat1, 1])
        self.assertTrue(
            "Invalid object encountered in the satellite data for an sgp4 propagator: a"
            " list of sgp4 Satrec objects is expected, but an object of type '<class"
            " 'int'>' was detected instead at index 1"
            in str(cm.exception)
        )

        prop = sgp4_propagator([sat1, sat2])

        self.assertEqual(prop.nsats, 2)
        self.assertEqual(prop.nouts, 7)
        self.assertEqual(prop.diff_order, 0)

        with self.assertRaises(ValueError) as cm:
            prop.diff_args
        self.assertTrue(
            "The function 'get_diff_args()' cannot be invoked on an sgp4 propagator"
            " without derivatives"
            in str(cm.exception)
        )

        self.assertEqual(prop.sat_data.shape, (9, 2))
        self.assertEqual(prop.sat_data[6, 0], 0.14841e-3)
        self.assertEqual(prop.sat_data[6, 1], 0.46268e-3)

        # Check that we cannot write into sat_data.
        with self.assertRaises(ValueError) as cm:
            prop.sat_data[0, 0] = 0

        # Check that we cannot fetch diff properties if derivatives
        # are not requested.
        with self.assertRaises(ValueError) as cm:
            prop.get_dslice(order=1)
        self.assertTrue(
            "The function 'get_dslice()' cannot be invoked on an sgp4 propagator"
            " without derivatives"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            prop.get_mindex(i=0)
        self.assertTrue(
            "The function 'get_mindex()' cannot be invoked on an sgp4 propagator"
            " without derivatives"
            in str(cm.exception)
        )

        self.assertTrue("SGP4 propagator" in repr(prop))
        self.assertTrue("N of satellites: 2" in repr(prop))

        prop.foobar = [1, 2, 3]
        cprop = copy(prop)
        self.assertEqual(cprop.foobar, [1, 2, 3])
        self.assertEqual(id(cprop.foobar), id(prop.foobar))
        self.assertEqual(cprop.sat_data.shape, (9, 2))
        self.assertEqual(cprop.sat_data[6, 0], 0.14841e-3)
        self.assertEqual(cprop.sat_data[6, 1], 0.46268e-3)
        cprop = deepcopy(prop)
        self.assertEqual(cprop.foobar, [1, 2, 3])
        self.assertNotEqual(id(cprop.foobar), id(prop.foobar))
        self.assertEqual(cprop.sat_data.shape, (9, 2))
        self.assertEqual(cprop.sat_data[6, 0], 0.14841e-3)
        self.assertEqual(cprop.sat_data[6, 1], 0.46268e-3)

        cprop = loads(dumps(prop))
        self.assertEqual(cprop.foobar, [1, 2, 3])
        self.assertNotEqual(id(cprop.foobar), id(prop.foobar))
        self.assertEqual(cprop.sat_data.shape, (9, 2))
        self.assertEqual(cprop.sat_data[6, 0], 0.14841e-3)
        self.assertEqual(cprop.sat_data[6, 1], 0.46268e-3)

        # A couple of tests with the derivatives and custom llvm settings.
        prop = sgp4_propagator(
            [sat1, sat2],
            diff_order=1,
            parjit=False,
            compact_mode=True,
            code_model=code_model.large,
        )
        self.assertEqual(
            prop.diff_args,
            make_vars("n0", "e0", "i0", "node0", "omega0", "m0", "bstar"),
        )
        self.assertEqual(prop.get_dslice(order=1), slice(7, 56, None))
        self.assertEqual(prop.get_mindex(7), [0, 1, 0, 0, 0, 0, 0, 0])

        # Test with GPE data passed in as array.
        with self.assertRaises(ValueError) as cm:
            sgp4_propagator(np.zeros((2, 2), dtype=float))
        self.assertTrue(
            "The array of input GPEs for an sgp4 propagator must have 9 rows, but the"
            " supplied array has 2 row(s) instead"
            in str(cm.exception)
        )
        with self.assertRaises(ValueError) as cm:
            sgp4_propagator(np.zeros((2,), dtype=float))
        self.assertTrue(
            "The array of input GPEs for an sgp4 propagator must have 2 dimensions, but"
            " the supplied array has 1 dimension(s) instead"
            in str(cm.exception)
        )
        with self.assertRaises(ValueError) as cm:
            sgp4_propagator(np.zeros((18, 2), dtype=float)[::2])
        self.assertTrue(
            "Invalid array of input GPEs detected in an sgp4 propagator: the array is"
            " not C-style contiguous, please consider using numpy.ascontiguousarray()"
            " to turn it into one"
            in str(cm.exception)
        )

        # Construct a GPE data array from the satellites sat1 and sat2.
        sat_attrs = [
            "no_kozai",
            "ecco",
            "inclo",
            "nodeo",
            "argpo",
            "mo",
            "bstar",
            "jdsatepoch",
            "jdsatepochF",
        ]
        gpe_data = np.ascontiguousarray(
            np.transpose(
                np.array(
                    [
                        [getattr(sat, name) for name in sat_attrs]
                        for sat in [sat1, sat2]
                    ],
                    dtype=float,
                )
            )
        )

        # Init the propagator and verify its gpe data matches prop.
        prop2 = sgp4_propagator(gpe_data)
        self.assertTrue(np.all(prop.sat_data == prop2.sat_data))

    def test_propagation(self):
        try:
            from sgp4.api import Satrec
        except ImportError:
            return

        import numpy as np
        from .model import sgp4_propagator

        s1 = sgp4_propagator_test_case.s1
        t1 = sgp4_propagator_test_case.t1

        s2 = sgp4_propagator_test_case.s2
        t2 = sgp4_propagator_test_case.t2

        sat1 = Satrec.twoline2rv(s1, t1)
        sat2 = Satrec.twoline2rv(s2, t2)

        for fp_type in [float, np.single]:
            prop = sgp4_propagator([sat1, sat2], diff_order=1, fp_type=fp_type)

            # Error checking.
            with self.assertRaises(ValueError) as cm:
                prop(np.zeros((), dtype=fp_type))
            self.assertTrue(
                "A times/dates array with 1 or 2 dimensions is expected as an input for"
                " the call operator of an sgp4 propagator, but an array with 0"
                " dimensions was provided instead"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                prop(np.zeros((0, 0, 0), dtype=fp_type))
            self.assertTrue(
                "A times/dates array with 1 or 2 dimensions is expected as an input for"
                " the call operator of an sgp4 propagator, but an array with 3"
                " dimensions was provided instead"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                prop(np.zeros((1, 5), dtype=fp_type))
            self.assertTrue(
                "Invalid times/dates array detected as an input for the call operator"
                " of an sgp4 propagator in batch mode: the number of satellites"
                " inferred from the times/dates array is 5, but the propagator contains"
                " 2 satellite(s) instead"
                in str(cm.exception)
            )

            with self.assertRaises(ValueError) as cm:
                prop(np.zeros((1,), dtype=fp_type))
            self.assertTrue(
                "Invalid times/dates array detected as an input for the call operator"
                " of an sgp4 propagator: the number of satellites inferred from the"
                " times/dates array is 1, but the propagator contains 2 satellite(s)"
                " instead"
                in str(cm.exception)
            )

            tmp = np.zeros((10, 2), dtype=fp_type)
            tmp2 = tmp[::2, :]

            with self.assertRaises(ValueError) as cm:
                prop(tmp2)
            self.assertTrue(
                "Invalid times/dates array detected as an input for the call operator"
                " of an sgp4 propagator: the array is not C-style contiguous"
                in str(cm.exception)
            )

            # Invalid output provided.
            with self.assertRaises(ValueError) as cm:
                prop(
                    np.zeros((2,), dtype=fp_type),
                    out=np.zeros((100,), dtype=fp_type)[::2],
                )
            self.assertTrue(
                "Invalid output array detected in the call operator of "
                "an sgp4 propagator: the array is not C-style contiguous and writeable"
                in str(cm.exception)
            )

            out = np.zeros(
                (
                    49,
                    2,
                ),
                dtype=fp_type,
            )
            out.flags.writeable = False
            with self.assertRaises(ValueError) as cm:
                prop(
                    np.zeros((2,), dtype=fp_type),
                    out=out,
                )
            self.assertTrue(
                "Invalid output array detected in the call operator of "
                "an sgp4 propagator: the array is not C-style contiguous and writeable"
                in str(cm.exception)
            )

            # Memory overlap.
            out = np.zeros((2,), dtype=fp_type)
            with self.assertRaises(ValueError) as cm:
                prop(
                    out,
                    out=out,
                )
            self.assertTrue(
                "Invalid input/output arrays detected in the call operator of "
                "an sgp4 propagator: the input/outputs arrays may overlap"
                in str(cm.exception)
            )

            # Scalar mode errors.
            out = np.zeros(
                (49,),
                dtype=fp_type,
            )
            with self.assertRaises(ValueError) as cm:
                prop(
                    np.zeros((2,), dtype=fp_type),
                    out=out,
                )
            self.assertTrue(
                "Invalid output array detected in the call operator of "
                "an sgp4 propagator: the array has 1 dimension(s), "
                "but 2 dimensions are expected instead"
                in str(cm.exception)
            )

            out = np.zeros(
                (48, 2),
                dtype=fp_type,
            )
            with self.assertRaises(ValueError) as cm:
                prop(
                    np.zeros((2,), dtype=fp_type),
                    out=out,
                )
            self.assertTrue(
                "Invalid output array detected in the call operator of "
                "an sgp4 propagator: the first dimension has a "
                "size of 48, but a "
                "size of 56 (i.e., equal to the number of outputs for each "
                "propagation) is required instead"
                in str(cm.exception)
            )

            out = np.zeros(
                (56, 1),
                dtype=fp_type,
            )
            with self.assertRaises(ValueError) as cm:
                prop(
                    np.zeros((2,), dtype=fp_type),
                    out=out,
                )
            self.assertTrue(
                "Invalid output array detected in the call operator of "
                "an sgp4 propagator: the second dimension has a "
                "size of 1, but a "
                "size of 2 (i.e., equal to the total number of satellites) is "
                "required instead"
                in str(cm.exception)
            )

            # Batch mode errors.
            out = np.zeros(
                (56, 1),
                dtype=fp_type,
            )
            with self.assertRaises(ValueError) as cm:
                prop(
                    np.zeros((10, 2), dtype=fp_type),
                    out=out,
                )
            self.assertTrue(
                "Invalid output array detected in the call operator of "
                "an sgp4 propagator in batch mode: the array has 2 dimension(s), "
                "but 3 dimensions are expected instead"
                in str(cm.exception)
            )

            out = np.zeros(
                (9, 56, 1),
                dtype=fp_type,
            )
            with self.assertRaises(ValueError) as cm:
                prop(
                    np.zeros((10, 2), dtype=fp_type),
                    out=out,
                )
            self.assertTrue(
                "Invalid output array detected in the call operator of an sgp4"
                " propagator in batch mode: the first dimension has a size of 9, but a"
                " size of 10 (i.e., equal to the number of evaluations) is required"
                " instead"
                in str(cm.exception)
            )

            out = np.zeros(
                (10, 48, 1),
                dtype=fp_type,
            )
            with self.assertRaises(ValueError) as cm:
                prop(
                    np.zeros((10, 2), dtype=fp_type),
                    out=out,
                )
            self.assertTrue(
                "Invalid output array detected in the call operator of "
                "an sgp4 propagator in batch mode: the second dimension has a "
                "size of 48, but a "
                "size of 56 (i.e., equal to the number of outputs for each "
                "propagation) is required instead"
                in str(cm.exception)
            )

            out = np.zeros(
                (10, 56, 0),
                dtype=fp_type,
            )
            with self.assertRaises(ValueError) as cm:
                prop(
                    np.zeros((10, 2), dtype=fp_type),
                    out=out,
                )
            self.assertTrue(
                "Invalid output array detected in the call operator of "
                "an sgp4 propagator in batch mode: the third dimension has a "
                "size of 0, but a "
                "size of 2 (i.e., equal to the total number of satellites) is "
                "required instead"
                in str(cm.exception)
            )

            # Run a simple propagation to the TLE epoch, check
            # that all derivatives wrt the error codes are zero.
            out = np.zeros(
                (56, 2),
                dtype=fp_type,
            )
            prop(
                np.zeros((2,), dtype=fp_type),
                out=out,
            )
            self.assertTrue(np.all(out[49:, :] == 0.0))

            # Same with a jdate.
            dates = np.zeros((2,), dtype=prop.jdtype)
            dates["jd"] = 2460496.5
            dates["frac"] = 0.5833449099999939
            prop(
                dates,
                out=out,
            )
            self.assertTrue(np.all(out[49:, :] == 0.0))

            # Test that a batch prop with zero nevals does not throw.
            dates = np.zeros((0, 2), dtype=prop.jdtype)
            out = np.zeros(
                (0, 56, 2),
                dtype=fp_type,
            )
            prop(
                dates,
                out=out,
            )

    def test_skyfield_comp(self):
        try:
            from skyfield.api import load
            from skyfield.iokit import parse_tle_file
        except ImportError:
            return

        from .model import sgp4_propagator
        from ._sgp4_test_data import sgp4_test_tle
        import math, numpy as np
        from sgp4.api import SatrecArray

        # Load the test dataset.
        ts = load.timescale()
        satellites = list(
            parse_tle_file((bytes(_, "ascii") for _ in sgp4_test_tle.split("\n")), ts)
        )

        # Filter out high-altitude satellites
        sats = list(
            filter(
                lambda sat: 2 * math.pi / sat.no_kozai < 225,
                (_.model for _ in satellites),
            )
        )

        # Create the propagator.
        prop = sgp4_propagator(sats)

        # Create the dates array.
        dates = np.zeros((prop.nsats,), dtype=prop.jdtype)

        # Pick a date.
        jd = 2460496.5
        jd_frac = 0.5833449099999939

        # Run the test between one month before and after the
        # reference date.
        for delta_day in range(-30, 30, 2):
            dates["jd"] = jd + delta_day
            dates["frac"] = jd_frac

            # Compute the state vectors.
            sv = prop(dates)

            # Compute them with the sgp4 module.
            sat_arr = SatrecArray(sats)
            e, r, v = sat_arr.sgp4(np.array([jd + delta_day]), np.array([jd_frac]))

            # Mask out the satellites that:
            # - generated an error code, or
            # - ended up farther than 8000km from the Earth, or
            # - contain non-finite positional data.
            mask = np.logical_and.reduce(
                (
                    e[:, 0] == 0,
                    np.linalg.norm(r[:, 0, :], axis=1) < 8000,
                    np.all(np.isfinite(r[:, 0, :]), axis=1),
                )
            )

            # Compute the positional errors in meters, and sort them.
            err = np.sort(np.linalg.norm(r[mask, 0] - sv[:3, mask].T, axis=1) * 1e3)

            # Check that the largest error is less than 1mm.
            self.assertTrue(err[-1] < 1e-3)

            # Check the error codes.
            self.assertTrue(np.all(e[:, 0] == sv[-1, :].T))

    def test_replace_sat_data(self):
        try:
            from sgp4.api import Satrec
        except ImportError:
            return

        from copy import deepcopy
        from .model import sgp4_propagator
        import numpy as np

        s1 = sgp4_propagator_test_case.s1
        t1 = sgp4_propagator_test_case.t1

        s2 = sgp4_propagator_test_case.s2
        t2 = sgp4_propagator_test_case.t2

        sat1 = Satrec.twoline2rv(s1, t1)
        sat2 = Satrec.twoline2rv(s2, t2)

        prop = sgp4_propagator([sat1, sat2])
        orig_prop = deepcopy(prop)

        orig_sat_data = deepcopy(prop.sat_data)

        prop.replace_sat_data([sat2, sat1])

        self.assertTrue(np.all(prop.sat_data[:, 1] == orig_sat_data[:, 0]))
        self.assertTrue(np.all(prop.sat_data[:, 0] == orig_sat_data[:, 1]))

        new_sat_data = deepcopy(prop.sat_data)

        with self.assertRaises(TypeError) as cm:
            prop.replace_sat_data([sat1, 1])
        self.assertTrue(
            "Invalid object encountered in the satellite data for an sgp4 propagator: a"
            " list of sgp4 Satrec objects is expected, but an object of type '<class"
            " 'int'>' was detected instead at index 1"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            prop.replace_sat_data([sat1])
        self.assertTrue(
            "Invalid array provided to replace_sat_data(): the number of "
            "columns (1) does not match the number of satellites (2)"
            in str(cm.exception)
        )

        self.assertTrue(np.all(prop.sat_data == new_sat_data))

        # Check also the overload with gpe data as a numpy array.

        # Construct a GPE data array from the satellites sat1 and sat2.
        sat_attrs = [
            "no_kozai",
            "ecco",
            "inclo",
            "nodeo",
            "argpo",
            "mo",
            "bstar",
            "jdsatepoch",
            "jdsatepochF",
        ]
        gpe_data = np.ascontiguousarray(
            np.transpose(
                np.array(
                    [
                        [getattr(sat, name) for name in sat_attrs]
                        for sat in [sat1, sat2]
                    ],
                    dtype=float,
                )
            )
        )

        # Init the propagator and verify its gpe data matches prop.
        prop.replace_sat_data(gpe_data)
        self.assertTrue(np.all(prop.sat_data == orig_prop.sat_data))
