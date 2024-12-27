# Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from .. import core as _core
from typing import Union
import numpy

_lst = list(filter(lambda name: name.startswith("_model_"), dir(_core)))

for _name in _lst:
    exec(f"from ..core import {_name} as {_name[7:]}")
    # NOTE: we also try to change the __name__ of the current object.
    # This will work only for exposed classes, not functions, hence
    # the try/except block. We do this because otherwise sphinx
    # will document classes as aliases for the implementation detail
    # objects with the weird name.
    try:
        exec(f"{_name[7:]}.__name__ = '{_name[7:]}'")
    except AttributeError:
        pass


def sgp4_propagator(
    sat_list: Union[list, numpy.ndarray],
    diff_order: int = 0,
    **kwargs,
):
    """
    Construct an SGP4 propagator.

    .. versionadded:: 5.1.0

    .. versionadded:: 7.0.0

       This function now also accepts *sat_list* as a NumPy array.

    .. note::

       A :ref:`tutorial <tut_sgp4_propagator>` explaining the use of this function
       is available.

    This function will construct an SGP4 propagator from the input arguments.

    The only mandatory argument is *sat_list*, which must be either a list
    of general perturbations element sets (GPEs) represented as ``Satrec`` objects from the
    `sgp4 Python module <https://pypi.org/project/sgp4/>`__, or a 2D array.

    In the former case, the GPE data is taken directly from the ``Satrec`` objects.
    In the latter case, *sat_list* is expected to be a 9 x ``n`` C-style contiguous
    array, where ``n`` is the total number of satellites and the rows contain the following
    GPE data:

    0. the mean motion (in [rad / min]),
    1. the eccentricity,
    2. the inclination (in [rad]),
    3. the right ascension of the ascending node (in [rad]),
    4. the argument of perigee (in [rad]),
    5. the mean anomaly (in [rad]),
    6. the `BSTAR <https://en.wikipedia.org/wiki/BSTAR>`__ drag term (in the same unit as given in the GPE),
    7. the reference epoch (as a Julian date),
    8. a fractional correction to the epoch (in Julian days).

    When *sat_list* is a list of ``Satrec`` objects, the GPE epochs are represented as UTC Julian dates,
    and consequently UTC Julian dates must also be used during propagation. Please note
    that the use of UTC Julian dates as a scale of time will produce slightly incorrect results when
    propagating across leap seconds, as explained in the :ref:`tutorial<tut_sgp4_propagator_epochs>`.

    If *sat_list* is a 2D array, the epochs must be provided as Julian dates in the terrestrial time scale.

    The *diff_order* argument indicates the desired differentiation order. If equal to 0, then
    derivatives are disabled.

    The :ref:`fp_type keyword argument <api_common_kwargs_fp_type>` can be passed in *kwargs*
    to select the precision of the propagator (double-precision is the default, single-precision
    is also supported).

    *kwargs* can also optionally contain keyword arguments from the :ref:`api_common_kwargs_llvm` set
    and the :ref:`api_common_kwargs_cfunc` set.

    :param sat_list: the GPE data.
    :param diff_order: the derivatives order.

    :raises TypeError: if an unsupported :ref:`fp_type <api_common_kwargs_fp_type>` is specified.
    :raises: any exception raised by the construction of the propagator.

    :returns: an SGP4 propagator object.
    :rtype: sgp4_propagator_dbl | sgp4_propagator_flt

    """
    from .. import _fp_to_suffix_dict, core

    fp_type = kwargs.pop("fp_type", float)

    try:
        fp_suffix = _fp_to_suffix_dict[fp_type]
    except KeyError:
        raise TypeError(f"Unknown fp type '{fp_type}'")

    if hasattr(core, f"_model_sgp4_propagator{fp_suffix}"):
        return getattr(core, f"_model_sgp4_propagator{fp_suffix}")(
            sat_list, diff_order, **kwargs
        )
    else:
        raise TypeError(f"No sgp4 propagator available for the fp type '{fp_type}'")


del _core, _lst, _name, numpy, Union
