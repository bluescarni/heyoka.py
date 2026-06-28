# Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from numpy import dtype
from . import _core


par = _core._par_generator()
"""
Parameter factory.

This global object is used to create :py:class:`~heyoka.expression` objects
representing :ref:`runtime parameters <runtime_param>`. The parameter index
must be passed to the index operator of the factory object.

Examples:
  >>> from heyoka import par
  >>> p0 = par[0] # p0 will represent the parameter value at index 0

"""


time: _core.expression = _core._time
"""
Time expression.

This global object is an :py:class:`~heyoka.expression` which is used to represent
time (i.e., the independent variable) in righ-hand side of differential equations.

"""

eop_data_row: dtype = _core.eop_data_row
"""
EOP data row.

.. versionadded:: 7.3.0

This is a :ref:`structured NumPy datatype<numpy:defining-structured-types>` used to represent
a row of EOP data in the :py:class:`~heyoka.eop_data` class. The fields in the datatype are:

- the UTC MJD,
- the UT1-UTC difference (in seconds),
- the :math:`x` component of the polar motion (in arcsecs),
- the :math:`y` component of the polar motion (in arcsecs),
- the :math:`x` component of the correction to the IAU 2000/2006
  precession/nutation model (in milliarcsecs),
- the :math:`y` component of the correction to the IAU 2000/2006
  precession/nutation model (in milliarcsecs).

"""

sw_data_row: dtype = _core.sw_data_row
"""
Space weather data row.

.. versionadded:: 7.3.0

This is a :ref:`structured NumPy datatype<numpy:defining-structured-types>` used to represent
a row of space weather (SW) data in the :py:class:`~heyoka.sw_data` class. The fields in the datatype are:

- the UTC MJD,
- the arithmetic average of the 8 Ap indices for the day,
- the observed 10.7-cm solar radio flux (F10.7),
- the 81-day arithmetic average of the observed F10.7 centred on the day.

"""
