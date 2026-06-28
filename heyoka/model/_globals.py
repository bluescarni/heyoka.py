# Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from .. import expression
from .._core import delta_tt_tai

delta_tt_tai: expression = delta_tt_tai
"""
Difference between TT and TAI.

.. versionadded:: 7.3.0

This expression is a constant representing the difference between `terrestrial time (TT) <https://en.wikipedia.org/wiki/Terrestrial_Time>`__
and `international atomic time (TAI) <https://en.wikipedia.org/wiki/International_Atomic_Time>`__.

This difference amounts to exactly 32.184 SI seconds.

"""
