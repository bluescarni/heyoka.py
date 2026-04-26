// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <string>

#include "docstrings.hpp"

namespace heyoka_py::docstrings
{

std::string eo_dynamics()
{
    return R"(eo_dynamics(*, max_geo_degree: int = 0, max_geo_order: int = 0, eop_data: eop_data = eop_data(), sw_data: sw_data = sw_data(), iau2006_thresh: float = 1e-3, Cb: expression | None = None, elp2000_thresh: float | None = None, vsop2013_thresh: float | None = None) -> list[tuple[expression, expression]]

Formulate the dynamics of an Earth-orbiting spacecraft.

.. versionadded:: 7.11.0

The dynamics is formulated in terms of the Cartesian state variables ``[x, y, z, vx, vy, vz]`` in the
`GCRS <https://en.wikipedia.org/wiki/Barycentric_and_geocentric_celestial_reference_systems>`__, with
distances measured in kilometres and time in `TT <https://en.wikipedia.org/wiki/Terrestrial_Time>`__
seconds since the epoch of J2000.

The precise formulation of the dynamics is controlled by several (optional) keyword arguments. By default (i.e., if no
arguments are passed in input), purely Keplerian dynamics is returned, with the Earth's gravitational parameter taken
from :py:func:`~heyoka.model.get_egm2008_mu()`.

Currently the dynamical model includes:

- :ref:`geopotential <tut_geopot>` via the EGM2008 model,
- :ref:`atmospheric drag <tut_eo_atmo>` via the NRLMSISE-00 model (enabled if the *Cb* argument is provided),
- :ref:`third-body perturbations <tut_3rd_body>` via the ELP2000 and VSOP2013 analytical theories (enabled if both the
  *elp2000_thresh* and *vsop2013_thresh* arguments are provided).

The *eop_data* argument is used in the formulation of the Earth's gravity and of the atmospheric drag. The *sw_data* argument
is used in the formulation of the atmospheric drag. The threshold arguments *iau2006_thresh*, *elp2000_thresh* and *vsop2013_thresh*
control the precision of the analytical theories used in the formulation of the dynamics. Please refer to the
:ref:`Earth-orbit dynamics section <tut_eo_dynamics>` in the documentation for a quantitative analysis of how these threshold
affect the accuracy of numerical integration.

:param max_geo_degree: the maximum geopotential degree.
:param max_geo_order: the maximum geopotential order.
:param eop_data: the :ref:`Earth orientation parameters <tut_eop_data>` data.
:param sw_data: the :ref:`space weather <tut_sw_data>` data.
:param iau2006_thresh: the truncation threshold for the :ref:`precession-nutation model <tut_iau2006>`.
:param Cb: the ballistic coefficient of the spacecraft, in ``m**2/kg``. If not provided, atmospheric drag is disabled.
:param elp2000_thresh: the truncation threshold for the :ref:`ELP2000 <tut_elp2000>` theory. If not provided, third-body perturbations are disabled.
:param vsop2013_thresh: the truncation threshold for the :ref:`VSOP2013 <tut_vsop2013>` theory. If not provided, third-body perturbations are disabled.

:returns: the differential equations for the state variables ``[x, y, z, vx, vy, vz]``.

:raises ValueError: if one or more input arguments are malformed.

)";
}

std::string lagrange_prop()
{
    return R"(lagrange_prop(*, pos0: typing.Iterable[expression], vel0: typing.Iterable[expression], mu: expression, tm: expression) -> tuple[list[expression], list[expression]]

Formulate a Lagrangian propagator.

.. versionadded:: 7.11.0

.. note::

   A :ref:`tutorial <tut_lagrange_prop>` explaining the use of this function is available.

Given in input:

- initial Cartesian position and velocity vectors *pos0* and *vel0*,
- a gravitational parameter *mu*,
- a propagation time *tm*,

this function will return the symbolic expressions for the propagated Cartesian position and velocity vectors, computed
according to Keplerian dynamics via the classical Lagrange (F&G) propagator.

The propagator currently assumes a bound (elliptical) orbit - the formulation will break down in case of parabolic or
hyperbolic orbits.

The lengths of *pos0* and *vel0* must both be three. The propagator is unit-agnostic: any consistent set of units for
position, velocity, gravitational parameter and time is accepted.

:param pos0: the initial Cartesian position vector.
:param vel0: the initial Cartesian velocity vector.
:param mu: the gravitational parameter.
:param tm: the propagation time.

:returns: a pair containing the symbolic expressions for the propagated Cartesian position and velocity vectors.

:raises TypeError: if *pos0* or *vel0* do not have length three.

)";
}

} // namespace heyoka_py::docstrings
