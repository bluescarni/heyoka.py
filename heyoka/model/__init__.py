# Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Imports from core.
from .._core import (
    Ap_avg,
    Ap_avgp,
    cart2geo,
    cr3bp,
    cr3bp_jacobi,
    dX,
    dXp,
    dY,
    dYp,
    dayfrac,
    delta_tdb_tt,
    egm2008_acc,
    egm2008_pot,
    elp2000_cartesian_e2000,
    elp2000_cartesian_fk5,
    eo_dynamics,
    era,
    erap,
    f107,
    f107a_center81,
    f107a_center81p,
    f107p,
    ffnn,
    fixed_centres,
    fixed_centres_energy,
    fixed_centres_potential,
    geo2cart,
    get_egm2008_CS,
    get_egm2008_a,
    get_egm2008_mu,
    get_elp2000_mus,
    get_vsop2013_mus,
    gmst82,
    gmst82p,
    gpe_is_deep_space,
    iau2006,
    jb08_tn,
    lagrange_prop,
    mascon,
    mascon_energy,
    mascon_potential,
    nbody,
    nbody_energy,
    nbody_potential,
    np1body,
    np1body_energy,
    np1body_potential,
    nrlmsise00_tn,
    pendulum,
    pendulum_energy,
    pm_x,
    pm_xp,
    pm_y,
    pm_yp,
    rot_fk5j2000_icrs,
    rot_icrs_fk5j2000,
    rot_icrs_itrs,
    rot_itrs_icrs,
    rot_itrs_teme,
    rot_teme_itrs,
    rotating,
    rotating_energy,
    rotating_potential,
    sgp4,
    sgp4_propagator_dbl,
    sgp4_propagator_flt,
    sh_gravity_acc,
    sh_gravity_pot,
    state_from_rsw,
    state_from_rsw_inertial,
    state_to_rsw,
    state_to_rsw_inertial,
    vsop2013_cartesian,
    vsop2013_cartesian_icrf,
    vsop2013_elliptic,
)

from ._sgp4_propagator import sgp4_propagator
from .. import _core

# NOTE: these global attributes need to be defined directly in this file - if we
# define them in a separate file and then import them, sphinx documentation is not
# properly built.

delta_tt_tai: _core.expression = _core.delta_tt_tai
"""
Difference between TT and TAI.

.. versionadded:: 7.3.0

This expression is a constant representing the difference between `terrestrial time (TT) <https://en.wikipedia.org/wiki/Terrestrial_Time>`__
and `international atomic time (TAI) <https://en.wikipedia.org/wiki/International_Atomic_Time>`__.

This difference amounts to exactly 32.184 SI seconds.

"""

__all__ = [
    # Core imports.
    "Ap_avg",
    "Ap_avgp",
    "cart2geo",
    "cr3bp",
    "cr3bp_jacobi",
    "dX",
    "dXp",
    "dY",
    "dYp",
    "dayfrac",
    "delta_tdb_tt",
    "egm2008_acc",
    "egm2008_pot",
    "elp2000_cartesian_e2000",
    "elp2000_cartesian_fk5",
    "eo_dynamics",
    "era",
    "erap",
    "f107",
    "f107a_center81",
    "f107a_center81p",
    "f107p",
    "ffnn",
    "fixed_centres",
    "fixed_centres_energy",
    "fixed_centres_potential",
    "geo2cart",
    "get_egm2008_CS",
    "get_egm2008_a",
    "get_egm2008_mu",
    "get_elp2000_mus",
    "get_vsop2013_mus",
    "gmst82",
    "gmst82p",
    "gpe_is_deep_space",
    "iau2006",
    "jb08_tn",
    "lagrange_prop",
    "mascon",
    "mascon_energy",
    "mascon_potential",
    "nbody",
    "nbody_energy",
    "nbody_potential",
    "np1body",
    "np1body_energy",
    "np1body_potential",
    "nrlmsise00_tn",
    "pendulum",
    "pendulum_energy",
    "pm_x",
    "pm_xp",
    "pm_y",
    "pm_yp",
    "rot_fk5j2000_icrs",
    "rot_icrs_fk5j2000",
    "rot_icrs_itrs",
    "rot_itrs_icrs",
    "rot_itrs_teme",
    "rot_teme_itrs",
    "rotating",
    "rotating_energy",
    "rotating_potential",
    "sgp4",
    "sgp4_propagator_dbl",
    "sgp4_propagator_flt",
    "sh_gravity_acc",
    "sh_gravity_pot",
    "state_from_rsw",
    "state_from_rsw_inertial",
    "state_to_rsw",
    "state_to_rsw_inertial",
    "vsop2013_cartesian",
    "vsop2013_cartesian_icrf",
    "vsop2013_elliptic",
    # Globals.
    "delta_tt_tai",
    # SGP4 propagator.
    "sgp4_propagator",
]
