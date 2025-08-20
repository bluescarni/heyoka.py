// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PY_DOCSTRINGS_HPP
#define HEYOKA_PY_DOCSTRINGS_HPP

#include <string>

namespace heyoka_py::docstrings
{

// Expression system.
std::string make_vars();
std::string diff_tensors();
std::string expression();
std::string expression_init();
std::string subs();
std::string sum();
std::string prod();

std::string dtens();
std::string dtens_init();
std::string dtens_get_derivatives();
std::string dtens_order();
std::string dtens_index_of();
std::string dtens_args();
std::string dtens_gradient();
std::string dtens_jacobian();
std::string dtens_hessian();
std::string dtens_nouts();
std::string dtens_nargs();

std::string diff_args();
std::string diff_args_vars();
std::string diff_args_pars();
std::string diff_args_all();

// Lagrangian/Hamiltonian mechanics.
std::string lagrangian();
std::string hamiltonian();

// Models.
std::string cart2geo();
std::string geo2cart();
std::string nrlmsise00_tn();
std::string jb08_tn();
std::string fixed_centres();
std::string pendulum();
std::string sgp4();
std::string gpe_is_deep_space();
std::string delta_tdb_tt();
std::string rot_fk5j2000_icrs();
std::string rot_icrs_fk5j2000();
std::string rot_itrs_icrs(double);
std::string rot_icrs_itrs(double);
std::string rot_itrs_teme();
std::string rot_teme_itrs();
std::string era();
std::string erap();
std::string gmst82();
std::string gmst82p();
std::string pm_x();
std::string pm_xp();
std::string pm_y();
std::string pm_yp();
std::string dX();
std::string dXp();
std::string dY();
std::string dYp();
std::string iau2006(double);
std::string egm2008_pot();
std::string egm2008_acc();
std::string Ap_avg();
std::string f107();
std::string f107a_center81();
std::string vsop2013_elliptic();
std::string vsop2013_cartesian();
std::string vsop2013_cartesian_icrf();
std::string get_vsop2013_mus();
std::string elp2000_cartesian_e2000();
std::string elp2000_cartesian_fk5();
std::string get_elp2000_mus();
std::string get_egm2008_mu();
std::string get_egm2008_a();
std::string dayfrac();

// var_ode_sys() and related.
std::string var_args();
std::string var_args_vars();
std::string var_args_params();
std::string var_args_time();
std::string var_args_all();
std::string var_ode_sys();
std::string var_ode_sys_init();
std::string var_ode_sys_sys();
std::string var_ode_sys_vargs();
std::string var_ode_sys_n_orig_sv();
std::string var_ode_sys_order();

// sgp4 propagator.
std::string sgp4_propagator(const std::string &);
std::string sgp4_propagator_init(const std::string &);
std::string sgp4_propagator_jdtype(const std::string &);
std::string sgp4_propagator_nsats();
std::string sgp4_propagator_nouts();
std::string sgp4_propagator_diff_args();
std::string sgp4_propagator_diff_order();
std::string sgp4_propagator_sat_data(const std::string &, const std::string &);
std::string sgp4_propagator_get_dslice();
std::string sgp4_propagator_get_mindex(const std::string &);
std::string sgp4_propagator_call(const std::string &, const std::string &);
std::string sgp4_propagator_replace_sat_data(const std::string &, const std::string &);

// code_model enum.
std::string code_model();
std::string code_model_tiny();
std::string code_model_small();
std::string code_model_kernel();
std::string code_model_medium();
std::string code_model_large();

// eop data.
std::string eop_data();
std::string eop_data_init();
std::string eop_data_table();
std::string eop_data_timestamp();
std::string eop_data_identifier();
std::string eop_data_fetch_latest_iers_rapid();
std::string eop_data_fetch_latest_iers_long_term();

// sw data.
std::string sw_data();
std::string sw_data_init();
std::string sw_data_table();
std::string sw_data_timestamp();
std::string sw_data_identifier();
std::string sw_data_fetch_latest_celestrak();

// func_args.
std::string func_args();
std::string func_args_init();
std::string func_args_args();
std::string func_args_is_shared();

} // namespace heyoka_py::docstrings

#endif
