// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/core.h>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Python.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/eop_data.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/models.hpp>
#include <heyoka/sw_data.hpp>

#include "common_utils.hpp"
#include "custom_casters.hpp"
#include "docstrings.hpp"
#include "dtypes.hpp"
#include "expose_models.hpp"

namespace heyoka_py
{

namespace detail
{

namespace
{

// Small helper to turn the variant v into an expression.
constexpr auto ex_from_variant
    = [](const auto &v) { return std::visit([](const auto &v) { return heyoka::expression(v); }, v); };

// Common logic to expose N-body helpers.
template <typename Op, typename V>
auto nbody_impl(const Op &op, std::uint32_t n, const V &Gconst, const std::optional<std::vector<V>> &masses)
{
    namespace hy = heyoka;

    const auto Gval = ex_from_variant(Gconst);

    if (masses) {
        std::vector<hy::expression> masses_vec;
        std::transform(masses->begin(), masses->end(), std::back_inserter(masses_vec), ex_from_variant);

        return op(n, hy::kw::Gconst = Gval, hy::kw::masses = masses_vec);
    } else {
        return op(n, hy::kw::Gconst = Gval);
    }
}

// Common logic to expose pendulum helpers.
template <typename Op, typename V>
auto pendulum_impl(const Op &op, const V &gconst, const V &l)
{
    namespace hy = heyoka;

    const auto gval = ex_from_variant(gconst);
    const auto lval = ex_from_variant(l);

    return op(hy::kw::gconst = gval, hy::kw::length = lval);
}

// Common logic to expose rotating rf helpers.
template <typename Op, typename V>
auto rotating_impl(const Op &op, const std::vector<V> &omega)
{
    namespace hy = heyoka;

    // Build the argument for the C++ function.
    std::vector<hy::expression> omega_vec;
    std::transform(omega.begin(), omega.end(), std::back_inserter(omega_vec), ex_from_variant);

    return op(hy::kw::omega = omega_vec);
}

// Common logic for building arguments to the fixed centres and mascon models.
template <typename V>
auto mascon_fc_common_args(const char *name, const V &Gconst, const std::vector<V> &masses,
                           const py::iterable &positions_)
{
    namespace hy = heyoka;

    // Attempt to convert the input position argument into
    // a NumPy array.
    py::array positions = positions_;

    // Check the shape of positions.
    if (positions.ndim() != 2) {
        py_throw(PyExc_ValueError, fmt::format("Invalid positions array in a {} model: the number of "
                                               "dimensions must be 2, but it is {} instead",
                                               name, positions.ndim())
                                       .c_str());
    }

    if (positions.shape(1) != 3) {
        py_throw(PyExc_ValueError, fmt::format("Invalid positions array in a {} model: the number of "
                                               "columns must be 3, but it is {} instead",
                                               name, positions.shape(1))
                                       .c_str());
    }

    // Flatten out postions into a list and convert to a vector of variants.
    py::list pflat = positions.attr("flatten")();
    std::vector<V> pvec;
    try {
        pvec = pflat.cast<std::vector<V>>();
    } catch (const py::cast_error &) {
        py_throw(
            PyExc_TypeError,
            fmt::format(
                "The positions array in a {} model could not be converted into an array "
                "of expressions - please make sure that the array's values can be converted into heyoka expressions",
                name)
                .c_str());
    }

    // Build the arguments for the C++ function.
    auto Gval = ex_from_variant(Gconst);
    std::vector<hy::expression> masses_vec, positions_vec;
    std::transform(masses.begin(), masses.end(), std::back_inserter(masses_vec), ex_from_variant);
    std::transform(pvec.begin(), pvec.end(), std::back_inserter(positions_vec), ex_from_variant);

    return std::tuple{std::move(Gval), std::move(masses_vec), std::move(positions_vec)};
}

// Common logic to expose fixed centres helpers.
template <typename Op, typename V>
auto fixed_centres_impl(const Op &op, const V &Gconst, const std::vector<V> &masses, const py::iterable &positions)
{
    namespace hy = heyoka;

    const auto [Gval, masses_vec, positions_vec] = mascon_fc_common_args("fixed centres", Gconst, masses, positions);

    return op(hy::kw::Gconst = Gval, hy::kw::masses = masses_vec, hy::kw::positions = positions_vec);
}

// Common logic to expose mascon helpers.
template <typename Op, typename V>
auto mascon_impl(const Op &op, const V &Gconst, const std::vector<V> &masses, const py::iterable &positions,
                 const std::vector<V> &omega)
{
    namespace hy = heyoka;

    const auto [Gval, masses_vec, positions_vec] = mascon_fc_common_args("mascon", Gconst, masses, positions);

    // Build the omega argument.
    std::vector<hy::expression> omega_vec;
    std::transform(omega.begin(), omega.end(), std::back_inserter(omega_vec), ex_from_variant);

    return op(hy::kw::Gconst = Gval, hy::kw::masses = masses_vec, hy::kw::positions = positions_vec,
              hy::kw::omega = omega_vec);
}

// Common logic to expose cr3bp helpers.
template <typename Op, typename V>
auto cr3bp_impl(const Op &op, const V &mu)
{
    namespace hy = heyoka;

    const auto muval = ex_from_variant(mu);

    return op(hy::kw::mu = muval);
}

} // namespace

} // namespace detail

namespace py = pybind11;

void expose_models(py::module_ &m)
{
    namespace hy = heyoka;
    using namespace pybind11::literals;

    // A variant containing either an expression or a numerical/string
    // type from which an expression can be constructed.
    using vex_t = std::variant<hy::expression, std::string, float, double, long double
#if defined(HEYOKA_HAVE_REAL128)
                               ,
                               mppp::real128

#endif
#if defined(HEYOKA_HAVE_REAL)
                               ,
                               mppp::real
#endif
                               >;

    // N-body.
    m.def(
        "_model_nbody",
        [](std::uint32_t n, const vex_t &Gconst, const std::optional<std::vector<vex_t>> &masses) {
            return detail::nbody_impl(hy::model::nbody, n, Gconst, masses);
        },
        "n"_a.noconvert(), "Gconst"_a.noconvert() = 1., "masses"_a.noconvert() = py::none{});
    m.def(
        "_model_nbody_energy",
        [](std::uint32_t n, const vex_t &Gconst, const std::optional<std::vector<vex_t>> &masses) {
            return detail::nbody_impl(hy::model::nbody_energy, n, Gconst, masses);
        },
        "n"_a.noconvert(), "Gconst"_a.noconvert() = 1., "masses"_a.noconvert() = py::none{});
    m.def(
        "_model_nbody_potential",
        [](std::uint32_t n, const vex_t &Gconst, const std::optional<std::vector<vex_t>> &masses) {
            return detail::nbody_impl(hy::model::nbody_potential, n, Gconst, masses);
        },
        "n"_a.noconvert(), "Gconst"_a.noconvert() = 1., "masses"_a.noconvert() = py::none{});
    m.def(
        "_model_np1body",
        [](std::uint32_t n, const vex_t &Gconst, const std::optional<std::vector<vex_t>> &masses) {
            return detail::nbody_impl(hy::model::np1body, n, Gconst, masses);
        },
        "n"_a.noconvert(), "Gconst"_a.noconvert() = 1., "masses"_a.noconvert() = py::none{});
    m.def(
        "_model_np1body_energy",
        [](std::uint32_t n, const vex_t &Gconst, const std::optional<std::vector<vex_t>> &masses) {
            return detail::nbody_impl(hy::model::np1body_energy, n, Gconst, masses);
        },
        "n"_a.noconvert(), "Gconst"_a.noconvert() = 1., "masses"_a.noconvert() = py::none{});
    m.def(
        "_model_np1body_potential",
        [](std::uint32_t n, const vex_t &Gconst, const std::optional<std::vector<vex_t>> &masses) {
            return detail::nbody_impl(hy::model::np1body_potential, n, Gconst, masses);
        },
        "n"_a.noconvert(), "Gconst"_a.noconvert() = 1., "masses"_a.noconvert() = py::none{});

    // Pendulum.
    m.def(
        "_model_pendulum",
        [](const vex_t &gconst, const vex_t &l) { return detail::pendulum_impl(hy::model::pendulum, gconst, l); },
        "gconst"_a.noconvert() = 1., "length"_a.noconvert() = 1., docstrings::pendulum().c_str());
    m.def(
        "_model_pendulum_energy",
        [](const vex_t &gconst, const vex_t &l) {
            return detail::pendulum_impl(hy::model::pendulum_energy, gconst, l);
        },
        "gconst"_a.noconvert() = 1., "length"_a.noconvert() = 1.);

    // Fixed centres.
    m.def(
        "_model_fixed_centres",
        [](const vex_t &Gconst, const std::vector<vex_t> &masses, const py::iterable &positions) {
            return detail::fixed_centres_impl(hy::model::fixed_centres, Gconst, masses, positions);
        },
        "Gconst"_a.noconvert() = 1., "masses"_a.noconvert() = py::list{},
        "positions"_a = py::array{py::dtype(get_dtype<double>()), py::array::ShapeContainer{0, 3}},
        docstrings::fixed_centres().c_str());
    m.def(
        "_model_fixed_centres_energy",
        [](const vex_t &Gconst, const std::vector<vex_t> &masses, const py::iterable &positions) {
            return detail::fixed_centres_impl(hy::model::fixed_centres_energy, Gconst, masses, positions);
        },
        "Gconst"_a.noconvert() = 1., "masses"_a.noconvert() = py::list{},
        "positions"_a = py::array{py::dtype(get_dtype<double>()), py::array::ShapeContainer{0, 3}});
    m.def(
        "_model_fixed_centres_potential",
        [](const vex_t &Gconst, const std::vector<vex_t> &masses, const py::iterable &positions) {
            return detail::fixed_centres_impl(hy::model::fixed_centres_potential, Gconst, masses, positions);
        },
        "Gconst"_a.noconvert() = 1., "masses"_a.noconvert() = py::list{},
        "positions"_a = py::array{py::dtype(get_dtype<double>()), py::array::ShapeContainer{0, 3}});

    // Rotating reference frame.
    m.def(
        "_model_rotating",
        [](const std::vector<vex_t> &omega) { return detail::rotating_impl(hy::model::rotating, omega); },
        "omega"_a.noconvert() = py::list{});
    m.def(
        "_model_rotating_energy",
        [](const std::vector<vex_t> &omega) { return detail::rotating_impl(hy::model::rotating_energy, omega); },
        "omega"_a.noconvert() = py::list{});
    m.def(
        "_model_rotating_potential",
        [](const std::vector<vex_t> &omega) { return detail::rotating_impl(hy::model::rotating_potential, omega); },
        "omega"_a.noconvert() = py::list{});

    // Mascon.
    m.def(
        "_model_mascon",
        [](const vex_t &Gconst, const std::vector<vex_t> &masses, const py::iterable &positions,
           const std::vector<vex_t> &omega) {
            return detail::mascon_impl(hy::model::mascon, Gconst, masses, positions, omega);
        },
        "Gconst"_a.noconvert() = 1., "masses"_a.noconvert() = py::list{},
        "positions"_a = py::array{py::dtype(get_dtype<double>()), py::array::ShapeContainer{0, 3}},
        "omega"_a.noconvert() = py::list{});
    m.def(
        "_model_mascon_energy",
        [](const vex_t &Gconst, const std::vector<vex_t> &masses, const py::iterable &positions,
           const std::vector<vex_t> &omega) {
            return detail::mascon_impl(hy::model::mascon_energy, Gconst, masses, positions, omega);
        },
        "Gconst"_a.noconvert() = 1., "masses"_a.noconvert() = py::list{},
        "positions"_a = py::array{py::dtype(get_dtype<double>()), py::array::ShapeContainer{0, 3}},
        "omega"_a.noconvert() = py::list{});
    m.def(
        "_model_mascon_potential",
        [](const vex_t &Gconst, const std::vector<vex_t> &masses, const py::iterable &positions,
           const std::vector<vex_t> &omega) {
            return detail::mascon_impl(hy::model::mascon_potential, Gconst, masses, positions, omega);
        },
        "Gconst"_a.noconvert() = 1., "masses"_a.noconvert() = py::list{},
        "positions"_a = py::array{py::dtype(get_dtype<double>()), py::array::ShapeContainer{0, 3}},
        "omega"_a.noconvert() = py::list{});

    // VSOP2013.
    m.def(
        "_model_vsop2013_elliptic",
        [](std::uint32_t pl_idx, std::uint32_t var_idx, hy::expression t_expr, double thresh) {
            return hy::model::vsop2013_elliptic(pl_idx, var_idx, hy::kw::time_expr = std::move(t_expr),
                                                hy::kw::thresh = thresh);
        },
        "pl_idx"_a, "var_idx"_a = 0, "time_expr"_a = hy::time, "thresh"_a.noconvert() = 1e-9,
        docstrings::vsop2013_elliptic().c_str());
    m.def(
        "_model_vsop2013_cartesian",
        [](std::uint32_t pl_idx, hy::expression t_expr, double thresh) {
            return hy::model::vsop2013_cartesian(pl_idx, hy::kw::time_expr = std::move(t_expr),
                                                 hy::kw::thresh = thresh);
        },
        "pl_idx"_a, "time_expr"_a = hy::time, "thresh"_a.noconvert() = 1e-9, docstrings::vsop2013_cartesian().c_str());
    m.def(
        "_model_vsop2013_cartesian_icrf",
        [](std::uint32_t pl_idx, hy::expression t_expr, double thresh) {
            return hy::model::vsop2013_cartesian_icrf(pl_idx, hy::kw::time_expr = std::move(t_expr),
                                                      hy::kw::thresh = thresh);
        },
        "pl_idx"_a, "time_expr"_a = hy::time, "thresh"_a.noconvert() = 1e-9,
        docstrings::vsop2013_cartesian_icrf().c_str());
    m.def("_model_get_vsop2013_mus", &hy::model::get_vsop2013_mus, docstrings::get_vsop2013_mus().c_str());

    // ELP2000.
    m.def(
        "_model_elp2000_cartesian_e2000",
        [](hy::expression t_expr, double thresh) {
            return hy::model::elp2000_cartesian_e2000(hy::kw::time_expr = std::move(t_expr), hy::kw::thresh = thresh);
        },
        "time_expr"_a = hy::time, "thresh"_a.noconvert() = 1e-6, docstrings::elp2000_cartesian_e2000().c_str());
    m.def(
        "_model_elp2000_cartesian_fk5",
        [](hy::expression t_expr, double thresh) {
            return hy::model::elp2000_cartesian_fk5(hy::kw::time_expr = std::move(t_expr), hy::kw::thresh = thresh);
        },
        "time_expr"_a = hy::time, "thresh"_a.noconvert() = 1e-6, docstrings::elp2000_cartesian_fk5().c_str());
    m.def("_model_get_elp2000_mus", &hy::model::get_elp2000_mus, docstrings::get_elp2000_mus().c_str());

    // CR3BP.
    m.def(
        "_model_cr3bp", [](const vex_t &mu) { return detail::cr3bp_impl(hy::model::cr3bp, mu); },
        "mu"_a.noconvert() = 1e-3);
    m.def(
        "_model_cr3bp_jacobi", [](const vex_t &mu) { return detail::cr3bp_impl(hy::model::cr3bp_jacobi, mu); },
        "mu"_a.noconvert() = 1e-3);

    // FFNN.
    m.def(
        "_model_ffnn",
        [](const std::vector<hy::expression> &inputs, const std::vector<std::uint32_t> &nn_hidden, std::uint32_t n_out,
           const std::vector<std::function<hy::expression(const hy::expression &)>> &activations,
           const std::variant<std::vector<hy::expression>, std::vector<double>> &nn_wb) {
            return std::visit(
                [&](const auto &w) {
                    return hy::model::ffnn(hy::kw::inputs = inputs, hy::kw::nn_hidden = nn_hidden,
                                           hy::kw::n_out = n_out, hy::kw::activations = activations, hy::kw::nn_wb = w);
                },
                nn_wb);
        },
        "inputs"_a, "nn_hidden"_a, "n_out"_a, "activations"_a, "nn_wb"_a);

    m.def(
        "_model_ffnn",
        [](const std::vector<hy::expression> &inputs, const std::vector<std::uint32_t> &nn_hidden, std::uint32_t n_out,
           const std::vector<std::function<hy::expression(const hy::expression &)>> &activations) {
            return hy::model::ffnn(hy::kw::inputs = inputs, hy::kw::nn_hidden = nn_hidden, hy::kw::n_out = n_out,
                                   hy::kw::activations = activations);
        },
        "inputs"_a, "nn_hidden"_a, "n_out"_a, "activations"_a);

    // Cartesian<->geodetic.
    m.def(
        "_model_cart2geo",
        [](const std::array<vex_t, 3> &xyz, double ecc2, double R_eq, unsigned n_iters) {
            return hy::model::cart2geo(
                {detail::ex_from_variant(xyz[0]), detail::ex_from_variant(xyz[1]), detail::ex_from_variant(xyz[2])},
                hy::kw::ecc2 = ecc2, hy::kw::R_eq = R_eq, hy::kw::n_iters = n_iters);
        },
        "xyz"_a,
        "ecc2"_a = 1
                   - hy::model::detail::b_earth * hy::model::detail::b_earth
                         / (hy::model::detail::a_earth * hy::model::detail::a_earth),
        "R_eq"_a = hy::model::detail::a_earth, "n_iters"_a = 4u, docstrings::cart2geo().c_str());
    m.def(
        "_model_geo2cart",
        [](const std::array<vex_t, 3> &geo, double ecc2, double R_eq) {
            return hy::model::geo2cart(
                {detail::ex_from_variant(geo[0]), detail::ex_from_variant(geo[1]), detail::ex_from_variant(geo[2])},
                hy::kw::ecc2 = ecc2, hy::kw::R_eq = R_eq);
        },
        "geo"_a,
        "ecc2"_a = 1
                   - hy::model::detail::b_earth * hy::model::detail::b_earth
                         / (hy::model::detail::a_earth * hy::model::detail::a_earth),
        "R_eq"_a = hy::model::detail::a_earth, docstrings::geo2cart().c_str());

    // Thermospheric model NRLMSISE00
    m.def(
        "_model_nrlmsise00_tn",
        [](const std::vector<hy::expression> &geodetic, const hy::expression &f107, const hy::expression &f107a,
           const hy::expression &ap, const hy::expression &time) -> hy::expression {
            return hy::model::nrlmsise00_tn(hy::kw::geodetic = geodetic, hy::kw::f107 = f107, hy::kw::f107a = f107a,
                                            hy::kw::ap = ap, hy::kw::time_expr = time);
        },
        "geodetic"_a, "f107"_a, "f107a"_a, "ap"_a, "time_expr"_a, docstrings::nrlmsise00_tn().c_str());

    // Thermospheric model JB08
    m.def(
        "_model_jb08_tn",
        [](const std::vector<hy::expression> &geodetic, const hy::expression &f107, const hy::expression &f107a,
           const hy::expression &s107, const hy::expression &s107a, const hy::expression &m107,
           const hy::expression &m107a, const hy::expression &y107a, const hy::expression &y107,
           const hy::expression &dDstdT, const hy::expression &time) -> hy::expression {
            return hy::model::jb08_tn(hy::kw::geodetic = geodetic, hy::kw::f107 = f107, hy::kw::f107a = f107a,
                                      hy::kw::s107 = s107, hy::kw::s107a = s107a, hy::kw::m107 = m107,
                                      hy::kw::m107a = m107a, hy::kw::y107 = y107, hy::kw::y107a = y107a,
                                      hy::kw::dDstdT = dDstdT, hy::kw::time_expr = time);
        },
        "geodetic"_a, "f107"_a, "f107a"_a, "s107"_a, "s107a"_a, "m107"_a, "m107a"_a, "y107"_a, "y107a"_a, "dDstdT"_a,
        "time_expr"_a, docstrings::jb08_tn().c_str());

    // sgp4.
    m.def(
        "_model_sgp4",
        [](const std::optional<std::vector<vex_t>> &inputs_) {
            if (inputs_) {
                std::vector<hy::expression> inputs;
                inputs.reserve(inputs_->size());
                std::ranges::transform(*inputs_, std::back_inserter(inputs), [](const auto &vex) {
                    return std::visit([](const auto &v) { return hy::expression(v); }, vex);
                });

                return hy::model::sgp4(inputs);
            } else {
                return hy::model::sgp4();
            }
        },
        "inputs"_a.noconvert() = py::none{}, docstrings::sgp4().c_str());
    expose_sgp4_propagators(m);
    m.def("_model_gpe_is_deep_space", &hy::model::gpe_is_deep_space, "n0"_a.noconvert(), "e0"_a.noconvert(),
          "i0"_a.noconvert(), docstrings::gpe_is_deep_space().c_str());

    // Time conversions.
    m.attr("_model_delta_tt_tai") = hy::model::delta_tt_tai;
    m.def(
        "_model_delta_tdb_tt",
        [](const vex_t &time_expr) { return hy::model::delta_tdb_tt(detail::ex_from_variant(time_expr)); },
        "time_expr"_a.noconvert() = hy::time, docstrings::delta_tdb_tt().c_str());

    // Frame transformations.
    m.def(
        "_model_rot_fk5j2000_icrs",
        [](const std::array<vex_t, 3> &xyz) {
            return hy::model::rot_fk5j2000_icrs(
                {detail::ex_from_variant(xyz[0]), detail::ex_from_variant(xyz[1]), detail::ex_from_variant(xyz[2])});
        },
        "xyz"_a, docstrings::rot_fk5j2000_icrs().c_str());
    m.def(
        "_model_rot_icrs_fk5j2000",
        [](const std::array<vex_t, 3> &xyz) {
            return hy::model::rot_icrs_fk5j2000(
                {detail::ex_from_variant(xyz[0]), detail::ex_from_variant(xyz[1]), detail::ex_from_variant(xyz[2])});
        },
        "xyz"_a, docstrings::rot_icrs_fk5j2000().c_str());
    m.def(
        "_model_rot_itrs_icrs",
        [](const std::array<vex_t, 3> &xyz, const vex_t &time_expr, double thresh, const hy::eop_data &data) {
            return hy::model::rot_itrs_icrs(
                {detail::ex_from_variant(xyz[0]), detail::ex_from_variant(xyz[1]), detail::ex_from_variant(xyz[2])},
                hy::kw::time_expr = detail::ex_from_variant(time_expr), hy::kw::thresh = thresh,
                hy::kw::eop_data = data);
        },
        "xyz"_a, "time_expr"_a = hy::time, "thresh"_a.noconvert() = hy::model::detail::iau2006_default_thresh,
        "eop_data"_a = hy::eop_data(), docstrings::rot_itrs_icrs(hy::model::detail::iau2006_default_thresh).c_str());
    m.def(
        "_model_rot_icrs_itrs",
        [](const std::array<vex_t, 3> &xyz, const vex_t &time_expr, double thresh, const hy::eop_data &data) {
            return hy::model::rot_icrs_itrs(
                {detail::ex_from_variant(xyz[0]), detail::ex_from_variant(xyz[1]), detail::ex_from_variant(xyz[2])},
                hy::kw::time_expr = detail::ex_from_variant(time_expr), hy::kw::thresh = thresh,
                hy::kw::eop_data = data);
        },
        "xyz"_a, "time_expr"_a = hy::time, "thresh"_a.noconvert() = hy::model::detail::iau2006_default_thresh,
        "eop_data"_a = hy::eop_data(), docstrings::rot_icrs_itrs(hy::model::detail::iau2006_default_thresh).c_str());
    m.def(
        "_model_rot_itrs_teme",
        [](const std::array<vex_t, 3> &xyz, const vex_t &time_expr, const hy::eop_data &data) {
            return hy::model::rot_itrs_teme(
                {detail::ex_from_variant(xyz[0]), detail::ex_from_variant(xyz[1]), detail::ex_from_variant(xyz[2])},
                hy::kw::time_expr = detail::ex_from_variant(time_expr), hy::kw::eop_data = data);
        },
        "xyz"_a, "time_expr"_a = hy::time, "eop_data"_a = hy::eop_data(), docstrings::rot_itrs_teme().c_str());
    m.def(
        "_model_rot_teme_itrs",
        [](const std::array<vex_t, 3> &xyz, const vex_t &time_expr, const hy::eop_data &data) {
            return hy::model::rot_teme_itrs(
                {detail::ex_from_variant(xyz[0]), detail::ex_from_variant(xyz[1]), detail::ex_from_variant(xyz[2])},
                hy::kw::time_expr = detail::ex_from_variant(time_expr), hy::kw::eop_data = data);
        },
        "xyz"_a, "time_expr"_a = hy::time, "eop_data"_a = hy::eop_data(), docstrings::rot_teme_itrs().c_str());

    // Use macro to expose the EOP models.
#define HEYOKA_PY_EXPOSE_MODEL_EOP(name)                                                                               \
    m.def(                                                                                                             \
        "_model_" #name,                                                                                               \
        [](const vex_t &time_expr, const hy::eop_data &data) {                                                         \
            return hy::model::name(hy::kw::time_expr = detail::ex_from_variant(time_expr), hy::kw::eop_data = data);   \
        },                                                                                                             \
        "time_expr"_a = hy::time, "eop_data"_a = hy::eop_data(), docstrings::name().c_str());                          \
    m.def(                                                                                                             \
        "_model_" #name "p",                                                                                           \
        [](const vex_t &time_expr, const hy::eop_data &data) {                                                         \
            return hy::model::name##p(hy::kw::time_expr = detail::ex_from_variant(time_expr),                          \
                                      hy::kw::eop_data = data);                                                        \
        },                                                                                                             \
        "time_expr"_a = hy::time, "eop_data"_a = hy::eop_data(), docstrings::name##p().c_str());

    HEYOKA_PY_EXPOSE_MODEL_EOP(era);
    HEYOKA_PY_EXPOSE_MODEL_EOP(gmst82);
    HEYOKA_PY_EXPOSE_MODEL_EOP(pm_x);
    HEYOKA_PY_EXPOSE_MODEL_EOP(pm_y);
    HEYOKA_PY_EXPOSE_MODEL_EOP(dX);
    HEYOKA_PY_EXPOSE_MODEL_EOP(dY);

#undef HEYOKA_PY_EXPOSE_MODEL_EOP

    // IAU2006 precession/nutation.
    m.def(
        "_model_iau2006",
        [](hy::expression t_expr, double thresh) {
            return hy::model::iau2006(hy::kw::time_expr = std::move(t_expr), hy::kw::thresh = thresh);
        },
        "time_expr"_a = hy::time, "thresh"_a.noconvert() = hy::model::detail::iau2006_default_thresh,
        docstrings::iau2006(hy::model::detail::iau2006_default_thresh).c_str());

    // EGM2008.
    m.def(
        "_model_egm2008_pot",
        [](const std::array<vex_t, 3> &xyz, std::uint32_t n, std::uint32_t m, const vex_t &mu, const vex_t &a) {
            return hy::model::egm2008_pot(
                {detail::ex_from_variant(xyz[0]), detail::ex_from_variant(xyz[1]), detail::ex_from_variant(xyz[2])}, n,
                m, hy::kw::mu = detail::ex_from_variant(mu), hy::kw::a = detail::ex_from_variant(a));
        },
        "xyz"_a, "n"_a.noconvert(), "m"_a.noconvert(), "mu"_a = hy::model::get_egm2008_mu(),
        "a"_a = hy::model::get_egm2008_a(), docstrings::egm2008_pot().c_str());
    m.def(
        "_model_egm2008_acc",
        [](const std::array<vex_t, 3> &xyz, std::uint32_t n, std::uint32_t m, const vex_t &mu, const vex_t &a) {
            return hy::model::egm2008_acc(
                {detail::ex_from_variant(xyz[0]), detail::ex_from_variant(xyz[1]), detail::ex_from_variant(xyz[2])}, n,
                m, hy::kw::mu = detail::ex_from_variant(mu), hy::kw::a = detail::ex_from_variant(a));
        },
        "xyz"_a, "n"_a.noconvert(), "m"_a.noconvert(), "mu"_a = hy::model::get_egm2008_mu(),
        "a"_a = hy::model::get_egm2008_a(), docstrings::egm2008_acc().c_str());
    m.def("_model_get_egm2008_mu", &hy::model::get_egm2008_mu, docstrings::get_egm2008_mu().c_str());
    m.def("_model_get_egm2008_a", &hy::model::get_egm2008_a, docstrings::get_egm2008_a().c_str());

    // Use macro to expose the SW models.
#define HEYOKA_PY_EXPOSE_MODEL_SW(name)                                                                                \
    m.def(                                                                                                             \
        "_model_" #name,                                                                                               \
        [](const vex_t &time_expr, const hy::sw_data &data) {                                                          \
            return hy::model::name(hy::kw::time_expr = detail::ex_from_variant(time_expr), hy::kw::sw_data = data);    \
        },                                                                                                             \
        "time_expr"_a = hy::time, "sw_data"_a = hy::sw_data(), docstrings::name().c_str());

    HEYOKA_PY_EXPOSE_MODEL_SW(Ap_avg);
    HEYOKA_PY_EXPOSE_MODEL_SW(f107);
    HEYOKA_PY_EXPOSE_MODEL_SW(f107a_center81);

#undef HEYOKA_PY_EXPOSE_MODEL_SW

    // dayfrac().
    m.def(
        "_model_dayfrac", [](const hy::expression &tm) { return hy::model::dayfrac(hy::kw::time_expr = tm); },
        "time_expr"_a = hy::time, docstrings::dayfrac().c_str());
}

} // namespace heyoka_py
