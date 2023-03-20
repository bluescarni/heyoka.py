// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <iterator>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <fmt/core.h>

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

#include <heyoka/expression.hpp>
#include <heyoka/models.hpp>

#include "common_utils.hpp"
#include "custom_casters.hpp"
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

    return op(hy::kw::gconst = gval, hy::kw::l = lval);
}

// Common logic to expose fixed centres helpers.
template <typename Op, typename V>
auto fixed_centres_impl(const Op &op, const V &Gconst, const std::vector<V> &masses, const py::iterable &positions_)
{
    namespace hy = heyoka;

    // Attempt to convert the input position argument into
    // a NumPy array.
    py::array positions = positions_;

    // Check the shape of positions.
    if (positions.ndim() != 2) {
        py_throw(PyExc_ValueError, fmt::format("Invalid positions array in a fixed centres model: the number of "
                                               "dimensions must be 2, but it is {} instead",
                                               positions.ndim())
                                       .c_str());
    }

    if (positions.shape(1) != 3) {
        py_throw(PyExc_ValueError, fmt::format("Invalid positions array in a fixed centres model: the number of "
                                               "columns must be 3, but it is {} instead",
                                               positions.shape(1))
                                       .c_str());
    }

    // Flatten out postions into a list and convert to a vector of variants.
    py::list pflat = positions.attr("flatten")();
    std::vector<V> pvec;
    try {
        pvec = pflat.cast<std::vector<V>>();
    } catch (const py::cast_error &) {
        py_throw(PyExc_TypeError,
                 "The positions array in a fixed centres model could not be converted into an array "
                 "of expressions - please make sure that the array's values can be converted into heyoka expressions");
    }

    // Build the arguments for the C++ function.
    const auto Gval = ex_from_variant(Gconst);
    std::vector<hy::expression> masses_vec, positions_vec;
    std::transform(masses.begin(), masses.end(), std::back_inserter(masses_vec), ex_from_variant);
    std::transform(pvec.begin(), pvec.end(), std::back_inserter(positions_vec), ex_from_variant);

    return op(hy::kw::Gconst = Gval, hy::kw::masses = masses_vec, hy::kw::positions = positions_vec);
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
    using vex_t = std::variant<hy::expression, std::string, double, long double
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
        "gconst"_a.noconvert() = 1., "l"_a.noconvert() = 1.);
    m.def(
        "_model_pendulum_energy",
        [](const vex_t &gconst, const vex_t &l) {
            return detail::pendulum_impl(hy::model::pendulum_energy, gconst, l);
        },
        "gconst"_a.noconvert() = 1., "l"_a.noconvert() = 1.);

    // Fixed centres.
    m.def(
        "_model_fixed_centres",
        [](const vex_t &Gconst, const std::vector<vex_t> &masses, const py::iterable &positions) {
            return detail::fixed_centres_impl(hy::model::fixed_centres, Gconst, masses, positions);
        },
        "Gconst"_a.noconvert() = 1., "masses"_a.noconvert() = py::list{},
        "positions"_a = py::array{py::dtype(get_dtype<double>()), py::array::ShapeContainer{0, 3}});
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
}

} // namespace heyoka_py
