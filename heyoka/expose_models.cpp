// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <heyoka/model/pendulum.hpp>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/models.hpp>

#include "custom_casters.hpp"
#include "expose_models.hpp"

namespace heyoka_py
{

namespace detail
{

namespace
{

// Small helper to turn v into an expression.
constexpr auto make_ex = [](const auto &v) { return heyoka::expression(v); };

// Common logic to expose N-body helpers.
template <typename Op, typename V>
auto nbody_impl(const Op &op, std::uint32_t n, const V &Gconst, const std::optional<std::vector<V>> &masses)
{
    namespace hy = heyoka;

    const auto Gval = std::visit(make_ex, Gconst);

    if (masses) {
        std::vector<hy::expression> masses_vec;
        for (const auto &mass : *masses) {
            masses_vec.push_back(std::visit(make_ex, mass));
        }

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

    const auto gval = std::visit(make_ex, gconst);
    const auto lval = std::visit(make_ex, l);

    return op(hy::kw::gconst = gval, hy::kw::l = lval);
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
}

} // namespace heyoka_py
