// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

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

    m.def(
        "_model_nbody",
        [](std::uint32_t n, const vex_t &Gconst, const std::optional<std::vector<vex_t>> &masses) {
            const auto Gval = std::visit(detail::make_ex, Gconst);

            if (masses) {
                std::vector<hy::expression> masses_vec;
                for (const auto &mass : *masses) {
                    masses_vec.push_back(std::visit(detail::make_ex, mass));
                }

                return hy::model::nbody(n, hy::kw::Gconst = Gval, hy::kw::masses = masses_vec);
            } else {
                return hy::model::nbody(n, hy::kw::Gconst = Gval);
            }
        },
        "n"_a, "Gconst"_a = 1., "masses"_a = py::none{});
}

} // namespace heyoka_py
