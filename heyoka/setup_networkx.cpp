// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <variant>

#include <pybind11/pybind11.h>

#include <Python.h>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/binary_op.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/variable.hpp>

#include "common_utils.hpp"

namespace heyoka_py
{

namespace py = pybind11;
namespace hy = heyoka;

namespace detail
{

namespace
{

py::object to_networkx_impl(py::object &, const hy::expression &);

py::object to_networkx_impl(py::object &G, const hy::variable &var)
{
    py::kwargs attr;
    attr["label"] = var.name();
    attr["color"] = "#a8e6aa";

    auto idx = py::cast(py::len(G));

    G.attr("add_node")(idx, **attr);

    return idx;
}

py::object to_networkx_impl(py::object &G, const hy::param &par)
{
    // NOTE: params are converted to symbolic variables
    // following a naming convention.
    using namespace fmt::literals;

    py::kwargs attr;
    attr["label"] = "par[{}]"_format(par.idx());
    attr["color"] = "#f2df9c";

    auto idx = py::cast(py::len(G));

    G.attr("add_node")(idx, **attr);

    return idx;
}

py::object to_networkx_impl(py::object &G, const hy::number &num)
{
    using namespace fmt::literals;

    py::kwargs attr;
    auto str = "{}"_format(num);
    str.resize(5u);
    attr["label"] = str + "...";
    attr["color"] = "#f0cfcc";

    auto idx = py::cast(py::len(G));

    G.attr("add_node")(idx, **attr);

    return idx;
}

py::object to_networkx_impl(py::object &G, const hy::func &f)
{
    py::kwargs attr;

    if (auto bop = f.extract<hy::detail::binary_op>()) {
        switch (bop->op()) {
            case hy::detail::binary_op::type::add:
                attr["label"] = "$+$";
                break;
            case hy::detail::binary_op::type::sub:
                attr["label"] = "$-$";
                break;
            case hy::detail::binary_op::type::mul:
                attr["label"] = "$\\cdot$";
                break;
            default:
                attr["label"] = "$/$";
        }
    } else {
        attr["label"] = f.get_name();
    }

    attr["color"] = "#b5dbfc";

    auto idx = py::cast(py::len(G));

    G.attr("add_node")(idx, **attr);

    for (const auto &arg : f.args()) {
        G.attr("add_edge")(idx, to_networkx_impl(G, arg));
    }

    return idx;
}

py::object to_networkx_impl(py::object &G, const hy::expression &ex)
{
    return std::visit([&G](const auto &v) { return to_networkx_impl(G, v); }, ex.value());
}

py::object to_networkx(const hy::expression &ex)
{
    auto nx = py::module_::import("networkx");

    py::object G = nx.attr("Graph")();

    to_networkx_impl(G, ex);

    return G;
}

} // namespace

} // namespace detail

void setup_networkx(py::module &m)
{
    bool has_nx = true;

    try {
        py::module_::import("networkx");
    } catch (...) {
        has_nx = false;
    }

    if (has_nx) {
        // Expose the conversion function.
        m.def("to_networkx", &detail::to_networkx);
    } else {
        m.def("to_networkx", [](const hy::expression &) {
            py_throw(PyExc_ImportError,
                     "The 'to_networkx()' function is not available because networkx is not installed");
        });
    }
}

} // namespace heyoka_py
