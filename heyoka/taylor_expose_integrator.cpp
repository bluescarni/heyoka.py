// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <heyoka/expression.hpp>
#include <heyoka/taylor.hpp>

#include "long_double_caster.hpp"
#include "taylor_expose_integrator.hpp"

namespace heyoka_py
{

namespace py = pybind11;
namespace hey = heyoka;

namespace detail
{

namespace
{

template <typename T>
void expose_taylor_integrator_impl(py::module &m, const std::string &suffix)
{
    using namespace pybind11::literals;
    using fmt::literals::operator""_format;

    auto ctor_impl = [](auto sys, std::vector<T> state, T time, std::vector<T> pars, T tol, bool high_accuracy,
                        bool compact_mode) {
        py::gil_scoped_release release;

        namespace kw = hey::kw;
        return hey::taylor_adaptive<T>{std::move(sys),
                                       std::move(state),
                                       kw::time = time,
                                       kw::tol = tol,
                                       kw::high_accuracy = high_accuracy,
                                       kw::compact_mode = compact_mode,
                                       kw::pars = std::move(pars)};
    };

    py::class_<hey::taylor_adaptive<T>>(m, ("taylor_adaptive_{}"_format(suffix)).c_str())
        .def(py::init([ctor_impl](std::vector<std::pair<hey::expression, hey::expression>> sys, std::vector<T> state,
                                  T time, std::vector<T> pars, T tol, bool high_accuracy, bool compact_mode) {
                 return ctor_impl(std::move(sys), std::move(state), time, std::move(pars), tol, high_accuracy,
                                  compact_mode);
             }),
             "sys"_a, "state"_a, "time"_a = T(0), "pars"_a = py::list{}, "tol"_a = T(0), "high_accuracy"_a = false,
             "compact_mode"_a = false)
        .def(py::init([ctor_impl](std::vector<hey::expression> sys, std::vector<T> state, T time, std::vector<T> pars,
                                  T tol, bool high_accuracy, bool compact_mode) {
                 return ctor_impl(std::move(sys), std::move(state), time, std::move(pars), tol, high_accuracy,
                                  compact_mode);
             }),
             "sys"_a, "state"_a, "time"_a = T(0), "pars"_a = py::list{}, "tol"_a = T(0), "high_accuracy"_a = false,
             "compact_mode"_a = false)
        .def_property_readonly("decomposition", &hey::taylor_adaptive<T>::get_decomposition)
        .def(
            "step", [](hey::taylor_adaptive<T> &ta, bool wtc) { return ta.step(wtc); }, "write_tc"_a = false)
        .def(
            "step", [](hey::taylor_adaptive<T> &ta, T max_delta_t, bool wtc) { return ta.step(max_delta_t, wtc); },
            "max_delta_t"_a, "write_tc"_a = false)
        .def(
            "step_backward", [](hey::taylor_adaptive<T> &ta, bool wtc) { return ta.step_backward(wtc); },
            "write_tc"_a = false)
        .def(
            "propagate_for",
            [](hey::taylor_adaptive<T> &ta, T delta_t, std::size_t max_steps) {
                py::gil_scoped_release release;
                return ta.propagate_for(delta_t, max_steps);
            },
            "delta_t"_a, "max_steps"_a = 0)
        .def(
            "propagate_until",
            [](hey::taylor_adaptive<T> &ta, T t, std::size_t max_steps) {
                py::gil_scoped_release release;
                return ta.propagate_until(t, max_steps);
            },
            "t"_a, "max_steps"_a = 0)
        .def(
            "propagate_grid",
            [](hey::taylor_adaptive<T> &ta, const std::vector<T> &grid, std::size_t max_steps) {
                decltype(ta.propagate_grid(grid, max_steps)) ret;

                {
                    py::gil_scoped_release release;
                    ret = ta.propagate_grid(grid, max_steps);
                }

                // Determine the number of state vectors returned
                // (could be < grid.size() if errors arise).
                assert(std::get<4>(ret).size() % ta.get_dim() == 0u);
                const auto nrows = boost::numeric_cast<py::ssize_t>(std::get<4>(ret).size() / ta.get_dim());
                const auto ncols = boost::numeric_cast<py::ssize_t>(ta.get_dim());

                // Convert the output to a NumPy array.
                py::array_t<T> a_ret(py::array::ShapeContainer{nrows, ncols}, std::get<4>(ret).data());

                return py::make_tuple(std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret),
                                      std::move(a_ret));
            },
            "grid"_a, "max_steps"_a = 0)
        .def_property("time", &hey::taylor_adaptive<T>::get_time, &hey::taylor_adaptive<T>::set_time)
        .def_property_readonly("state",
                               [](py::object &o) {
                                   auto *ta = py::cast<hey::taylor_adaptive<T> *>(o);
                                   return py::array_t<T>(py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(
                                                             ta->get_state().size())},
                                                         ta->get_state_data(), o);
                               })
        .def_property_readonly("pars",
                               [](py::object &o) {
                                   auto *ta = py::cast<hey::taylor_adaptive<T> *>(o);
                                   return py::array_t<T>(py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(
                                                             ta->get_pars().size())},
                                                         ta->get_pars_data(), o);
                               })
        .def_property_readonly(
            "tc",
            [](const py::object &o) {
                auto *ta = py::cast<const hey::taylor_adaptive<T> *>(o);

                const auto nvars = boost::numeric_cast<py::ssize_t>(ta->get_dim());
                const auto ncoeff = boost::numeric_cast<py::ssize_t>(ta->get_order() + 1u);

                auto ret = py::array_t<T>(py::array::ShapeContainer{nvars, ncoeff}, ta->get_tc().data(), o);

                // Ensure the returned array is read-only.
                ret.attr("flags").attr("writeable") = false;

                return ret;
            })
        .def_property_readonly("last_h", &hey::taylor_adaptive<T>::get_last_h)
        .def_property_readonly("d_output",
                               [](const py::object &o) {
                                   auto *ta = py::cast<const hey::taylor_adaptive<T> *>(o);

                                   auto ret = py::array_t<T>(py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(
                                                                 ta->get_d_output().size())},
                                                             ta->get_d_output().data(), o);

                                   // Ensure the returned array is read-only.
                                   ret.attr("flags").attr("writeable") = false;

                                   return ret;
                               })
        .def(
            "update_d_output",
            [](py::object &o, T t) {
                auto *ta = py::cast<hey::taylor_adaptive<T> *>(o);

                ta->update_d_output(t);

                auto ret = py::array_t<T>(
                    py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(ta->get_d_output().size())},
                    ta->get_d_output().data(), o);

                // Ensure the returned array is read-only.
                ret.attr("flags").attr("writeable") = false;

                return ret;
            },
            "t"_a)
        .def_property_readonly("order", &hey::taylor_adaptive<T>::get_order)
        .def_property_readonly("dim", &hey::taylor_adaptive<T>::get_dim)
        // Repr.
        .def("__repr__",
             [](const hey::taylor_adaptive<T> &ta) {
                 std::ostringstream oss;
                 oss << ta;
                 return oss.str();
             })
        // Copy/deepcopy.
        .def("__copy__",
             [](const hey::taylor_adaptive<T> &ta) {
                 py::gil_scoped_release release;
                 return ta;
             })
        .def(
            "__deepcopy__",
            [](const hey::taylor_adaptive<T> &ta, py::dict) {
                py::gil_scoped_release release;
                return ta;
            },
            "memo"_a);
}

} // namespace

} // namespace detail

void expose_taylor_integrator_dbl(py::module &m)
{
    detail::expose_taylor_integrator_impl<double>(m, "dbl");
}

void expose_taylor_integrator_ldbl(py::module &m)
{
    detail::expose_taylor_integrator_impl<long double>(m, "ldbl");
}

} // namespace heyoka_py