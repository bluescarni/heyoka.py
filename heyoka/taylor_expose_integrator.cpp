// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Python.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/extra/pybind11.hpp>
#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/taylor.hpp>

#include "common_utils.hpp"
#include "long_double_caster.hpp"
#include "pickle_wrappers.hpp"
#include "taylor_expose_integrator.hpp"

namespace heyoka_py
{

namespace py = pybind11;
namespace hey = heyoka;
namespace heypy = heyoka_py;

namespace detail
{

namespace
{

// Common bits for the exposition of the scalar integrators
// for double, long double and real128.
template <typename T>
void expose_taylor_integrator_common(py::class_<hey::taylor_adaptive<T>> &cl)
{
    using namespace pybind11::literals;
    using prop_cb_t = std::function<bool(hey::taylor_adaptive<T> &)>;
    namespace kw = heyoka::kw;

    cl.def_property_readonly("decomposition", &hey::taylor_adaptive<T>::get_decomposition)
        .def_property("time", &hey::taylor_adaptive<T>::get_time, &hey::taylor_adaptive<T>::set_time)
        // Step functions.
        .def(
            "step", [](hey::taylor_adaptive<T> &ta, bool wtc) { return ta.step(wtc); }, "write_tc"_a = false)
        .def(
            "step", [](hey::taylor_adaptive<T> &ta, T max_delta_t, bool wtc) { return ta.step(max_delta_t, wtc); },
            "max_delta_t"_a, "write_tc"_a = false)
        .def(
            "step_backward", [](hey::taylor_adaptive<T> &ta, bool wtc) { return ta.step_backward(wtc); },
            "write_tc"_a = false)
        // propagate_*().
        .def(
            "propagate_for",
            [](hey::taylor_adaptive<T> &ta, T delta_t, std::size_t max_steps, T max_delta_t, const prop_cb_t &cb_,
               bool write_tc, bool c_output) {
                // Create the callback wrapper.
                auto cb = make_prop_cb(cb_);

                // NOTE: after releasing the GIL here, the only potential
                // calls into the Python interpreter are when invoking cb
                // or the events' callbacks (which are all protected by GIL reacquire).
                // Note that copying cb around or destroying it is harmless, as it contains only
                // a reference to the original callback cb_, or it is an empty callback.
                py::gil_scoped_release release;
                return ta.propagate_for(delta_t, kw::max_steps = max_steps, kw::max_delta_t = max_delta_t,
                                        kw::callback = std::move(cb), kw::write_tc = write_tc, kw::c_output = c_output);
            },
            "delta_t"_a, "max_steps"_a = 0, "max_delta_t"_a = std::numeric_limits<T>::infinity(),
            "callback"_a = prop_cb_t{}, "write_tc"_a = false, "c_output"_a = false)
        .def(
            "propagate_until",
            [](hey::taylor_adaptive<T> &ta, T t, std::size_t max_steps, T max_delta_t, const prop_cb_t &cb_,
               bool write_tc, bool c_output) {
                // Create the callback wrapper.
                auto cb = make_prop_cb(cb_);

                py::gil_scoped_release release;
                return ta.propagate_until(t, kw::max_steps = max_steps, kw::max_delta_t = max_delta_t,
                                          kw::callback = std::move(cb), kw::write_tc = write_tc,
                                          kw::c_output = c_output);
            },
            "t"_a, "max_steps"_a = 0, "max_delta_t"_a = std::numeric_limits<T>::infinity(), "callback"_a = prop_cb_t{},
            "write_tc"_a = false, "c_output"_a = false)
        // Repr.
        .def("__repr__",
             [](const hey::taylor_adaptive<T> &ta) {
                 std::ostringstream oss;
                 oss << ta;
                 return oss.str();
             })
        // Copy/deepcopy.
        .def("__copy__", copy_wrapper<hey::taylor_adaptive<T>>)
        .def("__deepcopy__", deepcopy_wrapper<hey::taylor_adaptive<T>>, "memo"_a)
        // Pickle support.
        .def(py::pickle(&pickle_getstate_wrapper<hey::taylor_adaptive<T>>,
                        &pickle_setstate_wrapper<hey::taylor_adaptive<T>>))
        // Various read-only properties.
        .def_property_readonly("last_h", &hey::taylor_adaptive<T>::get_last_h)
        .def_property_readonly("order", &hey::taylor_adaptive<T>::get_order)
        .def_property_readonly("tol", &hey::taylor_adaptive<T>::get_tol)
        .def_property_readonly("dim", &hey::taylor_adaptive<T>::get_dim)
        .def_property_readonly("compact_mode", &hey::taylor_adaptive<T>::get_compact_mode)
        .def_property_readonly("high_accuracy", &hey::taylor_adaptive<T>::get_high_accuracy)
        .def_property_readonly("with_events", &hey::taylor_adaptive<T>::with_events)
        // Event detection.
        .def_property_readonly("with_events", &hey::taylor_adaptive<T>::with_events)
        .def_property_readonly("te_cooldowns", &hey::taylor_adaptive<T>::get_te_cooldowns)
        .def("reset_cooldowns", &hey::taylor_adaptive<T>::reset_cooldowns)
        .def_property_readonly("t_events", &hey::taylor_adaptive<T>::get_t_events)
        .def_property_readonly("nt_events", &hey::taylor_adaptive<T>::get_nt_events);

    // Expose the llvm state getter.
    expose_llvm_state_property(cl);
}

// Implementation of the exposition of the scalar integrators
// for double and long double.
template <typename T>
void expose_taylor_integrator_impl(py::module &m, const std::string &suffix)
{
    using namespace pybind11::literals;
    namespace kw = heyoka::kw;

    using t_ev_t = hey::t_event<T>;
    using nt_ev_t = hey::nt_event<T>;
    using prop_cb_t = std::function<bool(hey::taylor_adaptive<T> &)>;

    auto ctor_impl = [](const auto &sys, std::vector<T> state, T time, std::vector<T> pars, T tol, bool high_accuracy,
                        bool compact_mode, std::vector<t_ev_t> tes, std::vector<nt_ev_t> ntes) {
        // NOTE: GIL release is fine here even if the events contain
        // Python objects, as the event vectors are moved in
        // upon construction and thus we should never end up calling
        // into the interpreter.
        py::gil_scoped_release release;

        namespace kw = hey::kw;
        return hey::taylor_adaptive<T>{sys,
                                       std::move(state),
                                       kw::time = time,
                                       kw::tol = tol,
                                       kw::high_accuracy = high_accuracy,
                                       kw::compact_mode = compact_mode,
                                       kw::pars = std::move(pars),
                                       kw::t_events = std::move(tes),
                                       kw::nt_events = std::move(ntes)};
    };

    py::class_<hey::taylor_adaptive<T>> cl(m, (fmt::format("_taylor_adaptive_{}", suffix)).c_str(), py::dynamic_attr{});
    cl.def(py::init([ctor_impl](const std::vector<std::pair<hey::expression, hey::expression>> &sys,
                                std::vector<T> state, T time, std::vector<T> pars, T tol, bool high_accuracy,
                                bool compact_mode, std::vector<t_ev_t> tes, std::vector<nt_ev_t> ntes) {
               return ctor_impl(sys, std::move(state), time, std::move(pars), tol, high_accuracy, compact_mode,
                                std::move(tes), std::move(ntes));
           }),
           "sys"_a, "state"_a, "time"_a = T(0), "pars"_a = py::list{}, "tol"_a = T(0), "high_accuracy"_a = false,
           "compact_mode"_a = false, "t_events"_a = py::list{}, "nt_events"_a = py::list{})
        .def(py::init([ctor_impl](const std::vector<hey::expression> &sys, std::vector<T> state, T time,
                                  std::vector<T> pars, T tol, bool high_accuracy, bool compact_mode,
                                  std::vector<t_ev_t> tes, std::vector<nt_ev_t> ntes) {
                 return ctor_impl(sys, std::move(state), time, std::move(pars), tol, high_accuracy, compact_mode,
                                  std::move(tes), std::move(ntes));
             }),
             "sys"_a, "state"_a, "time"_a = T(0), "pars"_a = py::list{}, "tol"_a = T(0), "high_accuracy"_a = false,
             "compact_mode"_a = false, "t_events"_a = py::list{}, "nt_events"_a = py::list{})
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
            [](py::object &o, T t, bool rel_time) {
                auto *ta = py::cast<hey::taylor_adaptive<T> *>(o);

                ta->update_d_output(t, rel_time);

                auto ret = py::array_t<T>(
                    py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(ta->get_d_output().size())},
                    ta->get_d_output().data(), o);

                // Ensure the returned array is read-only.
                ret.attr("flags").attr("writeable") = false;

                return ret;
            },
            "t"_a, "rel_time"_a = false)
        .def(
            "propagate_grid",
            [](hey::taylor_adaptive<T> &ta, std::vector<T> grid, std::size_t max_steps, T max_delta_t,
               const prop_cb_t &cb_) {
                // Create the callback wrapper.
                auto cb = make_prop_cb(cb_);

                decltype(ta.propagate_grid(grid, max_steps)) ret;

                {
                    py::gil_scoped_release release;
                    ret = ta.propagate_grid(std::move(grid), kw::max_steps = max_steps, kw::max_delta_t = max_delta_t,
                                            kw::callback = std::move(cb));
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
            "grid"_a, "max_steps"_a = 0, "max_delta_t"_a = std::numeric_limits<T>::infinity(),
            "callback"_a = prop_cb_t{});

    expose_taylor_integrator_common(cl);
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

#if defined(HEYOKA_HAVE_REAL128)

void expose_taylor_integrator_f128(py::module &m)
{
    using namespace pybind11::literals;
    namespace kw = heyoka::kw;

    using t_ev_t = hey::t_event<mppp::real128>;
    using nt_ev_t = hey::nt_event<mppp::real128>;
    using prop_cb_t = std::function<bool(hey::taylor_adaptive<mppp::real128> &)>;

    // NOTE: we need to temporarily alter
    // the precision in mpmath to successfully
    // construct the default values of the parameters
    // for the constructor.
    scoped_quadprec_setter qs;

    auto taf128_ctor_impl = [](const auto &sys, std::vector<mppp::real128> state, mppp::real128 time,
                               std::vector<mppp::real128> pars, mppp::real128 tol, bool high_accuracy,
                               bool compact_mode, std::vector<t_ev_t> tes, std::vector<nt_ev_t> ntes) {
        // NOTE: GIL release is fine here even if the events contain
        // Python objects, as the event vectors are moved in
        // upon construction and thus we should never end up calling
        // into the interpreter.
        py::gil_scoped_release release;

        namespace kw = hey::kw;
        return hey::taylor_adaptive<mppp::real128>{sys,
                                                   std::move(state),
                                                   kw::time = time,
                                                   kw::tol = tol,
                                                   kw::high_accuracy = high_accuracy,
                                                   kw::compact_mode = compact_mode,
                                                   kw::pars = std::move(pars),
                                                   kw::t_events = std::move(tes),
                                                   kw::nt_events = std::move(ntes)};
    };

    py::class_<hey::taylor_adaptive<mppp::real128>> cl(m, "_taylor_adaptive_f128", py::dynamic_attr{});
    cl.def(py::init([taf128_ctor_impl](const std::vector<std::pair<hey::expression, hey::expression>> &sys,
                                       std::vector<mppp::real128> state, mppp::real128 time,
                                       std::vector<mppp::real128> pars, mppp::real128 tol, bool high_accuracy,
                                       bool compact_mode, std::vector<t_ev_t> tes, std::vector<nt_ev_t> ntes) {
               return taf128_ctor_impl(sys, std::move(state), time, std::move(pars), tol, high_accuracy, compact_mode,
                                       std::move(tes), std::move(ntes));
           }),
           "sys"_a, "state"_a, "time"_a = mppp::real128{0}, "pars"_a = py::list{}, "tol"_a = mppp::real128{0},
           "high_accuracy"_a = false, "compact_mode"_a = false, "t_events"_a = py::list{}, "nt_events"_a = py::list{})
        .def(py::init([taf128_ctor_impl](const std::vector<hey::expression> &sys, std::vector<mppp::real128> state,
                                         mppp::real128 time, std::vector<mppp::real128> pars, mppp::real128 tol,
                                         bool high_accuracy, bool compact_mode, std::vector<t_ev_t> tes,
                                         std::vector<nt_ev_t> ntes) {
                 return taf128_ctor_impl(sys, std::move(state), time, std::move(pars), tol, high_accuracy, compact_mode,
                                         std::move(tes), std::move(ntes));
             }),
             "sys"_a, "state"_a, "time"_a = mppp::real128{0}, "pars"_a = py::list{}, "tol"_a = mppp::real128{0},
             "high_accuracy"_a = false, "compact_mode"_a = false, "t_events"_a = py::list{}, "nt_events"_a = py::list{})
        .def(
            "propagate_grid",
            [](hey::taylor_adaptive<mppp::real128> &ta, std::vector<mppp::real128> grid, std::size_t max_steps,
               mppp::real128 max_delta_t, const prop_cb_t &cb_) {
                // Create the callback wrapper.
                auto cb = make_prop_cb(cb_);

                decltype(ta.propagate_grid(grid, max_steps)) ret;

                {
                    py::gil_scoped_release release;
                    ret = ta.propagate_grid(std::move(grid), kw::max_steps = max_steps, kw::max_delta_t = max_delta_t,
                                            kw::callback = std::move(cb));
                }

                // Determine the number of state vectors returned
                // (could be < grid.size() if errors arise).
                assert(std::get<4>(ret).size() % ta.get_dim() == 0u);
                const auto nrows = std::get<4>(ret).size() / ta.get_dim();
                const auto ncols = ta.get_dim();

                // Convert the output to a NumPy array.
                auto a_ret = py::array(py::cast(std::get<4>(ret)));

                // Reshape.
                a_ret.attr("shape") = py::make_tuple(nrows, ncols);

                return py::make_tuple(std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret),
                                      std::move(a_ret));
            },
            "grid"_a, "max_steps"_a = 0, "max_delta_t"_a = std::numeric_limits<mppp::real128>::infinity(),
            "callback"_a = prop_cb_t{})
        .def("get_state",
             [](const hey::taylor_adaptive<mppp::real128> &ta) { return py::array(py::cast(ta.get_state())); })
        .def("set_state",
             [](hey::taylor_adaptive<mppp::real128> &ta, const std::vector<mppp::real128> &v) {
                 if (v.size() != ta.get_state().size()) {
                     heypy::py_throw(PyExc_ValueError, fmt::format("Invalid state vector passed to set_state(): "
                                                                   "the new state vector has a size of {}, "
                                                                   "but the size should be {} instead",
                                                                   v.size(), ta.get_state().size())
                                                           .c_str());
                 }

                 std::copy(v.begin(), v.end(), ta.get_state_data());
             })
        .def("get_pars",
             [](const hey::taylor_adaptive<mppp::real128> &ta) { return py::array(py::cast(ta.get_pars())); })
        .def("set_pars",
             [](hey::taylor_adaptive<mppp::real128> &ta, const std::vector<mppp::real128> &v) {
                 if (v.size() != ta.get_pars().size()) {
                     heypy::py_throw(
                         PyExc_ValueError,
                         fmt::format("Invalid pars vector passed to set_pars(): "
                                     "the new pars vector has a size of {}, but the size should be {} instead",
                                     v.size(), ta.get_pars().size())
                             .c_str());
                 }

                 std::copy(v.begin(), v.end(), ta.get_state_data());
             })
        .def_property_readonly("tc",
                               [](const hey::taylor_adaptive<mppp::real128> &ta) {
                                   auto ret = py::array(py::cast(ta.get_tc()));

                                   const auto nvars = ta.get_dim();
                                   const auto ncoeff = ta.get_order() + 1u;

                                   ret.attr("shape") = py::make_tuple(nvars, ncoeff);

                                   return ret;
                               })
        .def("d_output",
             [](const hey::taylor_adaptive<mppp::real128> &ta) { return py::array(py::cast(ta.get_d_output())); })
        .def(
            "update_d_output",
            [](hey::taylor_adaptive<mppp::real128> &ta, mppp::real128 t, bool rel_time) {
                ta.update_d_output(t, rel_time);

                return py::array(py::cast(ta.get_d_output()));
            },
            "t"_a, "rel_time"_a = false);

    detail::expose_taylor_integrator_common(cl);
}

#endif

} // namespace heyoka_py
