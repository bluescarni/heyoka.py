// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <Python.h>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <heyoka/expression.hpp>
#include <heyoka/taylor.hpp>

#include "common_utils.hpp"
#include "dtypes.hpp"
#include "expose_batch_integrators.hpp"
#include "pickle_wrappers.hpp"

namespace heyoka_py
{

namespace py = pybind11;

namespace detail
{

namespace
{

template <typename T>
void expose_batch_integrator_impl(py::module_ &m, const std::string &suffix)
{
    namespace hey = heyoka;
    namespace kw = hey::kw;
    using namespace pybind11::literals;

    // The callback for the propagate_*() functions for
    // the batch integrator.
    using prop_cb_t = std::function<bool(hey::taylor_adaptive_batch<T> &)>;

    // Event types for the batch integrator.
    using t_ev_t = hey::t_event_batch<T>;
    using nt_ev_t = hey::nt_event_batch<T>;

    // Implementation of the ctor.
    auto tab_ctor_impl = [](const auto &sys, const py::iterable &state_ob, std::optional<py::iterable> time_ob,
                            std::optional<py::iterable> pars_ob, T tol, bool high_accuracy, bool compact_mode,
                            std::vector<t_ev_t> tes, std::vector<nt_ev_t> ntes, bool parallel_mode) {
        // Fetch the dtype corresponding to T.
        const auto dt = get_dtype<T>();

        // Convert state and pars to std::vector, after checking
        // dimensions and shape.
        py::array state_ = state_ob;
        if (state_.ndim() != 2) {
            py_throw(PyExc_ValueError,
                     fmt::format("Invalid state vector passed to the constructor of a batch integrator: "
                                 "the expected number of dimensions is 2, but the input array has a dimension of {}",
                                 state_.ndim())
                         .c_str());
        }

        // Enforce the correct dtype.
        if (state_.dtype().num() != dt) {
            state_ = state_.attr("astype")(py::dtype(dt), "casting"_a = "safe");
        }

        // Infer the batch size from the second dimension.
        const auto batch_size = boost::numeric_cast<std::uint32_t>(state_.shape(1));

        // Flatten out and convert to a C++ vector.
        auto state = py::cast<std::vector<T>>(state_.attr("flatten")());

        // If pars is none, an empty vector will be fine.
        std::vector<T> pars;
        if (pars_ob) {
            py::array pars_arr = *pars_ob;

            if (pars_arr.ndim() != 2 || boost::numeric_cast<std::uint32_t>(pars_arr.shape(1)) != batch_size) {
                py_throw(PyExc_ValueError,
                         fmt::format("Invalid parameter vector passed to the constructor of a batch integrator: "
                                     "the expected array shape is (n, {}), but the input array has either the wrong "
                                     "number of dimensions or the wrong shape",
                                     batch_size)
                             .c_str());
            }

            // Enforce the correct dtype.
            if (pars_arr.dtype().num() != dt) {
                pars_arr = pars_arr.attr("astype")(py::dtype(dt), "casting"_a = "safe");
            }

            pars = py::cast<std::vector<T>>(pars_arr.attr("flatten")());
        }

        if (time_ob) {
            // Times provided.
            py::array time_arr = *time_ob;
            if (time_arr.ndim() != 1 || boost::numeric_cast<std::uint32_t>(time_arr.shape(0)) != batch_size) {
                py_throw(PyExc_ValueError,
                         fmt::format("Invalid time vector passed to the constructor of a batch integrator: "
                                     "the expected array shape is ({}), but the input array has either the wrong "
                                     "number of dimensions or the wrong shape",
                                     batch_size)
                             .c_str());
            }
            auto time = py::cast<std::vector<T>>(time_arr);

            // NOTE: GIL release is fine here even if the events contain
            // Python objects, as the event vectors are moved in
            // upon construction and thus we should never end up calling
            // into the interpreter.
            py::gil_scoped_release release;

            return hey::taylor_adaptive_batch<T>{sys,
                                                 std::move(state),
                                                 batch_size,
                                                 kw::time = std::move(time),
                                                 kw::tol = tol,
                                                 kw::high_accuracy = high_accuracy,
                                                 kw::compact_mode = compact_mode,
                                                 kw::pars = std::move(pars),
                                                 kw::t_events = std::move(tes),
                                                 kw::nt_events = std::move(ntes),
                                                 kw::parallel_mode = parallel_mode};
        } else {
            // Times not provided.

            // NOTE: GIL release is fine here even if the events contain
            // Python objects, as the event vectors are moved in
            // upon construction and thus we should never end up calling
            // into the interpreter.
            py::gil_scoped_release release;

            return hey::taylor_adaptive_batch<T>{sys,
                                                 std::move(state),
                                                 batch_size,
                                                 kw::tol = tol,
                                                 kw::high_accuracy = high_accuracy,
                                                 kw::compact_mode = compact_mode,
                                                 kw::pars = std::move(pars),
                                                 kw::t_events = std::move(tes),
                                                 kw::nt_events = std::move(ntes),
                                                 kw::parallel_mode = parallel_mode};
        }
    };

    py::class_<hey::taylor_adaptive_batch<T>> tab_c(m, fmt::format("taylor_adaptive_batch_{}", suffix).c_str(),
                                                    py::dynamic_attr{});

    using variant_t
        = std::variant<std::vector<std::pair<hey::expression, hey::expression>>, std::vector<hey::expression>>;

    tab_c
        .def(py::init([tab_ctor_impl](const variant_t &sys, const py::iterable &state, std::optional<py::iterable> time,
                                      std::optional<py::iterable> pars, T tol, bool high_accuracy, bool compact_mode,
                                      std::vector<t_ev_t> tes, std::vector<nt_ev_t> ntes, bool parallel_mode) {
                 return std::visit(
                     [&](const auto &value) {
                         return tab_ctor_impl(value, state, std::move(time), std::move(pars), tol, high_accuracy,
                                              compact_mode, std::move(tes), std::move(ntes), parallel_mode);
                     },
                     sys);
             }),
             "sys"_a, "state"_a, "time"_a = py::none{}, "pars"_a = py::none{}, "tol"_a.noconvert() = static_cast<T>(0),
             "high_accuracy"_a = false, "compact_mode"_a = false, "t_events"_a = py::list{}, "nt_events"_a = py::list{},
             "parallel_mode"_a = false)
        .def_property_readonly("decomposition", &hey::taylor_adaptive_batch<T>::get_decomposition)
        .def(
            "step", [](hey::taylor_adaptive_batch<T> &ta, bool wtc) { ta.step(wtc); }, "write_tc"_a = false)
        .def(
            "step",
            [](hey::taylor_adaptive_batch<T> &ta, const std::vector<T> &max_delta_t, bool wtc) {
                ta.step(max_delta_t, wtc);
            },
            "max_delta_t"_a.noconvert(), "write_tc"_a = false)
        .def("step_backward", &hey::taylor_adaptive_batch<T>::step_backward, "write_tc"_a = false)
        .def_property_readonly("step_res", [](const hey::taylor_adaptive_batch<T> &ta) { return ta.get_step_res(); })
        .def(
            "propagate_for",
            [](hey::taylor_adaptive_batch<T> &ta, const std::variant<T, std::vector<T>> &delta_t, std::size_t max_steps,
               std::variant<T, std::vector<T>> max_delta_t, const prop_cb_t &cb_, bool write_tc, bool c_output) {
                return std::visit(
                    [&](const auto &dt, auto max_dts) {
                        // Create the callback wrapper.
                        auto cb = make_prop_cb(cb_);

                        // NOTE: after releasing the GIL here, the only potential
                        // calls into the Python interpreter are when invoking cb
                        // or the events' callbacks (which are all protected by GIL reacquire).
                        // Note that copying cb around or destroying it is harmless, as it contains only
                        // a reference to the original callback cb_, or it is an empty callback.
                        py::gil_scoped_release release;
                        return ta.propagate_for(dt, kw::max_steps = max_steps, kw::max_delta_t = std::move(max_dts),
                                                kw::callback = cb, kw::write_tc = write_tc, kw::c_output = c_output);
                    },
                    delta_t, std::move(max_delta_t));
            },
            "delta_t"_a.noconvert(), "max_steps"_a = 0, "max_delta_t"_a.noconvert() = std::vector<T>{},
            "callback"_a = prop_cb_t{}, "write_tc"_a = false, "c_output"_a = false)
        .def(
            "propagate_until",
            [](hey::taylor_adaptive_batch<T> &ta, const std::variant<T, std::vector<T>> &tm, std::size_t max_steps,
               std::variant<T, std::vector<T>> max_delta_t, const prop_cb_t &cb_, bool write_tc, bool c_output) {
                return std::visit(
                    [&](const auto &t, auto max_dts) {
                        // Create the callback wrapper.
                        auto cb = make_prop_cb(cb_);

                        py::gil_scoped_release release;
                        return ta.propagate_until(t, kw::max_steps = max_steps, kw::max_delta_t = std::move(max_dts),
                                                  kw::callback = cb, kw::write_tc = write_tc, kw::c_output = c_output);
                    },
                    tm, std::move(max_delta_t));
            },
            "t"_a.noconvert(), "max_steps"_a = 0, "max_delta_t"_a.noconvert() = std::vector<T>{},
            "callback"_a = prop_cb_t{}, "write_tc"_a = false, "c_output"_a = false)
        .def(
            "propagate_grid",
            [](hey::taylor_adaptive_batch<T> &ta, const py::iterable &grid_ob, std::size_t max_steps,
               std::variant<T, std::vector<T>> max_delta_t, const prop_cb_t &cb_) {
                return std::visit(
                    [&](auto max_dts) {
                        // Attempt to convert grid_ob to an array.
                        py::array grid = grid_ob;

                        // Check the grid dimension/shape.
                        if (grid.ndim() != 2) {
                            py_throw(
                                PyExc_ValueError,
                                fmt::format(
                                    "Invalid grid passed to the propagate_grid() method of a batch integrator: "
                                    "the expected number of dimensions is 2, but the input array has a dimension of {}",
                                    grid.ndim())
                                    .c_str());
                        }
                        if (boost::numeric_cast<std::uint32_t>(grid.shape(1)) != ta.get_batch_size()) {
                            py_throw(
                                PyExc_ValueError,
                                fmt::format("Invalid grid passed to the propagate_grid() method of a batch integrator: "
                                            "the shape must be (n, {}) but the number of columns is {} instead",
                                            ta.get_batch_size(), grid.shape(1))
                                    .c_str());
                        }

                        // Enforce the correct dtype.
                        const auto dt = get_dtype<T>();
                        if (grid.dtype().num() != dt) {
                            grid = grid.attr("astype")(py::dtype(dt), "casting"_a = "safe");
                        }

                        // Convert to a std::vector.
                        auto grid_v = py::cast<std::vector<T>>(grid.attr("flatten")());

#if !defined(NDEBUG)
                        // Store the grid size for debug.
                        const auto grid_v_size = grid_v.size();
#endif

                        // Create the callback wrapper.
                        auto cb = make_prop_cb(cb_);

                        // Run the propagation.
                        // NOTE: for batch integrators, ret is guaranteed to always have
                        // the same size regardless of errors.
                        decltype(ta.propagate_grid(grid_v, max_steps)) ret;
                        {
                            py::gil_scoped_release release;
                            ret = ta.propagate_grid(std::move(grid_v), kw::max_steps = max_steps,
                                                    kw::max_delta_t = std::move(max_dts), kw::callback = cb);
                        }

                        // Create the output array.
                        assert(ret.size() == grid_v_size * ta.get_dim());
                        py::array a_ret(grid.dtype(),
                                        py::array::ShapeContainer{grid.shape(0),
                                                                  boost::numeric_cast<py::ssize_t>(ta.get_dim()),
                                                                  grid.shape(1)},
                                        ret.data());

                        return a_ret;
                    },
                    std::move(max_delta_t));
            },
            "grid"_a, "max_steps"_a = 0, "max_delta_t"_a.noconvert() = std::vector<T>{}, "callback"_a = prop_cb_t{})
        .def_property_readonly("propagate_res",
                               [](const hey::taylor_adaptive_batch<T> &ta) { return ta.get_propagate_res(); })
        .def_property_readonly(
            "time",
            [](py::object &o) {
                auto *ta = py::cast<hey::taylor_adaptive_batch<T> *>(o);
                py::array ret(py::dtype(get_dtype<T>()),
                              py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(ta->get_time().size())},
                              ta->get_time_data(), o);

                // Ensure the returned array is read-only.
                ret.attr("flags").attr("writeable") = false;

                return ret;
            })
        .def_property_readonly("dtime",
                               [](py::object &o) {
                                   auto *ta = py::cast<hey::taylor_adaptive_batch<T> *>(o);

                                   const auto dt = get_dtype<T>();

                                   py::array hi_ret(py::dtype(dt),
                                                    py::array::ShapeContainer{
                                                        boost::numeric_cast<py::ssize_t>(ta->get_dtime().first.size())},
                                                    ta->get_dtime_data().first, o);
                                   py::array lo_ret(py::dtype(dt),
                                                    py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(
                                                        ta->get_dtime().second.size())},
                                                    ta->get_dtime_data().second, o);

                                   // Ensure the returned arrays are read-only.
                                   hi_ret.attr("flags").attr("writeable") = false;
                                   lo_ret.attr("flags").attr("writeable") = false;

                                   return py::make_tuple(hi_ret, lo_ret);
                               })
        .def(
            "set_time",
            [](hey::taylor_adaptive_batch<T> &ta, const std::variant<T, std::vector<T>> &tm) {
                std::visit([&ta](const auto &t) { ta.set_time(t); }, tm);
            },
            "time"_a.noconvert())
        .def(
            "set_dtime",
            [](hey::taylor_adaptive_batch<T> &ta, const std::variant<T, std::vector<T>> &hi_tm,
               const std::variant<T, std::vector<T>> &lo_tm) {
                std::visit(
                    [&ta](const auto &t_hi, const auto &t_lo) {
                        if constexpr (std::is_same_v<decltype(t_hi), decltype(t_lo)>) {
                            ta.set_dtime(t_hi, t_lo);
                        } else {
                            py_throw(PyExc_TypeError,
                                     "The two arguments to the set_dtime() method must be of the same type");
                        }
                    },
                    hi_tm, lo_tm);
            },
            "hi_time"_a.noconvert(), "lo_time"_a.noconvert())
        .def_property_readonly("state",
                               [](py::object &o) {
                                   auto *ta = py::cast<hey::taylor_adaptive_batch<T> *>(o);

                                   assert(ta->get_state().size() % ta->get_batch_size() == 0u);
                                   const auto nvars = boost::numeric_cast<py::ssize_t>(ta->get_dim());
                                   const auto bs = boost::numeric_cast<py::ssize_t>(ta->get_batch_size());

                                   return py::array(py::dtype(get_dtype<T>()), py::array::ShapeContainer{nvars, bs},
                                                    ta->get_state_data(), o);
                               })
        .def_property_readonly("pars",
                               [](py::object &o) {
                                   auto *ta = py::cast<hey::taylor_adaptive_batch<T> *>(o);

                                   assert(ta->get_pars().size() % ta->get_batch_size() == 0u);
                                   const auto npars
                                       = boost::numeric_cast<py::ssize_t>(ta->get_pars().size() / ta->get_batch_size());
                                   const auto bs = boost::numeric_cast<py::ssize_t>(ta->get_batch_size());

                                   return py::array(py::dtype(get_dtype<T>()), py::array::ShapeContainer{npars, bs},
                                                    ta->get_pars_data(), o);
                               })
        .def_property_readonly("tc",
                               [](const py::object &o) {
                                   const auto *ta = py::cast<const hey::taylor_adaptive_batch<T> *>(o);

                                   const auto nvars = boost::numeric_cast<py::ssize_t>(ta->get_dim());
                                   const auto ncoeff = boost::numeric_cast<py::ssize_t>(ta->get_order() + 1u);
                                   const auto bs = boost::numeric_cast<py::ssize_t>(ta->get_batch_size());

                                   auto ret = py::array(py::dtype(get_dtype<T>()),
                                                        py::array::ShapeContainer{nvars, ncoeff, bs},
                                                        ta->get_tc().data(), o);

                                   // Ensure the returned array is read-only.
                                   ret.attr("flags").attr("writeable") = false;

                                   return ret;
                               })
        .def_property_readonly(
            "last_h",
            [](const py::object &o) {
                const auto *ta = py::cast<const hey::taylor_adaptive_batch<T> *>(o);

                auto ret = py::array(py::dtype(get_dtype<T>()),
                                     py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(ta->get_batch_size())},
                                     ta->get_last_h().data(), o);

                // Ensure the returned array is read-only.
                ret.attr("flags").attr("writeable") = false;

                return ret;
            })
        .def_property_readonly("d_output",
                               [](const py::object &o) {
                                   const auto *ta = py::cast<const hey::taylor_adaptive_batch<T> *>(o);

                                   const auto nvars = boost::numeric_cast<py::ssize_t>(ta->get_dim());
                                   const auto bs = boost::numeric_cast<py::ssize_t>(ta->get_batch_size());

                                   auto ret = py::array(py::dtype(get_dtype<T>()), py::array::ShapeContainer{nvars, bs},
                                                        ta->get_d_output().data(), o);

                                   // Ensure the returned array is read-only.
                                   ret.attr("flags").attr("writeable") = false;

                                   return ret;
                               })
        .def(
            "update_d_output",
            [](py::object &o, const std::variant<T, std::vector<T>> &tm, bool rel_time) {
                return std::visit(
                    [&o, rel_time](const auto &t) {
                        auto *ta = py::cast<hey::taylor_adaptive_batch<T> *>(o);

                        ta->update_d_output(t, rel_time);

                        const auto nvars = boost::numeric_cast<py::ssize_t>(ta->get_dim());
                        const auto bs = boost::numeric_cast<py::ssize_t>(ta->get_batch_size());

                        auto ret = py::array(py::dtype(get_dtype<T>()), py::array::ShapeContainer{nvars, bs},
                                             ta->get_d_output().data(), o);

                        // Ensure the returned array is read-only.
                        ret.attr("flags").attr("writeable") = false;

                        return ret;
                    },
                    tm);
            },
            "t"_a.noconvert(), "rel_time"_a = false)
        .def_property_readonly("order", &hey::taylor_adaptive_batch<T>::get_order)
        .def_property_readonly("tol", &hey::taylor_adaptive_batch<T>::get_tol)
        .def_property_readonly("dim", &hey::taylor_adaptive_batch<T>::get_dim)
        .def_property_readonly("batch_size", &hey::taylor_adaptive_batch<T>::get_batch_size)
        .def_property_readonly("compact_mode", &hey::taylor_adaptive_batch<T>::get_compact_mode)
        .def_property_readonly("high_accuracy", &hey::taylor_adaptive_batch<T>::get_high_accuracy)
        .def_property_readonly("with_events", &hey::taylor_adaptive_batch<T>::with_events)
        // Event detection.
        .def_property_readonly("with_events", &hey::taylor_adaptive_batch<T>::with_events)
        .def_property_readonly("te_cooldowns", &hey::taylor_adaptive_batch<T>::get_te_cooldowns)
        .def("reset_cooldowns", [](hey::taylor_adaptive_batch<T> &ta) { ta.reset_cooldowns(); })
        .def("reset_cooldowns", [](hey::taylor_adaptive_batch<T> &ta, std::uint32_t i) { ta.reset_cooldowns(i); })
        .def_property_readonly("t_events", &hey::taylor_adaptive_batch<T>::get_t_events)
        .def_property_readonly("nt_events", &hey::taylor_adaptive_batch<T>::get_nt_events)
        // Repr.
        .def("__repr__",
             [](const hey::taylor_adaptive_batch<T> &ta) {
                 std::ostringstream oss;
                 oss << ta;
                 return oss.str();
             })
        // Copy/deepcopy.
        .def("__copy__", copy_wrapper<hey::taylor_adaptive_batch<T>>)
        .def("__deepcopy__", deepcopy_wrapper<hey::taylor_adaptive_batch<T>>, "memo"_a)
        // Pickle support.
        .def(py::pickle(&pickle_getstate_wrapper<hey::taylor_adaptive_batch<T>>,
                        &pickle_setstate_wrapper<hey::taylor_adaptive_batch<T>>));

    // Expose the llvm state getter.
    expose_llvm_state_property(tab_c);
}

} // namespace

} // namespace detail

void expose_batch_integrators(py::module_ &m)
{
    detail::expose_batch_integrator_impl<double>(m, "dbl");
}

} // namespace heyoka_py
