// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <Python.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <heyoka/events.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/step_callback.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/var_ode_sys.hpp>

#include "common_utils.hpp"
#include "custom_casters.hpp"
#include "dtypes.hpp"
#include "expose_batch_integrators.hpp"
#include "pickle_wrappers.hpp"
#include "step_cb_utils.hpp"

namespace heyoka_py
{

namespace py = pybind11;
namespace hey = heyoka;

namespace detail
{

namespace
{

// Helper to fetch the tstate from a variational integrator.
// Extracted here for re-use.
template <typename T>
auto fetch_tstate(const py::object &o)
{
    const auto *ta = py::cast<const hey::taylor_adaptive_batch<T> *>(o);

    assert(ta->get_tstate().size() % ta->get_batch_size() == 0u);
    const auto n_orig_sv = boost::numeric_cast<py::ssize_t>(ta->get_n_orig_sv());
    const auto bs = boost::numeric_cast<py::ssize_t>(ta->get_batch_size());

    auto ret
        = py::array(py::dtype(get_dtype<T>()), py::array::ShapeContainer{n_orig_sv, bs}, ta->get_tstate().data(), o);

    // Ensure the returned array is read-only.
    ret.attr("flags").attr("writeable") = false;

    return ret;
}

template <typename T>
void expose_batch_integrator_impl(py::module_ &m, const std::string &suffix)
{
    namespace kw = hey::kw;
    using namespace pybind11::literals;

    // Event types for the batch integrator.
    using t_ev_t = hey::t_event_batch<T>;
    using nt_ev_t = hey::nt_event_batch<T>;

    using sys_t = std::vector<std::pair<hey::expression, hey::expression>>;

    // Implementation of the ctor.
    auto tab_ctor_impl = [](std::variant<sys_t, hey::var_ode_sys> vsys, const py::iterable &state_ob,
                            std::optional<py::iterable> time_ob, std::optional<py::iterable> pars_ob, T tol,
                            bool high_accuracy, bool compact_mode, std::vector<t_ev_t> tes, std::vector<nt_ev_t> ntes,
                            bool parallel_mode, unsigned opt_level, bool force_avx512, bool slp_vectorize,
                            bool fast_math, hey::code_model code_model, bool parjit) {
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

        return std::visit(
            [&](auto &sys) {
                if (time_ob) {
                    // Times provided.
                    py::array time_arr = *time_ob;
                    if (time_arr.ndim() != 1 || boost::numeric_cast<std::uint32_t>(time_arr.shape(0)) != batch_size) {
                        py_throw(
                            PyExc_ValueError,
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

                    return hey::taylor_adaptive_batch<T>{std::move(sys),
                                                         std::move(state),
                                                         batch_size,
                                                         kw::time = std::move(time),
                                                         kw::tol = tol,
                                                         kw::high_accuracy = high_accuracy,
                                                         kw::compact_mode = compact_mode,
                                                         kw::pars = std::move(pars),
                                                         kw::t_events = std::move(tes),
                                                         kw::nt_events = std::move(ntes),
                                                         kw::parallel_mode = parallel_mode,
                                                         kw::opt_level = opt_level,
                                                         kw::force_avx512 = force_avx512,
                                                         kw::slp_vectorize = slp_vectorize,
                                                         kw::fast_math = fast_math,
                                                         kw::code_model = code_model,
                                                         kw::parjit = parjit};
                } else {
                    // Times not provided.

                    // NOTE: GIL release is fine here even if the events contain
                    // Python objects, as the event vectors are moved in
                    // upon construction and thus we should never end up calling
                    // into the interpreter.
                    py::gil_scoped_release release;

                    return hey::taylor_adaptive_batch<T>{std::move(sys),
                                                         std::move(state),
                                                         batch_size,
                                                         kw::tol = tol,
                                                         kw::high_accuracy = high_accuracy,
                                                         kw::compact_mode = compact_mode,
                                                         kw::pars = std::move(pars),
                                                         kw::t_events = std::move(tes),
                                                         kw::nt_events = std::move(ntes),
                                                         kw::parallel_mode = parallel_mode,
                                                         kw::opt_level = opt_level,
                                                         kw::force_avx512 = force_avx512,
                                                         kw::slp_vectorize = slp_vectorize,
                                                         kw::fast_math = fast_math,
                                                         kw::code_model = code_model,
                                                         kw::parjit = parjit};
                }
            },
            vsys);
    };

    py::class_<hey::taylor_adaptive_batch<T>> tab_c(m, fmt::format("taylor_adaptive_batch_{}", suffix).c_str(),
                                                    py::dynamic_attr{});

    tab_c
        .def(py::init([tab_ctor_impl](std::variant<sys_t, hey::var_ode_sys> vsys, const py::iterable &state,
                                      std::optional<py::iterable> time, std::optional<py::iterable> pars, T tol,
                                      bool high_accuracy, bool compact_mode, std::vector<t_ev_t> tes,
                                      std::vector<nt_ev_t> ntes, bool parallel_mode, unsigned opt_level,
                                      bool force_avx512, bool slp_vectorize, bool fast_math, hey::code_model code_model,
                                      bool parjit) {
                 return tab_ctor_impl(std::move(vsys), state, std::move(time), std::move(pars), tol, high_accuracy,
                                      compact_mode, std::move(tes), std::move(ntes), parallel_mode, opt_level,
                                      force_avx512, slp_vectorize, fast_math, code_model, parjit);
             }),
             "sys"_a, "state"_a, "time"_a = py::none{}, "pars"_a = py::none{}, "tol"_a.noconvert() = static_cast<T>(0),
             "high_accuracy"_a = false, "compact_mode"_a = false, "t_events"_a = py::list{}, "nt_events"_a = py::list{},
             "parallel_mode"_a = false, HEYOKA_PY_LLVM_STATE_ARGS)
        .def_property_readonly("decomposition", &hey::taylor_adaptive_batch<T>::get_decomposition)
        .def_property_readonly("sys", &hey::taylor_adaptive_batch<T>::get_sys)
        .def(
            "step", [](hey::taylor_adaptive_batch<T> &ta, bool wtc) { ta.step(wtc); }, "write_tc"_a = false)
        .def(
            "step",
            [](hey::taylor_adaptive_batch<T> &ta, const std::vector<T> &max_delta_t, bool wtc) {
                ta.step(max_delta_t, wtc);
            },
            "max_delta_t"_a.noconvert(), "write_tc"_a = false)
        .def("step_backward", &hey::taylor_adaptive_batch<T>::step_backward, "write_tc"_a = false)
        .def_property_readonly("step_res", &hey::taylor_adaptive_batch<T>::get_step_res)
        .def(
            "propagate_for",
            [](hey::taylor_adaptive_batch<T> &ta, const std::variant<T, std::vector<T>> &delta_t, std::size_t max_steps,
               std::variant<T, std::vector<T>> max_delta_t, std::optional<scb_arg_t> cb_, bool write_tc,
               bool c_output) {
                return std::visit(
                    [&](const auto &dt, auto max_dts) {
                        if (cb_) {
                            auto cb = scb_arg_to_step_callback<heyoka::step_callback_batch<T>>(*cb_);

                            auto ret = [&]() {
                                py::gil_scoped_release release;

                                return ta.propagate_for(
                                    dt, kw::max_steps = max_steps, kw::max_delta_t = std::move(max_dts),
                                    kw::callback = std::move(cb), kw::write_tc = write_tc, kw::c_output = c_output);
                            }();

                            return py::make_tuple(std::move(std::get<0>(ret)),
                                                  step_callback_to_scb_arg_t(*cb_, std::get<1>(ret)));
                        } else {
                            auto ret = [&]() {
                                py::gil_scoped_release release;

                                return ta.propagate_for(dt, kw::max_steps = max_steps,
                                                        kw::max_delta_t = std::move(max_dts), kw::write_tc = write_tc,
                                                        kw::c_output = c_output);
                            }();

                            return py::make_tuple(std::move(std::get<0>(ret)), py::none{});
                        }
                    },
                    delta_t, std::move(max_delta_t));
            },
            "delta_t"_a.noconvert(), "max_steps"_a = 0, "max_delta_t"_a.noconvert() = std::vector<T>{},
            "callback"_a = py::none{}, "write_tc"_a = false, "c_output"_a = false)
        .def(
            "propagate_until",
            [](hey::taylor_adaptive_batch<T> &ta, const std::variant<T, std::vector<T>> &tm, std::size_t max_steps,
               std::variant<T, std::vector<T>> max_delta_t, std::optional<scb_arg_t> cb_, bool write_tc,
               bool c_output) {
                return std::visit(
                    [&](const auto &t, auto max_dts) {
                        if (cb_) {
                            auto cb = scb_arg_to_step_callback<heyoka::step_callback_batch<T>>(*cb_);

                            auto ret = [&]() {
                                py::gil_scoped_release release;

                                return ta.propagate_until(
                                    t, kw::max_steps = max_steps, kw::max_delta_t = std::move(max_dts),
                                    kw::callback = std::move(cb), kw::write_tc = write_tc, kw::c_output = c_output);
                            }();

                            return py::make_tuple(std::move(std::get<0>(ret)),
                                                  step_callback_to_scb_arg_t(*cb_, std::get<1>(ret)));
                        } else {
                            auto ret = [&]() {
                                py::gil_scoped_release release;

                                return ta.propagate_until(t, kw::max_steps = max_steps,
                                                          kw::max_delta_t = std::move(max_dts), kw::write_tc = write_tc,
                                                          kw::c_output = c_output);
                            }();

                            return py::make_tuple(std::move(std::get<0>(ret)), py::none{});
                        }
                    },
                    tm, std::move(max_delta_t));
            },
            "t"_a.noconvert(), "max_steps"_a = 0, "max_delta_t"_a.noconvert() = std::vector<T>{},
            "callback"_a = py::none{}, "write_tc"_a = false, "c_output"_a = false)
        .def(
            "propagate_grid",
            [](hey::taylor_adaptive_batch<T> &ta, const py::iterable &grid_ob, std::size_t max_steps,
               std::variant<T, std::vector<T>> max_delta_t, std::optional<scb_arg_t> cb_) {
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

                        // Run the propagation.
                        // NOTE: for batch integrators, ret is guaranteed to always have
                        // the same size regardless of errors.
                        decltype(ta.propagate_grid(grid_v, max_steps)) ret;
                        {
                            if (cb_) {
                                auto cb = scb_arg_to_step_callback<heyoka::step_callback_batch<T>>(*cb_);

                                py::gil_scoped_release release;
                                ret = ta.propagate_grid(std::move(grid_v), kw::max_steps = max_steps,
                                                        kw::max_delta_t = std::move(max_dts),
                                                        kw::callback = std::move(cb));
                            } else {
                                py::gil_scoped_release release;
                                ret = ta.propagate_grid(std::move(grid_v), kw::max_steps = max_steps,
                                                        kw::max_delta_t = std::move(max_dts));
                            }
                        }

                        // Create the output array.
                        assert(std::get<1>(ret).size() == grid_v_size * ta.get_dim());
                        py::array a_ret(grid.dtype(),
                                        py::array::ShapeContainer{grid.shape(0),
                                                                  boost::numeric_cast<py::ssize_t>(ta.get_dim()),
                                                                  grid.shape(1)},
                                        std::get<1>(ret).data());

                        if (cb_) {
                            return py::make_tuple(step_callback_to_scb_arg_t(*cb_, std::get<0>(ret)), std::move(a_ret));
                        } else {
                            return py::make_tuple(py::none{}, std::move(a_ret));
                        }
                    },
                    std::move(max_delta_t));
            },
            "grid"_a, "max_steps"_a = 0, "max_delta_t"_a.noconvert() = std::vector<T>{}, "callback"_a = py::none{})
        .def_property_readonly("propagate_res", &hey::taylor_adaptive_batch<T>::get_propagate_res)
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
        // Variational-specific bits.
        .def_property_readonly("n_orig_sv", &hey::taylor_adaptive_batch<T>::get_n_orig_sv)
        .def_property_readonly("is_variational", &hey::taylor_adaptive_batch<T>::is_variational)
        .def_property_readonly("vargs", &hey::taylor_adaptive_batch<T>::get_vargs)
        .def_property_readonly("vorder", &hey::taylor_adaptive_batch<T>::get_vorder)
        .def_property_readonly("tstate", &fetch_tstate<T>)
        .def(
            "get_vslice",
            [](const hey::taylor_adaptive_batch<T> &ta, std::uint32_t order, std::optional<std::uint32_t> component) {
                const auto ret = component ? ta.get_vslice(*component, order) : ta.get_vslice(order);

                return py::slice(boost::numeric_cast<py::ssize_t>(ret.first),
                                 boost::numeric_cast<py::ssize_t>(ret.second), {});
            },
            "order"_a, "component"_a = py::none{})
        .def(
            "get_mindex",
            [](const hey::taylor_adaptive_batch<T> &ta, std::uint32_t i) {
                const auto &ret = ta.get_mindex(i);

                return dtens_t_it::sparse_to_dense(
                    ret, boost::numeric_cast<heyoka::dtens::v_idx_t::size_type>(ta.get_vargs().size()));
            },
            "i"_a)
        .def("eval_taylor_map",
             [](py::object &o, std::variant<py::array, py::iterable> in) {
                 auto *ta = py::cast<hey::taylor_adaptive_batch<T> *>(o);

                 // Fetch the dtype corresponding to T.
                 const auto dt = get_dtype<T>();

                 // Fetch the inputs array.
                 // NOTE: this will either fetch the existing array, or convert
                 // the input iterable to a py::array on the fly.
                 const auto inputs = std::visit(
                     [&]<typename U>(U &v) {
                         if constexpr (std::same_as<U, py::array>) {
                             // Check the dtype.
                             if (v.dtype().num() != dt) [[unlikely]] {
                                 py_throw(
                                     PyExc_TypeError,
                                     fmt::format(
                                         "Invalid dtype detected for the inputs of a Taylor map evaluation: the "
                                         "expected dtype is '{}', but the dtype of the inputs array is '{}' instead",
                                         str(py::dtype(dt)), str(v.dtype()))
                                         .c_str());
                             }

                             // Check that the inputs array is a C-style contiguous array.
                             if (!is_npy_array_carray(v)) [[unlikely]] {
                                 py_throw(PyExc_ValueError,
                                          "Invalid inputs array detected in a Taylor map evaluation: the array is not "
                                          "C-style contiguous, please "
                                          "consider using numpy.ascontiguousarray() to turn it into one");
                             }

                             return std::move(v);
                         } else {
                             return as_carray(v, dt);
                         }
                     },
                     in);

                 // Validate the number of dimensions for the inputs.
                 // NOTE: this needs to be done regardless of the original type of in.
                 if (inputs.ndim() != 2) [[unlikely]] {
                     py_throw(PyExc_ValueError, fmt::format("The array of inputs provided for the evaluation "
                                                            "of a Taylor map has {} dimension(s), "
                                                            "but it must have 2 dimensions instead",
                                                            inputs.ndim())
                                                    .c_str());
                 }

                 // Validate the shape for the inputs.
                 if (boost::numeric_cast<std::uint32_t>(inputs.shape(0)) != ta->get_vargs().size()) [[unlikely]] {
                     py_throw(PyExc_ValueError, fmt::format("The array of inputs provided for the evaluation "
                                                            "of a Taylor map has {} row(s), "
                                                            "but it must have {} row(s) instead",
                                                            inputs.shape(0), ta->get_vargs().size())
                                                    .c_str());
                 }

                 if (boost::numeric_cast<std::uint32_t>(inputs.shape(1)) != ta->get_batch_size()) [[unlikely]] {
                     py_throw(PyExc_ValueError, fmt::format("The array of inputs provided for the evaluation "
                                                            "of a Taylor map has {} column(s), "
                                                            "but it must have {} column(s) instead",
                                                            inputs.shape(1), ta->get_batch_size())
                                                    .c_str());
                 }

                 // Run the evaluation.
                 const auto *data_ptr = static_cast<const T *>(inputs.data());
                 ta->eval_taylor_map(std::ranges::subrange(data_ptr, data_ptr + inputs.size()));

                 // Return the tstate.
                 return fetch_tstate<T>(o);
             })
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
    expose_llvm_state_property_ta(tab_c);
}

} // namespace

} // namespace detail

void expose_batch_integrators(py::module_ &m)
{
    detail::expose_batch_integrator_impl<float>(m, "flt");
    detail::expose_batch_integrator_impl<double>(m, "dbl");
}

} // namespace heyoka_py
