// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

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
#include "pickle_wrappers.hpp"
#include "step_cb_utils.hpp"
#include "taylor_expose_integrator.hpp"

#if defined(HEYOKA_HAVE_REAL)

#include "expose_real.hpp"

#endif

namespace heyoka_py
{

namespace py = pybind11;
namespace hey = heyoka;

namespace detail
{

namespace
{

template <typename T>
constexpr bool default_cm =
#if defined(HEYOKA_HAVE_REAL)
    std::is_same_v<T, mppp::real>
#else
    false
#endif
    ;

// Helper to fetch the tstate from a variational integrator.
// Extracted here for re-use.
template <typename T>
auto fetch_tstate(const py::object &o)
{
    const auto *ta = py::cast<const hey::taylor_adaptive<T> *>(o);
    auto ret = py::array(py::dtype(get_dtype<T>()),
                         py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(ta->get_tstate().size())},
                         ta->get_tstate().data(), o);

    // Ensure the returned array is read-only.
    ret.attr("flags").attr("writeable") = false;

    return ret;
}

// Implementation of the exposition of the scalar integrators.
template <typename T>
void expose_taylor_integrator_impl(py::module &m, const std::string &suffix)
{
    using namespace py::literals;
    namespace kw = hey::kw;

    using t_ev_t = hey::t_event<T>;
    using nt_ev_t = hey::nt_event<T>;

    using sys_t = std::vector<std::pair<hey::expression, hey::expression>>;

    py::class_<hey::taylor_adaptive<T>> cl(m, (fmt::format("taylor_adaptive_{}", suffix)).c_str(), py::dynamic_attr{});
    cl.def(py::init([](std::variant<sys_t, hey::var_ode_sys> vsys, std::vector<T> state, T time, std::vector<T> pars,
                       T tol, bool high_accuracy, bool compact_mode, std::vector<t_ev_t> tes, std::vector<nt_ev_t> ntes,
                       bool parallel_mode, unsigned opt_level, bool force_avx512, bool slp_vectorize, bool fast_math,
                       hey::code_model code_model, bool parjit, long long prec) {
               // NOTE: GIL release is fine here even if the events contain
               // Python objects, as the event vectors are moved in
               // upon construction and thus we should never end up calling
               // into the interpreter.
               py::gil_scoped_release release;

               return std::visit(
                   [&](auto &sys) {
                       return hey::taylor_adaptive<T>{std::move(sys),
                                                      std::move(state),
                                                      kw::time = time,
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
                                                      kw::prec = prec,
                                                      kw::code_model = code_model,
                                                      kw::parjit = parjit};
                   },
                   vsys);
           }),
           "sys"_a, "state"_a.noconvert(), "time"_a.noconvert() = static_cast<T>(0), "pars"_a.noconvert() = py::list{},
           "tol"_a.noconvert() = static_cast<T>(0), "high_accuracy"_a = false, "compact_mode"_a = default_cm<T>,
           "t_events"_a = py::list{}, "nt_events"_a = py::list{}, "parallel_mode"_a = false, HEYOKA_PY_LLVM_STATE_ARGS,
           "prec"_a.noconvert() = 0)
        .def_property_readonly(
            "state",
            [](py::object &o) {
                auto *ta = py::cast<hey::taylor_adaptive<T> *>(o);
                return py::array(py::dtype(get_dtype<T>()),
                                 py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(ta->get_state().size())},
                                 ta->get_state_data(), o);
            })
        .def_property_readonly(
            "pars",
            [](py::object &o) {
                auto *ta = py::cast<hey::taylor_adaptive<T> *>(o);
                return py::array(py::dtype(get_dtype<T>()),
                                 py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(ta->get_pars().size())},
                                 ta->get_pars_data(), o);
            })
        .def_property_readonly("tc",
                               [](const py::object &o) {
                                   auto *ta = py::cast<const hey::taylor_adaptive<T> *>(o);

                                   const auto nvars = boost::numeric_cast<py::ssize_t>(ta->get_dim());
                                   const auto ncoeff = boost::numeric_cast<py::ssize_t>(ta->get_order() + 1u);

                                   auto ret
                                       = py::array(py::dtype(get_dtype<T>()), py::array::ShapeContainer{nvars, ncoeff},
                                                   ta->get_tc().data(), o);

                                   // Ensure the returned array is read-only.
                                   ret.attr("flags").attr("writeable") = false;

                                   return ret;
                               })
        .def_property_readonly(
            "d_output",
            [](const py::object &o) {
                auto *ta = py::cast<const hey::taylor_adaptive<T> *>(o);

                auto ret
                    = py::array(py::dtype(get_dtype<T>()),
                                py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(ta->get_d_output().size())},
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

                auto ret
                    = py::array(py::dtype(get_dtype<T>()),
                                py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(ta->get_d_output().size())},
                                ta->get_d_output().data(), o);

                // Ensure the returned array is read-only.
                ret.attr("flags").attr("writeable") = false;

                return ret;
            },
            "t"_a.noconvert(), "rel_time"_a = false)
        .def_property_readonly("decomposition", &hey::taylor_adaptive<T>::get_decomposition)
        .def_property_readonly("sys", &hey::taylor_adaptive<T>::get_sys)
        .def_property("time", &hey::taylor_adaptive<T>::get_time, &hey::taylor_adaptive<T>::set_time)
        .def_property("dtime", &hey::taylor_adaptive<T>::get_dtime,
                      [](hey::taylor_adaptive<T> &ta, std::pair<T, T> p) { ta.set_dtime(p.first, p.second); })
        // Step functions.
        .def(
            "step", [](hey::taylor_adaptive<T> &ta, bool wtc) { return ta.step(wtc); }, "write_tc"_a = false)
        .def(
            "step", [](hey::taylor_adaptive<T> &ta, T max_delta_t, bool wtc) { return ta.step(max_delta_t, wtc); },
            "max_delta_t"_a.noconvert(), "write_tc"_a = false)
        .def(
            "step_backward", [](hey::taylor_adaptive<T> &ta, bool wtc) { return ta.step_backward(wtc); },
            "write_tc"_a = false)
        // propagate_*().
        .def(
            "propagate_for",
            [](hey::taylor_adaptive<T> &ta, T delta_t, std::size_t max_steps, T max_delta_t,
               std::optional<scb_arg_t> cb_, bool write_tc, bool c_output) {
                // NOTE: after releasing the GIL, the only potential
                // calls into the Python interpreter are when invoking the event or
                // step callbacks (which are all protected by GIL reacquire).
                if (cb_) {
                    // NOTE: here we convert the input scb_arg_t object into a
                    // step_callback, with the following logic:
                    // - if cb_ contains a single scb_t, then use it to construct
                    //   a step_callback (passing through a step_cb_wrapper in case
                    //   of a Pythonic callback). Otherwise,
                    // - cb_ contains a list of scb_t objects: convert each of them
                    //   into a step_callback (with the same logic as above), and then
                    //   assemble them into a step_callback_set.
                    // NOTE: C++ callbacks will be moved into cb, while Pythonic callback
                    // will be referenced into cb (this is necessary in order to avoid
                    // calling into the Python interpreter in case of exceptions being
                    // raised by the callback within the propagate_*() call).
                    auto cb = scb_arg_to_step_callback<heyoka::step_callback<T>>(*cb_);

                    auto ret = [&]() {
                        // Release the GIL during propagation.
                        py::gil_scoped_release release;

                        return ta.propagate_for(delta_t, kw::max_steps = max_steps, kw::max_delta_t = max_delta_t,
                                                kw::write_tc = write_tc, kw::c_output = c_output,
                                                kw::callback = std::move(cb));
                    }();

                    return py::make_tuple(std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret),
                                          std::move(std::get<4>(ret)),
                                          step_callback_to_scb_arg_t(*cb_, std::get<5>(ret)));
                } else {
                    auto ret = [&]() {
                        py::gil_scoped_release release;

                        return ta.propagate_for(delta_t, kw::max_steps = max_steps, kw::max_delta_t = max_delta_t,
                                                kw::write_tc = write_tc, kw::c_output = c_output);
                    }();

                    return py::make_tuple(std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret),
                                          std::move(std::get<4>(ret)), py::none{});
                }
            },
            "delta_t"_a.noconvert(), "max_steps"_a = 0,
            "max_delta_t"_a.noconvert() = hey::detail::taylor_default_max_delta_t<T>(), "callback"_a = py::none{},
            "write_tc"_a = false, "c_output"_a = false)
        .def(
            "propagate_until",
            [](hey::taylor_adaptive<T> &ta, T t, std::size_t max_steps, T max_delta_t, std::optional<scb_arg_t> cb_,
               bool write_tc, bool c_output) {
                if (cb_) {
                    auto cb = scb_arg_to_step_callback<heyoka::step_callback<T>>(*cb_);

                    auto ret = [&]() {
                        py::gil_scoped_release release;

                        return ta.propagate_until(t, kw::max_steps = max_steps, kw::max_delta_t = max_delta_t,
                                                  kw::write_tc = write_tc, kw::c_output = c_output,
                                                  kw::callback = std::move(cb));
                    }();

                    return py::make_tuple(std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret),
                                          std::move(std::get<4>(ret)),
                                          step_callback_to_scb_arg_t(*cb_, std::get<5>(ret)));
                } else {
                    auto ret = [&]() {
                        py::gil_scoped_release release;

                        return ta.propagate_until(t, kw::max_steps = max_steps, kw::max_delta_t = max_delta_t,
                                                  kw::write_tc = write_tc, kw::c_output = c_output);
                    }();

                    return py::make_tuple(std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret),
                                          std::move(std::get<4>(ret)), py::none{});
                }
            },
            "t"_a.noconvert(), "max_steps"_a = 0,
            "max_delta_t"_a.noconvert() = hey::detail::taylor_default_max_delta_t<T>(), "callback"_a = py::none{},
            "write_tc"_a = false, "c_output"_a = false)
        .def(
            "propagate_grid",
            [](hey::taylor_adaptive<T> &ta, std::vector<T> grid, std::size_t max_steps, T max_delta_t,
               std::optional<scb_arg_t> cb_) {
                decltype(ta.propagate_grid(grid, max_steps)) ret;

                {
                    if (cb_) {
                        auto cb = scb_arg_to_step_callback<heyoka::step_callback<T>>(*cb_);

                        py::gil_scoped_release release;
                        ret = ta.propagate_grid(std::move(grid), kw::max_steps = max_steps,
                                                kw::max_delta_t = max_delta_t, kw::callback = std::move(cb));
                    } else {
                        py::gil_scoped_release release;
                        ret = ta.propagate_grid(std::move(grid), kw::max_steps = max_steps,
                                                kw::max_delta_t = max_delta_t);
                    }
                }

                // Determine the number of state vectors returned
                // (could be < grid.size() if errors arise).
                assert(std::get<5>(ret).size() % ta.get_dim() == 0u);
                const auto nrows = boost::numeric_cast<py::ssize_t>(std::get<5>(ret).size() / ta.get_dim());
                const auto ncols = boost::numeric_cast<py::ssize_t>(ta.get_dim());

                // Convert the output to a NumPy array.
                py::array a_ret(py::dtype(get_dtype<T>()), py::array::ShapeContainer{nrows, ncols},
                                std::get<5>(ret).data());

                if (cb_) {
                    return py::make_tuple(std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret),
                                          step_callback_to_scb_arg_t(*cb_, std::get<4>(ret)), std::move(a_ret));
                } else {
                    return py::make_tuple(std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret),
                                          py::none{}, std::move(a_ret));
                }
            },
            "grid"_a.noconvert(), "max_steps"_a = 0,
            "max_delta_t"_a.noconvert() = hey::detail::taylor_default_max_delta_t<T>(), "callback"_a = py::none{})
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
        // Variational-specific bits.
        .def_property_readonly("n_orig_sv", &hey::taylor_adaptive<T>::get_n_orig_sv)
        .def_property_readonly("is_variational", &hey::taylor_adaptive<T>::is_variational)
        .def_property_readonly("vargs", &hey::taylor_adaptive<T>::get_vargs)
        .def_property_readonly("vorder", &hey::taylor_adaptive<T>::get_vorder)
        .def_property_readonly("tstate", &fetch_tstate<T>)
        .def(
            "get_vslice",
            [](const hey::taylor_adaptive<T> &ta, std::uint32_t order, std::optional<std::uint32_t> component) {
                const auto ret = component ? ta.get_vslice(*component, order) : ta.get_vslice(order);

                return py::slice(boost::numeric_cast<py::ssize_t>(ret.first),
                                 boost::numeric_cast<py::ssize_t>(ret.second), {});
            },
            "order"_a, "component"_a = py::none{})
        .def(
            "get_mindex",
            [](const hey::taylor_adaptive<T> &ta, std::uint32_t i) {
                const auto &ret = ta.get_mindex(i);

                return dtens_t_it::sparse_to_dense(
                    ret, boost::numeric_cast<heyoka::dtens::v_idx_t::size_type>(ta.get_vargs().size()));
            },
            "i"_a)
        .def("eval_taylor_map",
             [](py::object &o, std::variant<py::array, py::iterable> in) {
                 auto *ta = py::cast<hey::taylor_adaptive<T> *>(o);

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
                 if (inputs.ndim() != 1) [[unlikely]] {
                     py_throw(PyExc_ValueError, fmt::format("The array of inputs provided for the evaluation "
                                                            "of a Taylor map has {} dimensions, "
                                                            "but it must have 1 dimension instead",
                                                            inputs.ndim())
                                                    .c_str());
                 }

                 // Validate the shape for the inputs.
                 if (boost::numeric_cast<std::uint32_t>(inputs.shape(0)) != ta->get_vargs().size()) [[unlikely]] {
                     py_throw(PyExc_ValueError, fmt::format("The array of inputs provided for the evaluation "
                                                            "of a Taylor map has {} elements, "
                                                            "but it must have {} elements instead",
                                                            inputs.shape(0), ta->get_vargs().size())
                                                    .c_str());
                 }

#if defined(HEYOKA_HAVE_REAL)

                 // Run the checks specific for mppp::real.
                 if constexpr (std::same_as<T, mppp::real>) {
                     // Check that the inputs array contains values with the correct precision.
                     pyreal_check_array(inputs, ta->get_prec());
                 }

#endif

                 // Run the evaluation.
                 const auto *data_ptr = static_cast<const T *>(inputs.data());
                 ta->eval_taylor_map(std::ranges::subrange(data_ptr, data_ptr + inputs.shape(0)));

                 // Return the tstate.
                 return fetch_tstate<T>(o);
             })
        // Event detection.
        .def_property_readonly("with_events", &hey::taylor_adaptive<T>::with_events)
        .def_property_readonly("te_cooldowns", &hey::taylor_adaptive<T>::get_te_cooldowns)
        .def("reset_cooldowns", &hey::taylor_adaptive<T>::reset_cooldowns)
        .def_property_readonly("t_events", &hey::taylor_adaptive<T>::get_t_events)
        .def_property_readonly("nt_events", &hey::taylor_adaptive<T>::get_nt_events);

    // Expose the llvm state getter.
    expose_llvm_state_property_ta(cl);

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::is_same_v<T, mppp::real>) {
        cl.def_property_readonly("prec", &hey::taylor_adaptive<T>::get_prec);
    }

#endif
}

} // namespace

} // namespace detail

void expose_taylor_integrator_flt(py::module &m)
{
    detail::expose_taylor_integrator_impl<float>(m, "flt");
}

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
    detail::expose_taylor_integrator_impl<mppp::real128>(m, "f128");
}

#endif

#if defined(HEYOKA_HAVE_REAL)

void expose_taylor_integrator_real(py::module &m)
{
    detail::expose_taylor_integrator_impl<mppp::real>(m, "real");
}

#endif

} // namespace heyoka_py
