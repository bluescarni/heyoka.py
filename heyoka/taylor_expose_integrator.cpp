// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <initializer_list>
#include <optional>
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

#include <heyoka/expression.hpp>
#include <heyoka/step_callback.hpp>
#include <heyoka/taylor.hpp>

#include "common_utils.hpp"
#include "custom_casters.hpp"
#include "dtypes.hpp"
#include "pickle_wrappers.hpp"
#include "step_cb_wrapper.hpp"
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
constexpr bool default_cm =
#if defined(HEYOKA_HAVE_REAL)
    std::is_same_v<T, mppp::real>
#else
    false
#endif
    ;

// Implementation of the exposition of the scalar integrators.
template <typename T>
void expose_taylor_integrator_impl(py::module &m, const std::string &suffix)
{
    using namespace py::literals;
    namespace kw = heyoka::kw;

    using t_ev_t = hey::t_event<T>;
    using nt_ev_t = hey::nt_event<T>;

    // Union of ODE system types, used in the ctor.
    using sys_t = std::variant<std::vector<std::pair<hey::expression, hey::expression>>, std::vector<hey::expression>>;

    py::class_<hey::taylor_adaptive<T>> cl(m, (fmt::format("taylor_adaptive_{}", suffix)).c_str(), py::dynamic_attr{});
    cl.def(py::init([](const sys_t &sys, std::vector<T> state, T time, std::vector<T> pars, T tol, bool high_accuracy,
                       bool compact_mode, std::vector<t_ev_t> tes, std::vector<nt_ev_t> ntes, bool parallel_mode,
                       unsigned opt_level, bool force_avx512, bool fast_math, long long prec) {
               return std::visit(
                   [&](const auto &val) {
                       // NOTE: GIL release is fine here even if the events contain
                       // Python objects, as the event vectors are moved in
                       // upon construction and thus we should never end up calling
                       // into the interpreter.
                       py::gil_scoped_release release;

                       return hey::taylor_adaptive<T>{val,
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
                                                      kw::fast_math = fast_math,
                                                      kw::prec = prec};
                   },
                   sys);
           }),
           "sys"_a, "state"_a.noconvert(), "time"_a.noconvert() = static_cast<T>(0), "pars"_a.noconvert() = py::list{},
           "tol"_a.noconvert() = static_cast<T>(0), "high_accuracy"_a = false, "compact_mode"_a = default_cm<T>,
           "t_events"_a = py::list{}, "nt_events"_a = py::list{}, "parallel_mode"_a = false,
           "opt_level"_a.noconvert() = 3, "force_avx512"_a.noconvert() = false, "fast_math"_a.noconvert() = false,
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
        .def_property_readonly("state_vars", &hey::taylor_adaptive<T>::get_state_vars)
        .def_property_readonly("rhs", &hey::taylor_adaptive<T>::get_rhs)
        .def_property("time", &hey::taylor_adaptive<T>::get_time, &hey::taylor_adaptive<T>::set_time)
        .def_property("dtime", &hey::taylor_adaptive<T>::get_dtime,
                      [](hey::taylor_adaptive<T> &ta, std::pair<double, double> p) { ta.set_dtime(p.first, p.second); })
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
               std::optional<py::object> &cb_, bool write_tc, bool c_output) {
                // NOTE: after releasing the GIL, the only potential
                // calls into the Python interpreter are when invoking the event or
                // step callbacks (which are all protected by GIL reacquire).

                if (cb_) {
                    // NOTE: because cb is a step_callback, it will be passed by reference
                    // into the propagate_for() function. Thus, no copies are made and no
                    // calling into the Python interpreter takes place (and no need to hold
                    // the GIL).
                    // NOTE: because cb is created before the GIL scoped releaser, it will be
                    // destroyed *after* the GIL has been re-acquired. Thus, the reference
                    // count decrease associated with the destructor is safe.
                    auto cb = hey::step_callback<T>(step_cb_wrapper(*cb_));

                    py::gil_scoped_release release;
                    return ta.propagate_for(delta_t, kw::max_steps = max_steps, kw::max_delta_t = max_delta_t,
                                            kw::write_tc = write_tc, kw::c_output = c_output, kw::callback = cb);
                } else {
                    py::gil_scoped_release release;
                    return ta.propagate_for(delta_t, kw::max_steps = max_steps, kw::max_delta_t = max_delta_t,
                                            kw::write_tc = write_tc, kw::c_output = c_output);
                }
            },
            "delta_t"_a.noconvert(), "max_steps"_a = 0,
            "max_delta_t"_a.noconvert() = hey::detail::taylor_default_max_delta_t<T>(), "callback"_a = py::none{},
            "write_tc"_a = false, "c_output"_a = false)
        .def(
            "propagate_until",
            [](hey::taylor_adaptive<T> &ta, T t, std::size_t max_steps, T max_delta_t, std::optional<py::object> &cb_,
               bool write_tc, bool c_output) {
                if (cb_) {
                    auto cb = hey::step_callback<T>(step_cb_wrapper(*cb_));

                    py::gil_scoped_release release;
                    return ta.propagate_until(t, kw::max_steps = max_steps, kw::max_delta_t = max_delta_t,
                                              kw::write_tc = write_tc, kw::c_output = c_output, kw::callback = cb);
                } else {
                    py::gil_scoped_release release;
                    return ta.propagate_until(t, kw::max_steps = max_steps, kw::max_delta_t = max_delta_t,
                                              kw::write_tc = write_tc, kw::c_output = c_output);
                }
            },
            "t"_a.noconvert(), "max_steps"_a = 0,
            "max_delta_t"_a.noconvert() = hey::detail::taylor_default_max_delta_t<T>(), "callback"_a = py::none{},
            "write_tc"_a = false, "c_output"_a = false)
        .def(
            "propagate_grid",
            [](hey::taylor_adaptive<T> &ta, std::vector<T> grid, std::size_t max_steps, T max_delta_t,
               std::optional<py::object> &cb_) {
                decltype(ta.propagate_grid(grid, max_steps)) ret;

                {
                    if (cb_) {
                        auto cb = hey::step_callback<T>(step_cb_wrapper(*cb_));

                        py::gil_scoped_release release;
                        ret = ta.propagate_grid(std::move(grid), kw::max_steps = max_steps,
                                                kw::max_delta_t = max_delta_t, kw::callback = cb);
                    } else {
                        py::gil_scoped_release release;
                        ret = ta.propagate_grid(std::move(grid), kw::max_steps = max_steps,
                                                kw::max_delta_t = max_delta_t);
                    }
                }

                // Determine the number of state vectors returned
                // (could be < grid.size() if errors arise).
                assert(std::get<4>(ret).size() % ta.get_dim() == 0u);
                const auto nrows = boost::numeric_cast<py::ssize_t>(std::get<4>(ret).size() / ta.get_dim());
                const auto ncols = boost::numeric_cast<py::ssize_t>(ta.get_dim());

                // Convert the output to a NumPy array.
                py::array a_ret(py::dtype(get_dtype<T>()), py::array::ShapeContainer{nrows, ncols},
                                std::get<4>(ret).data());

                return py::make_tuple(std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret),
                                      std::move(a_ret));
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
        // Event detection.
        .def_property_readonly("with_events", &hey::taylor_adaptive<T>::with_events)
        .def_property_readonly("te_cooldowns", &hey::taylor_adaptive<T>::get_te_cooldowns)
        .def("reset_cooldowns", &hey::taylor_adaptive<T>::reset_cooldowns)
        .def_property_readonly("t_events", &hey::taylor_adaptive<T>::get_t_events)
        .def_property_readonly("nt_events", &hey::taylor_adaptive<T>::get_nt_events);

    // Expose the llvm state getter.
    expose_llvm_state_property(cl);

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::is_same_v<T, mppp::real>) {
        cl.def_property_readonly("prec", &hey::taylor_adaptive<T>::get_prec);
    }

#endif
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
