// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cstddef>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <sstream>
#include <type_traits>

#include <oneapi/tbb/global_control.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define PY_ARRAY_UNIQUE_SYMBOL heyoka_py_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL heyoka_py_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_1_22_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/hamiltonian.hpp>
#include <heyoka/lagrangian.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>

#include "cfunc.hpp"
#include "custom_casters.hpp"
#include "docstrings.hpp"
#include "dtypes.hpp"
#include "expose_batch_integrators.hpp"
#include "expose_callbacks.hpp"
#include "expose_expression.hpp"
#include "expose_models.hpp"
#include "expose_real.hpp"
#include "expose_real128.hpp"
#include "expose_var_ode_sys.hpp"
#include "logging.hpp"
#include "pickle_wrappers.hpp"
#include "setup_sympy.hpp"
#include "taylor_expose_c_output.hpp"
#include "taylor_expose_events.hpp"
#include "taylor_expose_integrator.hpp"

namespace py = pybind11;
namespace hey = heyoka;
namespace heypy = heyoka_py;

namespace heyoka_py::detail
{

namespace
{

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::optional<oneapi::tbb::global_control> tbb_gc;

// Helper to import the NumPy API bits.
PyObject *import_numpy(PyObject *m)
{
    import_array();

    import_umath();

    return m;
}

} // namespace

} // namespace heyoka_py::detail

#if defined(__clang__)

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"

#endif

PYBIND11_MODULE(core, m)
{
    using namespace pybind11::literals;
    namespace docstrings = heypy::docstrings;

    // Disable automatic function signatures in the docs.
    // NOTE: the 'options' object needs to stay alive
    // throughout the whole definition of the module.
    py::options options;
    options.disable_function_signatures();

    // Import the NumPy API bits.
    if (heypy::detail::import_numpy(m.ptr()) == nullptr) {
        // NOTE: on failure, the NumPy macros already set
        // the error indicator. Thus, all it is left to do
        // is to throw the pybind11 exception.
        throw py::error_already_set();
    }

    m.doc() = "The core heyoka module";

    // Flag PPC arch.
    m.attr("_ppc_arch") =
#if defined(HEYOKA_ARCH_PPC)
        true
#else
        false
#endif
        ;

    // Expose the real128 type.
    // NOTE: it is *important* this is done
    // before the exposition of real, since the real
    // exposition registers NumPy converters to/from
    // real128, and in order for that to work the NumPy
    // type descriptor for real128 needs to have been
    // set up.
    heypy::expose_real128(m);

    // Expose the real type.
    heypy::expose_real(m);

    // Expose the logging setter functions.
    heypy::expose_logging_setters(m);

    // Export the heyoka version.
    m.attr("_heyoka_cpp_version_major") = HEYOKA_VERSION_MAJOR;
    m.attr("_heyoka_cpp_version_minor") = HEYOKA_VERSION_MINOR;
    m.attr("_heyoka_cpp_version_patch") = HEYOKA_VERSION_PATCH;

    // Register heyoka's custom exceptions.
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const hey::not_implemented_error &nie) {
            PyErr_SetString(PyExc_NotImplementedError, nie.what());
        }
    });

    // code_model enum.
    py::enum_<hey::code_model>(m, "code_model", docstrings::code_model().c_str())
        .value("tiny", hey::code_model::tiny, docstrings::code_model_tiny().c_str())
        .value("small", hey::code_model::small, docstrings::code_model_small().c_str())
        .value("kernel", hey::code_model::kernel, docstrings::code_model_kernel().c_str())
        .value("medium", hey::code_model::medium, docstrings::code_model_medium().c_str())
        .value("large", hey::code_model::large, docstrings::code_model_large().c_str());

    // LLVM state.
    py::class_<hey::llvm_state>(m, "llvm_state", py::dynamic_attr{})
        .def("get_ir", &hey::llvm_state::get_ir)
        .def("get_bc", [](const hey::llvm_state &s) { return py::bytes(s.get_bc()); })
        .def("get_object_code", [](const hey::llvm_state &s) { return py::bytes(s.get_object_code()); })
        .def_property_readonly("opt_level", &hey::llvm_state::get_opt_level)
        .def_property_readonly("fast_math", [](const hey::llvm_state &s) { return s.fast_math(); })
        .def_property_readonly("force_avx512", [](const hey::llvm_state &s) { return s.force_avx512(); })
        .def_property_readonly("slp_vectorize", &hey::llvm_state::get_slp_vectorize)
        .def_property_readonly("code_model", &hey::llvm_state::get_code_model)
        // Repr.
        .def("__repr__",
             [](const hey::llvm_state &s) {
                 std::ostringstream oss;
                 oss << s;
                 return oss.str();
             })
        // Copy/deepcopy.
        .def("__copy__", heypy::copy_wrapper<hey::llvm_state>)
        .def("__deepcopy__", heypy::deepcopy_wrapper<hey::llvm_state>, "memo"_a)
        // Pickle support.
        .def(py::pickle(&heypy::pickle_getstate_wrapper<hey::llvm_state>,
                        &heypy::pickle_setstate_wrapper<hey::llvm_state>))
        // Cache management.
        .def_property_readonly_static("memcache_size",
                                      [](const py::object &) { return hey::llvm_state::get_memcache_size(); })
        .def_property_static(
            "memcache_limit", [](const py::object &) { return hey::llvm_state::get_memcache_limit(); },
            [](const py::object &, std::size_t limit) { hey::llvm_state::set_memcache_limit(limit); })
        .def_static("clear_memcache", &hey::llvm_state::clear_memcache);

    // LLVM multi state.
    py::class_<hey::llvm_multi_state>(m, "llvm_multi_state", py::dynamic_attr{})
        .def("get_ir", &hey::llvm_multi_state::get_ir)
        .def("get_bc",
             [](const hey::llvm_multi_state &s) {
                 py::list ret;

                 for (const auto &cur_bc : s.get_bc()) {
                     ret.append(py::bytes(cur_bc));
                 }

                 return ret;
             })
        .def("get_object_code",
             [](const hey::llvm_multi_state &s) {
                 py::list ret;

                 for (const auto &cur_ob : s.get_object_code()) {
                     ret.append(py::bytes(cur_ob));
                 }

                 return ret;
             })
        .def_property_readonly("opt_level", &hey::llvm_multi_state::get_opt_level)
        .def_property_readonly("fast_math", [](const hey::llvm_multi_state &s) { return s.fast_math(); })
        .def_property_readonly("force_avx512", [](const hey::llvm_multi_state &s) { return s.force_avx512(); })
        .def_property_readonly("slp_vectorize", &hey::llvm_multi_state::get_slp_vectorize)
        .def_property_readonly("parjit", &hey::llvm_multi_state::get_parjit)
        .def_property_readonly("code_model", &hey::llvm_multi_state::get_code_model)
        // Repr.
        .def("__repr__",
             [](const hey::llvm_multi_state &s) {
                 std::ostringstream oss;
                 oss << s;
                 return oss.str();
             })
        // Copy/deepcopy.
        .def("__copy__", heypy::copy_wrapper<hey::llvm_multi_state>)
        .def("__deepcopy__", heypy::deepcopy_wrapper<hey::llvm_multi_state>, "memo"_a)
        // Pickle support.
        .def(py::pickle(&heypy::pickle_getstate_wrapper<hey::llvm_multi_state>,
                        &heypy::pickle_setstate_wrapper<hey::llvm_multi_state>));

    // Expression.
    heypy::expose_expression(m);

    // var_ode_sys.
    heypy::expose_var_ode_sys(m);

    // Models.
    heypy::expose_models(m);

    // Lagrangian/Hamiltonian.
    m.def("lagrangian", &hey::lagrangian, "L"_a, "qs"_a, "qdots"_a, "D"_a = hey::expression{0.},
          docstrings::lagrangian().c_str());
    m.def("hamiltonian", &hey::hamiltonian, "H"_a, "qs"_a, "ps"_a, docstrings::hamiltonian().c_str());

    // taylor_outcome enum.
    py::enum_<hey::taylor_outcome>(m, "taylor_outcome", py::arithmetic())
        .value("success", hey::taylor_outcome::success)
        .value("step_limit", hey::taylor_outcome::step_limit)
        .value("time_limit", hey::taylor_outcome::time_limit)
        .value("err_nf_state", hey::taylor_outcome::err_nf_state)
        .value("cb_stop", hey::taylor_outcome::cb_stop);

    // event_direction enum.
    py::enum_<hey::event_direction>(m, "event_direction")
        .value("any", hey::event_direction::any)
        .value("positive", hey::event_direction::positive)
        .value("negative", hey::event_direction::negative);

    // Compiled functions.
    heypy::expose_add_cfunc_flt(m);
    heypy::expose_add_cfunc_dbl(m);
    heypy::expose_add_cfunc_ldbl(m);

#if defined(HEYOKA_HAVE_REAL128)

    heypy::expose_add_cfunc_f128(m);

#endif

#if defined(HEYOKA_HAVE_REAL)

    heypy::expose_add_cfunc_real(m);

#endif

    // Expose the continuous output function objects.
    // NOTE: do it *before* the integrators so that
    // the type name of the continuous output function
    // objects shows up correctly in the signature of the
    // propagate_*() functions.
    heypy::taylor_expose_c_output(m);

    // Expose the events.
    // NOTE: make sure these are exposed *before* the integrators.
    // Events are used in the integrator API (e.g., ctors), and in order
    // to have their types pretty-printed in the pybind11 machinery (e.g.,
    // when raising an error message about wrong types being passed to an
    // integrator's method), they need to be already exposed.
    heypy::expose_taylor_t_event_flt(m);
    heypy::expose_taylor_t_event_dbl(m);
    heypy::expose_taylor_t_event_ldbl(m);

#if defined(HEYOKA_HAVE_REAL128)

    heypy::expose_taylor_t_event_f128(m);

#endif

#if defined(HEYOKA_HAVE_REAL)

    heypy::expose_taylor_t_event_real(m);

#endif

    heypy::expose_taylor_nt_event_flt(m);
    heypy::expose_taylor_nt_event_dbl(m);
    heypy::expose_taylor_nt_event_ldbl(m);

#if defined(HEYOKA_HAVE_REAL128)

    heypy::expose_taylor_nt_event_f128(m);

#endif

#if defined(HEYOKA_HAVE_REAL)

    heypy::expose_taylor_nt_event_real(m);

#endif

    // Batch mode.
    heypy::expose_taylor_nt_event_batch_flt(m);
    heypy::expose_taylor_t_event_batch_flt(m);
    heypy::expose_taylor_nt_event_batch_dbl(m);
    heypy::expose_taylor_t_event_batch_dbl(m);

    // Expose the callbacks.
    heypy::expose_callbacks(m);

    // Scalar adaptive taylor integrators.
    heypy::expose_taylor_integrator_flt(m);
    heypy::expose_taylor_integrator_dbl(m);
    heypy::expose_taylor_integrator_ldbl(m);

#if defined(HEYOKA_HAVE_REAL128)

    heypy::expose_taylor_integrator_f128(m);

#endif

#if defined(HEYOKA_HAVE_REAL)

    heypy::expose_taylor_integrator_real(m);

#endif

    // Batch integrators.
    heypy::expose_batch_integrators(m);

    // Recommended simd size helper.
    m.def("_recommended_simd_size_flt", &hey::recommended_simd_size<float>);
    m.def("_recommended_simd_size_dbl", &hey::recommended_simd_size<double>);

    // Setup the sympy integration bits.
    heypy::setup_sympy(m);

    // Expose the helpers to get/set the number of threads in use by heyoka.py.
    // NOTE: these are not thread-safe themselves. Should they be? If not, their
    // thread unsafety must be highlighted in the docs.
    m.def("set_nthreads", [](std::size_t n) {
        if (n == 0u) {
            heypy::detail::tbb_gc.reset();
        } else {
            heypy::detail::tbb_gc.emplace(oneapi::tbb::global_control::max_allowed_parallelism, n);
        }
    });

    m.def("get_nthreads", []() {
        return oneapi::tbb::global_control::active_value(oneapi::tbb::global_control::max_allowed_parallelism);
    });

    // Make sure the TBB control structure is cleaned
    // up before shutdown.
    auto atexit = py::module_::import("atexit");
    atexit.attr("register")(py::cpp_function([]() {
#if !defined(NDEBUG)
        std::cout << "Cleaning up the TBB control structure" << std::endl;
#endif
        heypy::detail::tbb_gc.reset();
    }));
}

#if defined(__clang__)

#pragma clang diagnostic pop

#endif
