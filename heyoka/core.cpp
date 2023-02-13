// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#include <oneapi/tbb/global_control.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define PY_ARRAY_UNIQUE_SYMBOL heyoka_py_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL heyoka_py_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/celmec/vsop2013.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/mascon.hpp>
#include <heyoka/math.hpp>
#include <heyoka/nbody.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>

#include "cfunc.hpp"
#include "custom_casters.hpp"
#include "dtypes.hpp"
#include "expose_batch_integrators.hpp"
#include "expose_expression.hpp"
#include "expose_real.hpp"
#include "expose_real128.hpp"
#include "logging.hpp"
#include "pickle_wrappers.hpp"
#include "setup_sympy.hpp"
#include "taylor_add_jet.hpp"
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
    namespace kw = hey::kw;

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

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const hey::zero_division_error &zde) {
            PyErr_SetString(PyExc_ZeroDivisionError, zde.what());
        }
    });

    // LLVM state.
    py::class_<hey::llvm_state>(m, "llvm_state", py::dynamic_attr{})
        .def("get_ir", &hey::llvm_state::get_ir)
        .def("get_object_code", [](hey::llvm_state &s) { return py::bytes(s.get_object_code()); })
        .def_property_readonly("opt_level", [](const hey::llvm_state &s) { return s.opt_level(); })
        .def_property_readonly("fast_math", [](const hey::llvm_state &s) { return s.fast_math(); })
        .def_property_readonly("force_avx512", [](const hey::llvm_state &s) { return s.force_avx512(); })
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
                        &heypy::pickle_setstate_wrapper<hey::llvm_state>));

    // Expression.
    heypy::expose_expression(m);

    // N-body builders.
    m.def(
        "make_nbody_sys",
        [](std::uint32_t n, const py::object &Gconst, std::optional<py::iterable> masses) {
            const auto G = heypy::to_number(Gconst);

            std::vector<hey::number> m_vec;
            if (masses) {
                for (auto ms : *masses) {
                    m_vec.push_back(heypy::to_number(ms));
                }
            } else {
                // If masses are not provided, all masses are 1.
                m_vec.resize(static_cast<decltype(m_vec.size())>(n), hey::number{1.});
            }

            return hey::make_nbody_sys(n, kw::Gconst = G, kw::masses = m_vec);
        },
        "n"_a, "Gconst"_a = 1., "masses"_a = py::none{});

    m.def(
        "make_np1body_sys",
        [](std::uint32_t n, const py::object &Gconst, std::optional<py::iterable> masses) {
            const auto G = heypy::to_number(Gconst);

            std::vector<hey::number> m_vec;
            if (masses) {
                for (auto ms : *masses) {
                    m_vec.push_back(heypy::to_number(ms));
                }
            } else {
                // If masses are not provided, all masses are 1.
                // NOTE: instead of computing n+1 here, do it in two
                // steps to avoid potential overflow issues.
                m_vec.resize(static_cast<decltype(m_vec.size())>(n), hey::number{1.});
                m_vec.emplace_back(1.);
            }

            return hey::make_np1body_sys(n, kw::Gconst = G, kw::masses = m_vec);
        },
        "n"_a, "Gconst"_a = 1., "masses"_a = py::none{});

    m.def(
        "make_nbody_par_sys",
        [](std::uint32_t n, const py::object &Gconst, std::optional<std::uint32_t> n_massive) {
            const auto G = heypy::to_number(Gconst);

            if (n_massive) {
                return hey::make_nbody_par_sys(n, kw::Gconst = G, kw::n_massive = *n_massive);
            } else {
                return hey::make_nbody_par_sys(n, kw::Gconst = G);
            }
        },
        "n"_a, "Gconst"_a = 1., "n_massive"_a = py::none{});

    // mascon dynamics builder
    m.def(
        "make_mascon_system",
        [](py::object Gconst, py::iterable points, py::iterable masses, py::iterable omega) {
            const auto G = heypy::to_number(Gconst);

            std::vector<std::vector<hey::expression>> points_vec;
            for (auto p : points) {
                std::vector<hey::expression> tmp;
                for (auto el : py::cast<py::iterable>(p)) {
                    tmp.emplace_back(heypy::to_number(el));
                }
                points_vec.emplace_back(tmp);
            }

            std::vector<hey::expression> mass_vec;
            for (auto ms : masses) {
                mass_vec.emplace_back(heypy::to_number(ms));
            }
            std::vector<hey::expression> omega_vec;
            for (auto w : omega) {
                omega_vec.emplace_back(heypy::to_number(w));
            }

            return hey::make_mascon_system(kw::Gconst = G, kw::points = points_vec, kw::masses = mass_vec,
                                           kw::omega = omega_vec);
        },
        "Gconst"_a, "points"_a, "masses"_a, "omega"_a);

    m.def(
        "energy_mascon_system",
        [](py::object Gconst, py::iterable state, py::iterable points, py::iterable masses, py::iterable omega) {
            const auto G = heypy::to_number(Gconst);

            std::vector<hey::expression> state_vec;
            for (auto s : state) {
                state_vec.emplace_back(heypy::to_number(s));
            }

            std::vector<std::vector<hey::expression>> points_vec;
            for (auto p : points) {
                std::vector<hey::expression> tmp;
                for (auto el : py::cast<py::iterable>(p)) {
                    tmp.emplace_back(heypy::to_number(el));
                }
                points_vec.emplace_back(tmp);
            }

            std::vector<hey::expression> mass_vec;
            for (auto ms : masses) {
                mass_vec.emplace_back(heypy::to_number(ms));
            }

            std::vector<hey::expression> omega_vec;
            for (auto w : omega) {
                omega_vec.emplace_back(heypy::to_number(w));
            }

            return hey::energy_mascon_system(kw::Gconst = G, kw::state = state_vec, kw::points = points_vec,
                                             kw::masses = mass_vec, kw::omega = omega_vec);
        },
        "Gconst"_a, "state"_a, "points"_a, "masses"_a, "omega"_a);

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

    // Computation of the jet of derivatives.
    heypy::expose_taylor_add_jet_dbl(m);
    heypy::expose_taylor_add_jet_ldbl(m);

#if defined(HEYOKA_HAVE_REAL128)

    heypy::expose_taylor_add_jet_f128(m);

#endif

#if defined(HEYOKA_HAVE_REAL)

    heypy::expose_taylor_add_jet_real(m);

#endif

    // Compiled functions.
    heypy::expose_add_cfunc_dbl(m);
    heypy::expose_add_cfunc_ldbl(m);

#if defined(HEYOKA_HAVE_REAL128)

    heypy::expose_add_cfunc_f128(m);

#endif

#if defined(HEYOKA_HAVE_REAL)

    heypy::expose_add_cfunc_real(m);

#endif

    // Expose the events.
    // NOTE: make sure these are exposed *before* the integrators.
    // Events are used in the integrator API (e.g., ctors), and in order
    // to have their types pretty-printed in the pybind11 machinery (e.g.,
    // when raising an error message about wrong types being passed to an
    // integrator's method), they need to be already exposed.
    heypy::expose_taylor_t_event_dbl(m);
    heypy::expose_taylor_t_event_ldbl(m);

#if defined(HEYOKA_HAVE_REAL128)

    heypy::expose_taylor_t_event_f128(m);

#endif

#if defined(HEYOKA_HAVE_REAL)

    heypy::expose_taylor_t_event_real(m);

#endif

    heypy::expose_taylor_nt_event_dbl(m);
    heypy::expose_taylor_nt_event_ldbl(m);

#if defined(HEYOKA_HAVE_REAL128)

    heypy::expose_taylor_nt_event_f128(m);

#endif

#if defined(HEYOKA_HAVE_REAL)

    heypy::expose_taylor_nt_event_real(m);

#endif

    // Batch mode.
    heypy::expose_taylor_nt_event_batch_dbl(m);
    heypy::expose_taylor_t_event_batch_dbl(m);

    // Scalar adaptive taylor integrators.
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
    m.def("_recommended_simd_size_dbl", &hey::recommended_simd_size<double>);

    // Setup the sympy integration bits.
    heypy::setup_sympy(m);

    // Expose the vsop2013 functions.
    m.def(
        "vsop2013_elliptic",
        [](std::uint32_t pl_idx, std::uint32_t var_idx, hey::expression t_expr, double thresh) {
            return hey::vsop2013_elliptic(pl_idx, var_idx, kw::time = std::move(t_expr), kw::thresh = thresh);
        },
        "pl_idx"_a, "var_idx"_a = 0, "time"_a = hey::time, "thresh"_a.noconvert() = 1e-9);
    m.def(
        "vsop2013_cartesian",
        [](std::uint32_t pl_idx, hey::expression t_expr, double thresh) {
            return hey::vsop2013_cartesian(pl_idx, kw::time = std::move(t_expr), kw::thresh = thresh);
        },
        "pl_idx"_a, "time"_a = hey::time, "thresh"_a.noconvert() = 1e-9);
    m.def(
        "vsop2013_cartesian_icrf",
        [](std::uint32_t pl_idx, hey::expression t_expr, double thresh) {
            return hey::vsop2013_cartesian_icrf(pl_idx, kw::time = std::move(t_expr), kw::thresh = thresh);
        },
        "pl_idx"_a, "time"_a = hey::time, "thresh"_a.noconvert() = 1e-9);
    m.def("get_vsop2013_mus", &hey::get_vsop2013_mus);

    // Expose the continuous output function objects.
    heypy::taylor_expose_c_output(m);

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
