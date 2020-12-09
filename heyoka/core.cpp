// Copyright 2020 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Python.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/extra/pybind11.hpp>
#include <mp++/real128.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/mascon.hpp>
#include <heyoka/math_functions.hpp>
#include <heyoka/nbody.hpp>
#include <heyoka/number.hpp>
#include <heyoka/square.hpp>
#include <heyoka/taylor.hpp>

#include "common_utils.hpp"
#include "long_double_caster.hpp"

namespace py = pybind11;
namespace hey = heyoka;
namespace heypy = heyoka_py;

PYBIND11_MODULE(core, m)
{
#if defined(HEYOKA_HAVE_REAL128)
    // Init the pybind11 integration for this module.
    mppp_pybind11::init();
#endif

    using namespace pybind11::literals;

    m.doc() = "The core heyoka module";

    // Flag the presence of real128.
    m.attr("with_real128") =
#if defined(HEYOKA_HAVE_REAL128)
        true
#else
        false
#endif
        ;

    // Export the heyoka version.
    m.attr("_heyoka_cpp_version_major") = HEYOKA_VERSION_MAJOR;
    m.attr("_heyoka_cpp_version_minor") = HEYOKA_VERSION_MINOR;
    m.attr("_heyoka_cpp_version_patch") = HEYOKA_VERSION_PATCH;

    // NOTE: typedef to avoid complications in the
    // exposition of the operators.
    using ld_t = long double;

    py::class_<hey::expression>(m, "expression")
        .def(py::init<>())
        .def(py::init<double>())
        .def(py::init<long double>())
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::init<mppp::real128>())
#endif
        .def(py::init<std::string>())
        // Unary operators.
        .def(-py::self)
        .def(+py::self)
        // Binary operators.
        .def(py::self + py::self)
        .def(py::self + double())
        .def(double() + py::self)
        .def(py::self + ld_t())
        .def(ld_t() + py::self)
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self + mppp::real128())
        .def(mppp::real128() + py::self)
#endif
        .def(py::self - py::self)
        .def(py::self - double())
        .def(double() - py::self)
        .def(py::self - ld_t())
        .def(ld_t() - py::self)
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self - mppp::real128())
        .def(mppp::real128() - py::self)
#endif
        .def(py::self * py::self)
        .def(py::self * double())
        .def(double() * py::self)
        .def(py::self * ld_t())
        .def(ld_t() * py::self)
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self * mppp::real128())
        .def(mppp::real128() * py::self)
#endif
        .def(py::self / py::self)
        .def(py::self / double())
        .def(double() / py::self)
        .def(py::self / ld_t())
        .def(ld_t() / py::self)
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self / mppp::real128())
        .def(mppp::real128() / py::self)
#endif
        // In-place operators.
        .def(py::self += py::self)
        .def(py::self += double())
        .def(py::self += ld_t())
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self += mppp::real128())
#endif
        .def(py::self -= py::self)
        .def(py::self -= double())
        .def(py::self -= ld_t())
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self -= mppp::real128())
#endif
        .def(py::self *= py::self)
        .def(py::self *= double())
        .def(py::self *= ld_t())
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self *= mppp::real128())
#endif
        .def(py::self /= py::self)
        .def(py::self /= double())
        .def(py::self /= ld_t())
#if defined(HEYOKA_HAVE_REAL128)
        .def(py::self /= mppp::real128())
#endif
        // Comparisons.
        .def(py::self == py::self)
        .def(py::self != py::self)
        // pow().
        .def("__pow__", [](const hey::expression &b, const hey::expression &e) { return hey::pow(b, e); })
        .def("__pow__", [](const hey::expression &b, double e) { return hey::pow(b, e); })
        .def("__pow__", [](const hey::expression &b, long double e) { return hey::pow(b, e); })
#if defined(HEYOKA_HAVE_REAL128)
        .def("__pow__", [](const hey::expression &b, mppp::real128 e) { return hey::pow(b, e); })
#endif
        // Repr.
        .def("__repr__",
             [](const hey::expression &e) {
                 std::ostringstream oss;
                 oss << e;
                 return oss.str();
             })
        // Copy/deepcopy.
        .def("__copy__", [](const hey::expression &e) { return e; })
        .def(
            "__deepcopy__", [](const hey::expression &e, py::dict) { return e; }, "memo"_a);

    // Pairwise sum.
    m.def("pairwise_sum", [](const std::vector<hey::expression> &v_ex) { return hey::pairwise_sum(v_ex); });

    // make_vars() helper.
    m.def("make_vars", [](py::args v_str) {
        py::list retval;
        for (const auto &o : v_str) {
            retval.append(hey::expression(py::cast<std::string>(o)));
        }
        return retval;
    });

    // Math functions.
    m.def("sin", &hey::sin);
    m.def("cos", &hey::cos);
    m.def("log", &hey::log);
    m.def("exp", &hey::exp);
    m.def("sqrt", &hey::sqrt);
    m.def("square", &hey::square);

    // N-body builder.
    m.def(
        "make_nbody_sys",
        [](std::uint32_t n, py::object Gconst, py::iterable masses) {
            const auto G = heypy::to_number(Gconst);

            std::vector<hey::number> m_vec;
            if (masses.is_none()) {
                // If masses are not provided, all masses are 1.
                m_vec.resize(static_cast<decltype(m_vec.size())>(n), hey::number{1.});
            } else {
                for (const auto &ms : masses) {
                    m_vec.push_back(heypy::to_number(ms));
                }
            }

            namespace kw = hey::kw;
            return hey::make_nbody_sys(n, kw::Gconst = G, kw::masses = m_vec);
        },
        "n"_a, "Gconst"_a = py::cast(1.), "masses"_a = py::none{});

    // mascon dynamics builder
    m.def(
        "make_mascon_system",
        [](py::object Gconst, py::iterable points, py::iterable masses, py::iterable omega) {
            const auto G = heypy::to_number(Gconst);

            std::vector<std::vector<hey::expression>> points_vec;
            for (const auto &p : points) {
                std::vector<hey::expression> tmp;
                for (const auto &el : py::cast<py::iterable>(p)) {
                    tmp.emplace_back(heypy::to_number(el));
                }
                points_vec.emplace_back(tmp);
            }

            std::vector<hey::expression> mass_vec;
            for (const auto &ms : masses) {
                mass_vec.emplace_back(heypy::to_number(ms));
            }
            std::vector<hey::expression> omega_vec;
            for (const auto &w : omega) {
                omega_vec.emplace_back(heypy::to_number(w));
            }
            namespace kw = hey::kw;
            return hey::make_mascon_system(kw::Gconst = G, kw::points = points_vec, kw::masses = mass_vec,
                                           kw::omega = omega_vec);
        },
        "Gconst"_a, "points"_a, "masses"_a, "omega"_a);

    m.def(
        "energy_mascon_system",
        [](py::object Gconst, py::iterable state, py::iterable points, py::iterable masses, py::iterable omega) {
            const auto G = heypy::to_number(Gconst);

            std::vector<hey::expression> state_vec;
            for (const auto &s : state) {
                state_vec.emplace_back(heypy::to_number(s));
            }

            std::vector<std::vector<hey::expression>> points_vec;
            for (const auto &p : points) {
                std::vector<hey::expression> tmp;
                for (const auto &el : py::cast<py::iterable>(p)) {
                    tmp.emplace_back(heypy::to_number(el));
                }
                points_vec.emplace_back(tmp);
            }

            std::vector<hey::expression> mass_vec;
            for (const auto &ms : masses) {
                mass_vec.emplace_back(heypy::to_number(ms));
            }

            std::vector<hey::expression> omega_vec;
            for (const auto &w : omega) {
                omega_vec.emplace_back(heypy::to_number(w));
            }
            namespace kw = hey::kw;
            return hey::energy_mascon_system(kw::Gconst = G, kw::state = state_vec, kw::points = points_vec,
                                             kw::masses = mass_vec, kw::omega = omega_vec);
        },
        "Gconst"_a, "state"_a, "points"_a, "masses"_a, "omega"_a);

    // Elliptic orbit generator.
    m.def(
        "random_elliptic_state",
        [](double mu, const std::array<std::pair<double, double>, 6> &bounds, py::object seed) {
            const auto retval = seed.is_none() ? hey::random_elliptic_state(mu, bounds)
                                               : hey::random_elliptic_state(mu, bounds, py::cast<unsigned>(seed));

            return py::array_t<double>({6}, retval.data());
        },
        "mu"_a, "bounds"_a, "seed"_a = py::none{});

    // Conversion from cartesian state to orbital elements.
    m.def(
        "cartesian_to_oe",
        [](double mu, const std::array<double, 6> &s) {
            const auto retval = hey::cartesian_to_oe(mu, s);

            return py::array_t<double>({6}, retval.data());
        },
        "mu"_a, "s"_a);

    // taylor_outcome enum.
    py::enum_<hey::taylor_outcome>(m, "taylor_outcome")
        .value("success", hey::taylor_outcome::success)
        .value("step_limit", hey::taylor_outcome::step_limit)
        .value("time_limit", hey::taylor_outcome::time_limit)
        .value("err_nf_state", hey::taylor_outcome::err_nf_state)
        .export_values();

    // Adaptive taylor integrators.
    auto tad_ctor_impl
        = [](const auto &sys, py::iterable state, double time, double tol, bool high_accuracy, bool compact_mode) {
              // Build the initial state vector.
              std::vector<double> s_vector;
              for (const auto &x : state) {
                  s_vector.push_back(py::cast<double>(x));
              }

              namespace kw = hey::kw;
              return hey::taylor_adaptive<double>{sys,
                                                  std::move(s_vector),
                                                  kw::time = time,
                                                  kw::tol = tol,
                                                  kw::high_accuracy = high_accuracy,
                                                  kw::compact_mode = compact_mode};
          };
    py::class_<hey::taylor_adaptive<double>>(m, "taylor_adaptive_double")
        .def(py::init([tad_ctor_impl](const std::vector<std::pair<hey::expression, hey::expression>> &sys,
                                      py::iterable state, double time, double tol, bool high_accuracy,
                                      bool compact_mode) {
                 return tad_ctor_impl(sys, state, time, tol, high_accuracy, compact_mode);
             }),
             "sys"_a, "state"_a, "time"_a = 0., "tol"_a = 0., "high_accuracy"_a = false, "compact_mode"_a = false)
        .def(py::init([tad_ctor_impl](const std::vector<hey::expression> &sys, py::iterable state, double time,
                                      double tol, bool high_accuracy, bool compact_mode) {
                 return tad_ctor_impl(sys, state, time, tol, high_accuracy, compact_mode);
             }),
             "sys"_a, "state"_a, "time"_a = 0., "tol"_a = 0., "high_accuracy"_a = false, "compact_mode"_a = false)
        .def("get_decomposition", &hey::taylor_adaptive<double>::get_decomposition)
        .def("step", [](hey::taylor_adaptive<double> &ta) { return ta.step(); })
        .def(
            "step", [](hey::taylor_adaptive<double> &ta, double max_delta_t) { return ta.step(max_delta_t); },
            "max_delta_t"_a)
        .def("step_backward", [](hey::taylor_adaptive<double> &ta) { return ta.step_backward(); })
        .def(
            "propagate_for",
            [](hey::taylor_adaptive<double> &ta, double delta_t, std::size_t max_steps) {
                py::gil_scoped_release release;
                return ta.propagate_for(delta_t, max_steps);
            },
            "delta_t"_a, "max_steps"_a = 0)
        .def(
            "propagate_until",
            [](hey::taylor_adaptive<double> &ta, double t, std::size_t max_steps) {
                py::gil_scoped_release release;
                return ta.propagate_until(t, max_steps);
            },
            "t"_a, "max_steps"_a = 0)
        .def_property("time", &hey::taylor_adaptive<double>::get_time, &hey::taylor_adaptive<double>::set_time)
        .def_property_readonly("state",
                               [](py::object &o) {
                                   auto *ta = py::cast<hey::taylor_adaptive<double> *>(o);
                                   return py::array_t<double>(
                                       {boost::numeric_cast<py::ssize_t>(ta->get_state().size())}, ta->get_state_data(),
                                       o);
                               })
        .def_property_readonly("order", &hey::taylor_adaptive<double>::get_order)
        .def_property_readonly("dim", &hey::taylor_adaptive<double>::get_dim)
        // Repr.
        .def("__repr__",
             [](const hey::taylor_adaptive<double> &ta) {
                 std::ostringstream oss;
                 oss << ta;
                 return oss.str();
             })
        // Copy/deepcopy.
        .def("__copy__",
             [](const hey::taylor_adaptive<double> &ta) {
                 py::gil_scoped_release release;
                 return ta;
             })
        .def(
            "__deepcopy__",
            [](const hey::taylor_adaptive<double> &ta, py::dict) {
                py::gil_scoped_release release;
                return ta;
            },
            "memo"_a);

    auto tald_ctor_impl = [](const auto &sys, py::iterable state, long double time, long double tol, bool high_accuracy,
                             bool compact_mode) {
        // Build the initial state vector.
        std::vector<long double> s_vector;
        for (const auto &x : state) {
            s_vector.push_back(py::cast<long double>(x));
        }

        namespace kw = hey::kw;
        return hey::taylor_adaptive<long double>{sys,
                                                 std::move(s_vector),
                                                 kw::time = time,
                                                 kw::tol = tol,
                                                 kw::high_accuracy = high_accuracy,
                                                 kw::compact_mode = compact_mode};
    };
    py::class_<hey::taylor_adaptive<long double>>(m, "taylor_adaptive_long_double")
        .def(py::init([tald_ctor_impl](const std::vector<std::pair<hey::expression, hey::expression>> &sys,
                                       py::iterable state, long double time, long double tol, bool high_accuracy,
                                       bool compact_mode) {
                 return tald_ctor_impl(sys, state, time, tol, high_accuracy, compact_mode);
             }),
             "sys"_a, "state"_a, "time"_a = 0.l, "tol"_a = 0.l, "high_accuracy"_a = false, "compact_mode"_a = false)
        .def(py::init([tald_ctor_impl](const std::vector<hey::expression> &sys, py::iterable state, long double time,
                                       long double tol, bool high_accuracy, bool compact_mode) {
                 return tald_ctor_impl(sys, state, time, tol, high_accuracy, compact_mode);
             }),
             "sys"_a, "state"_a, "time"_a = 0.l, "tol"_a = 0.l, "high_accuracy"_a = false, "compact_mode"_a = false)
        .def("get_decomposition", &hey::taylor_adaptive<long double>::get_decomposition)
        .def("step", [](hey::taylor_adaptive<long double> &ta) { return ta.step(); })
        .def(
            "step", [](hey::taylor_adaptive<long double> &ta, long double max_delta_t) { return ta.step(max_delta_t); },
            "max_delta_t"_a)
        .def("step_backward", [](hey::taylor_adaptive<long double> &ta) { return ta.step_backward(); })
        .def(
            "propagate_for",
            [](hey::taylor_adaptive<long double> &ta, long double delta_t, std::size_t max_steps) {
                py::gil_scoped_release release;
                return ta.propagate_for(delta_t, max_steps);
            },
            "delta_t"_a, "max_steps"_a = 0)
        .def(
            "propagate_until",
            [](hey::taylor_adaptive<long double> &ta, long double t, std::size_t max_steps) {
                py::gil_scoped_release release;
                return ta.propagate_until(t, max_steps);
            },
            "t"_a, "max_steps"_a = 0)
        .def_property("time", &hey::taylor_adaptive<long double>::get_time,
                      &hey::taylor_adaptive<long double>::set_time)
        .def_property_readonly("state",
                               [](py::object &o) {
                                   auto *ta = py::cast<hey::taylor_adaptive<long double> *>(o);
                                   return py::array_t<long double>(
                                       {boost::numeric_cast<py::ssize_t>(ta->get_state().size())}, ta->get_state_data(),
                                       o);
                               })
        .def_property_readonly("order", &hey::taylor_adaptive<long double>::get_order)
        .def_property_readonly("dim", &hey::taylor_adaptive<long double>::get_dim)
        // Repr.
        .def("__repr__",
             [](const hey::taylor_adaptive<long double> &ta) {
                 std::ostringstream oss;
                 oss << ta;
                 return oss.str();
             })
        // Copy/deepcopy.
        .def("__copy__",
             [](const hey::taylor_adaptive<long double> &ta) {
                 py::gil_scoped_release release;
                 return ta;
             })
        .def(
            "__deepcopy__",
            [](const hey::taylor_adaptive<long double> &ta, py::dict) {
                py::gil_scoped_release release;
                return ta;
            },
            "memo"_a);

#if defined(HEYOKA_HAVE_REAL128)
    py::class_<hey::taylor_adaptive<mppp::real128>>(m, "taylor_adaptive_real128")
        .def(py::init([](const std::vector<std::pair<hey::expression, hey::expression>> &sys, py::iterable state) {
            // Build the initial state vector.
            std::vector<mppp::real128> s_vector;
            for (const auto &x : state) {
                s_vector.push_back(py::cast<mppp::real128>(x));
            }

            return hey::taylor_adaptive<mppp::real128>{sys, std::move(s_vector)};
        }))
        // TODO return an array somehow?
        .def("get_state", [](const hey::taylor_adaptive<mppp::real128> &ta) { return ta.get_state(); });
#endif

    auto tabd_ctor_impl = [](auto sys, std::vector<double> state, std::uint32_t batch_size, py::object time, double tol,
                             bool high_accuracy, bool compact_mode) {
        namespace kw = hey::kw;

        if (time.is_none()) {
            // Times not provided.
            return hey::taylor_adaptive_batch<double>{std::move(sys),
                                                      std::move(state),
                                                      batch_size,
                                                      kw::tol = tol,
                                                      kw::high_accuracy = high_accuracy,
                                                      kw::compact_mode = compact_mode};
        } else {
            // Times provided.
            std::vector<double> time_v;
            for (const auto &x : py::cast<py::iterable>(time)) {
                time_v.push_back(py::cast<double>(x));
            }

            return hey::taylor_adaptive_batch<double>{std::move(sys),
                                                      std::move(state),
                                                      batch_size,
                                                      kw::time = std::move(time_v),
                                                      kw::tol = tol,
                                                      kw::high_accuracy = high_accuracy,
                                                      kw::compact_mode = compact_mode};
        }
    };
    py::class_<hey::taylor_adaptive_batch<double>>(m, "taylor_adaptive_batch_double")
        .def(py::init([tabd_ctor_impl](std::vector<std::pair<hey::expression, hey::expression>> sys,
                                       std::vector<double> state, std::uint32_t batch_size, py::object time, double tol,
                                       bool high_accuracy, bool compact_mode) {
                 return tabd_ctor_impl(std::move(sys), std::move(state), batch_size, time, tol, high_accuracy,
                                       compact_mode);
             }),
             "sys"_a, "state"_a, "batch_size"_a, "time"_a = py::none{}, "tol"_a = 0., "high_accuracy"_a = false,
             "compact_mode"_a = false)
        .def(py::init([tabd_ctor_impl](std::vector<hey::expression> sys, std::vector<double> state,
                                       std::uint32_t batch_size, py::object time, double tol, bool high_accuracy,
                                       bool compact_mode) {
                 return tabd_ctor_impl(std::move(sys), std::move(state), batch_size, time, tol, high_accuracy,
                                       compact_mode);
             }),
             "sys"_a, "state"_a, "batch_size"_a, "time"_a = py::none{}, "tol"_a = 0., "high_accuracy"_a = false,
             "compact_mode"_a = false)
        .def("get_decomposition", &hey::taylor_adaptive_batch<double>::get_decomposition)
        .def("step", [](hey::taylor_adaptive_batch<double> &ta) { return ta.step(); })
        .def(
            "step",
            [](hey::taylor_adaptive_batch<double> &ta, const std::vector<double> &max_delta_t) {
                return ta.step(max_delta_t);
            },
            "max_delta_t"_a)
        .def("step_backward", [](hey::taylor_adaptive_batch<double> &ta) { return ta.step_backward(); })
        .def(
            "propagate_for",
            [](hey::taylor_adaptive_batch<double> &ta, const std::vector<double> &delta_t, std::size_t max_steps) {
                py::gil_scoped_release release;
                return ta.propagate_for(delta_t, max_steps);
            },
            "delta_t"_a, "max_steps"_a = 0)
        .def(
            "propagate_until",
            [](hey::taylor_adaptive_batch<double> &ta, const std::vector<double> &t, std::size_t max_steps) {
                py::gil_scoped_release release;
                return ta.propagate_until(t, max_steps);
            },
            "t"_a, "max_steps"_a = 0)
        .def_property_readonly("time",
                               [](py::object &o) {
                                   auto *ta = py::cast<hey::taylor_adaptive_batch<double> *>(o);
                                   return py::array_t<double>({boost::numeric_cast<py::ssize_t>(ta->get_time().size())},
                                                              ta->get_time_data(), o);
                               })
        .def_property_readonly("state",
                               [](py::object &o) {
                                   auto *ta = py::cast<hey::taylor_adaptive_batch<double> *>(o);
                                   return py::array_t<double>(
                                       {boost::numeric_cast<py::ssize_t>(ta->get_state().size())}, ta->get_state_data(),
                                       o);
                               })
        .def_property_readonly("order", &hey::taylor_adaptive_batch<double>::get_order)
        .def_property_readonly("dim", &hey::taylor_adaptive_batch<double>::get_dim)
        .def_property_readonly("batch_size", &hey::taylor_adaptive_batch<double>::get_batch_size)
        // Repr.
        .def("__repr__",
             [](const hey::taylor_adaptive_batch<double> &ta) {
                 std::ostringstream oss;
                 oss << ta;
                 return oss.str();
             })
        // Copy/deepcopy.
        .def("__copy__",
             [](const hey::taylor_adaptive_batch<double> &ta) {
                 py::gil_scoped_release release;
                 return ta;
             })
        .def(
            "__deepcopy__",
            [](const hey::taylor_adaptive_batch<double> &ta, py::dict) {
                py::gil_scoped_release release;
                return ta;
            },
            "memo"_a);
}
