// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PY_COMMON_UTILS_HPP
#define HEYOKA_PY_COMMON_UTILS_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#if defined(__GLIBCXX__)

#include <cxxabi.h>

#endif

#include <boost/numeric/conversion/cast.hpp>
#include <boost/pfr/core.hpp>
#include <boost/pfr/core_name.hpp>
#include <boost/pfr/tuple_size.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Python.h>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>

namespace heyoka_py
{

namespace py = pybind11;

py::object builtins();

py::object type(const py::handle &);

std::string str(const py::handle &);

[[noreturn]] void py_throw(PyObject *, const char *);

bool callable(const py::handle &);

// Helper to expose the llvm_state getter
// for a Taylor integrator.
template <typename T>
inline void expose_llvm_state_property_ta(py::class_<T> &c)
{
    c.def_property_readonly("llvm_state", [](const T &self) {
        const auto &st = self.get_llvm_state();

        using ret_t = std::variant<std::reference_wrapper<const heyoka::llvm_state>,
                                   std::reference_wrapper<const heyoka::llvm_multi_state>>;

        return std::visit([](const auto &v) -> ret_t { return std::cref(v); }, st);
    });
}

// Helper to expose the llvm_state getter
// for a generic object.
template <typename T>
inline void expose_llvm_state_property(py::class_<T> &c)
{
    c.def_property_readonly("llvm_state", &T::get_llvm_state);
}

// NOTE: these are wrappers for the implementation of
// copy/deepcopy semantics for exposed C++ classes.
// Doing a simple C++ copy and casting it to Python
// won't work because it ignores dynamic Python
// attributes that might have been set on the input
// object o. Thus, the strategy is to first make
// a C++ copy of the original object and then attach
// to it copies of the dynamic attributes that were
// added to the original object from Python.
template <typename T>
py::object copy_wrapper(py::object o)
{
    // Fetch a pointer to the C++ copy.
    auto *o_cpp = py::cast<const T *>(o);

    // Copy the C++ object and transform it into
    // a Python object.
    // NOTE: no room for GIL unlock here, due
    // to possible copy of Pythonic event callbacks.
    py::object ret = py::cast(T(*o_cpp));

    // Fetch the list of attributes from the original
    // object and turn it into a set.
    auto orig_dir = py::set(builtins().attr("dir")(o));

    // Fetch the list of attributes form the copy
    // and turn it into a set.
    auto new_dir = py::set(builtins().attr("dir")(ret));

    // Compute the difference.
    // NOTE: this will be the list of attributes that
    // are in o but not in its copy.
    auto set_diff = orig_dir.attr("difference")(new_dir);

    // Iterate over the difference and assign the
    // missing attributes.
    for (auto attr_name : set_diff) {
        py::setattr(ret, attr_name, o.attr(attr_name));
    }

    return ret;
}

template <typename T>
py::object deepcopy_wrapper(py::object o, py::dict memo)
{
    // Fetch a pointer to the C++ copy.
    auto *o_cpp = py::cast<const T *>(o);

    // Copy the C++ object and transform it into
    // a Python object.
    // NOTE: no room for GIL unlock here, due
    // to possible copy of Pythonic event callbacks.
    py::object ret = py::cast(T(*o_cpp));

    // Fetch the list of attributes from the original
    // object and turn it into a set.
    auto orig_dir = py::set(builtins().attr("dir")(o));

    // Fetch the list of attributes form the copy
    // and turn it into a set.
    auto new_dir = py::set(builtins().attr("dir")(ret));

    // Compute the difference.
    // NOTE: this will be the list of attributes that
    // are in o but not in its copy.
    auto set_diff = orig_dir.attr("difference")(new_dir);

    // Iterate over the difference and deep copy the
    // missing attributes.
    auto copy_func = py::module_::import("copy").attr("deepcopy");
    for (auto attr_name : set_diff) {
        py::setattr(ret, attr_name, copy_func(o.attr(attr_name), memo));
    }

    return ret;
}

// Helper to check if a list of arrays may share any memory with each other.
// Quadratic complexity.
bool may_share_memory(const py::array &, const py::array &);

template <typename... Args>
bool may_share_memory(const py::array &a, const py::array &b, const Args &...args)
{
    const std::array args_arr = {std::cref(a), std::cref(b), std::cref(args)...};
    const auto nargs = args_arr.size();

    for (std::size_t i = 0; i < nargs; ++i) {
        for (std::size_t j = i + 1u; j < nargs; ++j) {
            if (may_share_memory(args_arr[i].get(), args_arr[j].get())) {
                return true;
            }
        }
    }

    return false;
}

// Check that an inputs array does not overlap with the integrator's internal state during the evaluation of a Taylor
// map in a Taylor integrator. The evaluation is implemented internally via a cfunc, which requires disjoint memory
// areas.
template <typename TA>
void check_eval_taylor_map_overlap(const TA *ta, const py::array &inputs, const py::object &o, const int dt)
{
    // NOTE: the easiest thing is to re-use may_share_memory() which requires creating numpy views on the data. We
    // create 1D numpy views - the shape does not matter for the overlap check.
    const py::array state_arr(py::dtype(dt),
                              py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(ta->get_state().size())},
                              ta->get_state_data(), o);
    const py::array tstate_arr(py::dtype(dt),
                               py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(ta->get_tstate().size())},
                               ta->get_tstate().data(), o);

    if (may_share_memory(inputs, state_arr) || may_share_memory(inputs, tstate_arr)) [[unlikely]] {
        py_throw(PyExc_ValueError, "The inputs array provided for the evaluation of a Taylor map may overlap with the "
                                   "integrator's internal data: please make sure that the inputs array does not share "
                                   "memory with the state or tstate arrays of the integrator");
    }
}

// Helper to check if a numpy array is a NPY_ARRAY_CARRAY (i.e., C-style
// contiguous and with properly aligned storage). The flag signals whether
// the array must also be writeable or not.
bool is_npy_array_carray(const py::array &, bool = false);

namespace detail
{

bool with_pybind11_eh_impl();

} // namespace detail

// This function will invoke the function object f,
// wrapping its execution in the pybind11 C++ -> Python
// exception translation logic. If a C++ exception is
// thrown by the execution of f, the Python error flag is set
// and true is returned. Otherwise, false will be returned.
// The return value of f is ignored.
template <typename F>
bool with_pybind11_eh(const F &f)
{
    try {
        f();

        return false;
    } catch (py::error_already_set &e) {
        e.restore();
        return true;
#ifdef __GLIBCXX__
    } catch (abi::__forced_unwind &) {
        throw;
#endif
    } catch (...) {
        return detail::with_pybind11_eh_impl();
    }
}

// Functor to transform on-the-fly the content of a dtens
// from sparse format into dense format.
struct dtens_t_it {
    const heyoka::dtens *dt = nullptr;

    std::pair<heyoka::dtens::v_idx_t, heyoka::expression>
    operator()(const std::pair<heyoka::dtens::sv_idx_t, heyoka::expression> &) const;

    // Helper to implement the conversion from sparse to dense format.
    static heyoka::dtens::v_idx_t sparse_to_dense(const heyoka::dtens::sv_idx_t &, heyoka::dtens::v_idx_t::size_type);
};

py::array as_carray(const py::iterable &, int);

// Macros to avoid repetitions in commonly-used keyword arguments.
#define HEYOKA_PY_LLVM_STATE_ARGS                                                                                      \
    "opt_level"_a.noconvert() = 3, "force_avx512"_a.noconvert() = false, "slp_vectorize"_a.noconvert() = false,        \
    "fast_math"_a.noconvert() = false, "code_model"_a.noconvert() = heyoka::code_model::small,                         \
    "parjit"_a.noconvert() = heyoka::detail::default_parjit

#define HEYOKA_PY_CFUNC_ARGS(default_cm)                                                                               \
    "high_accuracy"_a.noconvert() = false, "compact_mode"_a.noconvert() = default_cm,                                  \
    "parallel_mode"_a.noconvert() = false, "batch_size"_a.noconvert() = 0, "prec"_a.noconvert() = 0

// Common helper for the implementation of the ctor for the EOP/SW data classes.
template <typename Data>
Data eop_sw_ctor(const char *, const std::optional<py::array> &, const std::optional<std::string> &,
                 const std::optional<std::string> &);

// Small helper to make a structured dtype from the POD-like C++ type T.
//
// This is similar in spirit to the PYBIND11_NUMPY_DTYPE macro, but:
//
// - it is not necessary to list the data members, these are inferred via Boost.PFR, and
// - the resulting dtype is built with align=true (whereas PYBIND11_NUMPY_DTYPE leaves the dtype unaligned).
//
// The second bit is important because it guarantees that, if we are given a numpy array which is flagged as C style,
// contiguous and properly aligned, then we can fetch T pointers from its underlying memory buffer and use them for
// read/write operations without running into UB due to misaligned loads/stores.
template <typename T>
auto make_aligned_dtype()
{
    using namespace py::literals;

    // Fetch the names of T's data members.
    const auto fields = boost::pfr::names_as_array<T>();

    // Construct the list of tuples from which the dtype will be inited. These are pairs containing the name of the
    // field and its dtype.
    py::list dlist;
    const auto add_fields = [&dlist, &fields]<std::size_t... I>(std::index_sequence<I...>) {
        // NOTE: cast to void in order to enforce the use of the builtin comma operator, which, through guaranteed
        // sequencing, ensures that the tuples are appended in the correct order.
        (..., static_cast<void>(
                  dlist.append(py::make_tuple(fields[I], py::dtype::of<boost::pfr::tuple_element_t<I, T>>()))));
    };
    add_fields(std::make_index_sequence<boost::pfr::tuple_size_v<T>>{});

    // NOTE: ensure proper alignment with align=true.
    auto ret = py::module_::import("numpy").attr("dtype")(dlist, "align"_a = true).cast<pybind11::dtype>();

    // NOTE: let's make sure that the size and alignment computed by numpy match the C++ values.
    assert(ret.itemsize() == boost::numeric_cast<py::ssize_t>(sizeof(T)));
    assert(ret.alignment() == boost::numeric_cast<py::ssize_t>(alignof(T)));

    return ret;
}

} // namespace heyoka_py

#endif
