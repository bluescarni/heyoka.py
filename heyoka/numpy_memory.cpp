// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <tuple>
#include <typeindex>
#include <utility>

#include <boost/safe_numerics/safe_integer.hpp>

#include <pybind11/pybind11.h>

#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#define PY_ARRAY_UNIQUE_SYMBOL heyoka_py_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL heyoka_py_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>

#include "common_utils.hpp"
#include "numpy_memory.hpp"

namespace heyoka_py
{

numpy_mem_metadata::numpy_mem_metadata(std::size_t size) noexcept : m_tot_size(size)
{
    // NOTE: metadata not needed/supported for empty buffers.
    assert(size > 0u);
}

// This function will ensure that this contains an array
// of construction flags m_ct_flags for elements of size sz.
// If it does not, it will create a new array of m_tot_size / sz
// flags all inited to false. This function can be
// invoked concurrently from multiple threads.
// dtor_func is a function that will be invoked to destroy
// the elements allocated in the memory buffer when it is deallocated.
// tp is the type of the elements stored in the memory buffer.
bool *numpy_mem_metadata::ensure_ct_flags_inited_impl(std::size_t sz, dtor_func_t dtor_func,
                                                      const std::type_index &tp) noexcept
{
    assert(sz > 0u);
    assert(m_tot_size > 0u);
    assert(m_tot_size % sz == 0u);

    std::lock_guard lock(m_mut);

    if (m_ct_flags == nullptr) {
        assert(m_el_size == 0u);
        assert(m_dtor_func == nullptr);
        assert(!m_type);

        // Init a new array of flags.
        // NOTE: this will init all flags to false.
        // NOTE: this could in principle throw, in which case the application will
        // exit - this seems fine as it signals an out of memory condition.
        auto new_ct_flags = std::make_unique<bool[]>(m_tot_size / sz);

        // Assign the new array of flags.
        m_ct_flags = new_ct_flags.release();

        // Assign the element size, the dtor and the type.
        m_el_size = sz;
        m_dtor_func = dtor_func;
        m_type.emplace(tp);
    }

    // NOTE: not sure if we can assert m_dtor_func == dtor_func
    // here, since dtor_func is ultimately produced from a lambda
    // wrapped in an inline function and perhaps there's a chance
    // different translation units end up with different function
    // pointers for the same lambda...
    assert(m_el_size == sz);

    // NOTE: should the checks on the type be turned into proper
    // exceptions, rather than assertions?
    assert(m_type);
    assert(*m_type == tp);

    return m_ct_flags;
}

namespace detail
{

namespace
{

// Global dictionary that maps NumPy memory buffers
// to metadata. Need std::greater (rather than std::less)
// as comparator because of the way we use lower_bound()
// (see below).
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables
std::map<unsigned char *, numpy_mem_metadata, std::greater<>> memory_map;

// Mutex to synchronise access to memory_map.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables
std::mutex memory_map_mutex;

// Small helper to perform an action on memory_map while
// keeping memory_map_mutex locked.
template <typename F>
auto with_locked_memory_map(const F &f) noexcept
{
    std::unique_lock<std::mutex> lock(memory_map_mutex);

    return f(lock);
}

} // namespace

} // namespace detail

// This function will try to locate the memory area (in memory_map)
// which ptr belongs to. If successful, it will return a pair containing:
//
// - the starting address of the memory area,
// - the metadata of the memory area.
//
// Otherwise, it means that ptr belongs to a memory area not managed by NumPy, and
// {nullptr, nullptr} will be returned.
std::pair<unsigned char *, numpy_mem_metadata *> get_memory_metadata(void *ptr) noexcept
{
    return detail::with_locked_memory_map([&](auto &) -> std::pair<unsigned char *, numpy_mem_metadata *> {
        // Try to locate ptr in the memory map.
        auto *const cptr = reinterpret_cast<unsigned char *>(ptr);
        // NOTE: lower_bound() here finds the first element in memory_map
        // which is less than or equal to cptr (thanks to the fact that we are
        // using std::greater as comparator, rather than the default std::less).
        auto it = detail::memory_map.lower_bound(cptr);

        if (it == detail::memory_map.end() || !std::less{}(cptr, it->first + it->second.m_tot_size)) {
            // ptr does not belong to any memory area managed by NumPy.
            return {nullptr, nullptr};
        } else {
            return {it->first, &it->second};
        }
    });
}

namespace detail
{

namespace
{

// Custom malloc that registers the memory buffer
// in the memory map.
void *numpy_custom_malloc(void *, std::size_t sz) noexcept
{
    // NOTE: we need to be able to count the bytes in the buffer
    // via std::ptrdiff_t, as we will be performing pointer
    // subtractions. Hence, check that we can represent the size
    // in bytes of the buffer via std::ptrdiff_t.
    try {
        using safe_ptrdiff_t = boost::safe_numerics::safe<std::ptrdiff_t>;
        (void)static_cast<safe_ptrdiff_t>(sz);
    } catch (...) {
        return nullptr;
    }

    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory,hicpp-no-malloc,cppcoreguidelines-no-malloc)
    auto *ret = std::malloc(sz);

    if (sz != 0u && ret != nullptr) {
        // Formally construct the storage array.
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        auto *cret = ::new (ret) unsigned char[sz];

        // Make tuples for use with std::piecewise_construct.
        auto cret_tuple = std::make_tuple(cret);
        auto sz_tuple = std::make_tuple(sz);

        // Register the memory area in the map.
        with_locked_memory_map([&](auto &) {
            [[maybe_unused]] const auto iret = memory_map.emplace(std::piecewise_construct, cret_tuple, sz_tuple);
            assert(iret.second);
        });
    }

    return ret;
}

// Custom calloc that registers the memory buffer
// in the memory map.
void *numpy_custom_calloc(void *, std::size_t nelem, std::size_t elsize) noexcept
{
    std::size_t tot_size = 0;

    // Overflow check on the total allocated size.
    try {
        using safe_size_t = boost::safe_numerics::safe<std::size_t>;
        using safe_ptrdiff_t = boost::safe_numerics::safe<std::ptrdiff_t>;

        tot_size = nelem * safe_size_t(elsize);
        // NOTE: need overflow check also wrt std::ptrdiff_t.
        (void)static_cast<safe_ptrdiff_t>(tot_size);
    } catch (...) {
        return nullptr;
    }

    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory,hicpp-no-malloc,cppcoreguidelines-no-malloc)
    auto *ret = std::malloc(tot_size);

    if (tot_size != 0u && ret != nullptr) {
        // Formally construct the storage array.
        // NOTE: value-init to zero-init.
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        auto *cret = ::new (ret) unsigned char[tot_size]{};

        // Make tuples for use with std::piecewise_construct.
        auto cret_tuple = std::make_tuple(cret);
        auto tot_size_tuple = std::make_tuple(tot_size);

        // Register the memory area in the map.
        with_locked_memory_map([&](auto &) {
            [[maybe_unused]] const auto iret = memory_map.emplace(std::piecewise_construct, cret_tuple, tot_size_tuple);
            assert(iret.second);
        });
    }

    return ret;
}

void *numpy_custom_realloc(void *, void *, std::size_t) noexcept
{
    // FIX.
    // TODO: can we take a speedy path in case no ct flag array
    // exists here? Could we invoke realloc directly? But then is
    // this consistent with the in-place construction of a char array
    // we do in malloc/calloc? Need also to see what the NumPy requirements
    // exactly are on this function.
    // TODO remember overflow checks here as well.
    std::exit(1);
}

} // namespace

void numpy_custom_free(void *, void *p, std::size_t sz) noexcept
{
    if (sz != 0u && p != nullptr) {
        with_locked_memory_map([&](auto &) {
            auto *const cptr = reinterpret_cast<unsigned char *>(p);

            auto it = memory_map.find(cptr);
            assert(it != memory_map.end());

            // NOTE: no need to lock to access m_ct_flags/m_el_size while freeing
            // the memory area.
            if (it->second.m_ct_flags != nullptr) {
                assert(it->second.m_el_size != 0u);
                assert(it->second.m_tot_size != 0u);
                assert(it->second.m_tot_size % it->second.m_el_size == 0u);
                assert(it->second.m_dtor_func != nullptr);

                const std::size_t n_elems = it->second.m_tot_size / it->second.m_el_size;
                for (std::size_t i = 0; i < n_elems; ++i) {
                    if (it->second.m_ct_flags[i]) {
                        auto *cur_ptr = cptr + i * it->second.m_el_size;
                        it->second.m_dtor_func(cur_ptr);
                    }
                }

                // Delete ct_ptr.
                std::unique_ptr<bool[]> ct_ptr(it->second.m_ct_flags);
            }

            memory_map.erase(it);
        });
    }

    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory,hicpp-no-malloc,cppcoreguidelines-no-malloc)
    std::free(p);
}

namespace
{

// The NumPy custom memory handler.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyDataMem_Handler npy_custom_mem_handler
    = {"npy_custom_allocator",
       1,
       {nullptr, numpy_custom_malloc, numpy_custom_calloc, numpy_custom_realloc, numpy_custom_free}};

// Flag to signal if the default NumPy memory handler
// has been overriden by npy_custom_mem_handler.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool numpy_mh_overridden = false;

} // namespace

} // namespace detail

// Helper to install the custom memory handling functions.
// This can be called multiple times, all invocations past the
// first one are no-ops.
void install_custom_numpy_mem_handler()
{
    if (detail::numpy_mh_overridden) {
        // Don't do anything if we have overridden
        // the memory management functions already.
        return;
    }

    // NOTE: in principle here we could fetch the original memory handling
    // capsule (which is also called "mem_handler"), and re-use the original
    // memory functions in our implementations, instead of calling malloc/calloc/etc.
    // This would make our custom implementations "good citizens", in the sense
    // that we would respect existing custom memory allocating routines instead of
    // outright overriding and ignoring them. Probably this is not an immediate concern
    // as the memory management API is rather new, but it is something we should
    // keep in mind moving forward.
    auto *new_mem_handler = PyCapsule_New(&detail::npy_custom_mem_handler, "mem_handler", nullptr);
    if (new_mem_handler == nullptr) {
        // NOTE: if PyCapsule_New() fails, it already sets the error flag.
        throw pybind11::error_already_set();
    }

    auto *old = PyDataMem_SetHandler(new_mem_handler);
    Py_DECREF(new_mem_handler);
    if (old == nullptr) {
        // NOTE: if PyDataMem_SetHandler() fails, it already sets the error flag.
        throw pybind11::error_already_set();
    }

    Py_DECREF(old);

    // Set the flag.
    detail::numpy_mh_overridden = true;
}

} // namespace heyoka_py
