// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <tuple>
#include <typeindex>
#include <utility>

#include <pybind11/pybind11.h>

#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#define PY_ARRAY_UNIQUE_SYMBOL heyoka_py_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL heyoka_py_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPY_TARGET_VERSION NPY_1_22_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>

#include <heyoka/detail/safe_integer.hpp>

#include "common_utils.hpp"
#include "numpy_memory.hpp"

// NOTE: ideas to improve performance, if ever needed:
// - reduce the locking, which currently is probably excessive
//   (but easy to check for safety, so not sure the complexity
//   is worth it?);
// - don't allocate separately buffers and ct_flags: do a single
//   allocation, use chars instead of bools for ct_flags, and
//   place them at the end of the buffer.

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
bool *numpy_mem_metadata::ensure_ct_flags_inited_impl(std::size_t sz, dtor_func_t dtor_func, move_func_t move_func,
                                                      const std::type_index &tp) noexcept
{
    assert(sz > 0u);
    assert(m_tot_size > 0u);
    assert(m_tot_size % sz == 0u);

    std::lock_guard lock(m_mut);

    if (m_ct_flags == nullptr) {
        assert(m_el_size == 0u);
        assert(m_dtor_func == nullptr);
        assert(m_move_func == nullptr);
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
        m_move_func = move_func;
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
std::pair<unsigned char *, numpy_mem_metadata *> get_memory_metadata(const void *ptr) noexcept
{
    return detail::with_locked_memory_map([&](auto &) -> std::pair<unsigned char *, numpy_mem_metadata *> {
        // Try to locate ptr in the memory map.
        const auto *cptr = reinterpret_cast<const unsigned char *>(ptr);
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
        // NOTE: in case of exceptions, the noexcept nature of ths
        // function will lead to program termination. This is ok,
        // as recovering from errors here seems quite hard.
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
        // NOTE: in case of exceptions, the noexcept nature of ths
        // function will lead to program termination. This is ok,
        // as recovering from errors here seems quite hard.
        with_locked_memory_map([&](auto &) {
            [[maybe_unused]] const auto iret = memory_map.emplace(std::piecewise_construct, cret_tuple, tot_size_tuple);
            assert(iret.second);
        });
    }

    return ret;
}

} // namespace

// This function will take as input an iterator to an element in
// the memory map and it will:
// - destroy all C++ objects constructed in the memory buffer
//   associated to the iterator,
// - remove the entry associated to it from the memory map,
// - call free() on the memory buffer.
// NOTE: the memory map needs to be locked while calling this function.
template <typename It>
void numpy_custom_free_impl(It it) noexcept
{
    auto *const cptr = it->first;

    // NOTE: no need to lock to access m_ct_flags/m_el_size while freeing
    // the memory area.
    if (it->second.m_ct_flags != nullptr) {
        assert(it->second.m_el_size != 0u);
        assert(it->second.m_tot_size % it->second.m_el_size == 0u);
        assert(it->second.m_dtor_func != nullptr);
        assert(it->second.m_move_func != nullptr);
        assert(it->second.m_type);

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

    // Remove the element from the memory map.
    memory_map.erase(it);

    // Free the underlying buffer.
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory,hicpp-no-malloc,cppcoreguidelines-no-malloc)
    std::free(cptr);
}

namespace
{

void numpy_custom_free(void *, void *p, std::size_t sz) noexcept
{
    if (sz != 0u && p != nullptr) {
        with_locked_memory_map([&](auto &) {
            auto *const cptr = reinterpret_cast<unsigned char *>(p);

            auto it = memory_map.find(cptr);
            assert(it != memory_map.end());
            // NOTE: this assert mostly works, but it fires
            // when an array is resized to zero via realloc(): it seems like
            // NumPy avoids calling realloc() with zero (probably
            // due to the platform-dependent behaviour?), reallocs
            // to one element instead, and free() is eventually called with a
            // sz value of 1 (i.e., 1 byte), instead of whatever
            // the size of the dtype is. Not sure if this is a NumPy bug,
            // or if I misunderstood the purpose of the sz param here.
            // assert(it->second.m_tot_size == sz);

            numpy_custom_free_impl(it);
        });
    }
}

} // namespace

void *numpy_custom_realloc(void *ctx, void *ptr, std::size_t size) noexcept
{
    // Handle the special case ptr == null. This is supposed
    // to be equivalent to malloc().
    if (ptr == nullptr) {
        return numpy_custom_malloc(ctx, size);
    }

    // NOTE: we need to be able to count the bytes in the buffer
    // via std::ptrdiff_t, as we will be performing pointer
    // subtractions. Hence, check that we can represent the size
    // in bytes of the buffer via std::ptrdiff_t.
    try {
        using safe_ptrdiff_t = boost::safe_numerics::safe<std::ptrdiff_t>;
        (void)static_cast<safe_ptrdiff_t>(size);
    } catch (...) {
        return nullptr;
    }

    // Setup the return value. If not set
    // from within with_locked_memory_map(), this
    // will remain null and signal either an error or
    // the special case size == 0.
    void *retval = nullptr;

    with_locked_memory_map([&](auto &) {
        auto *const cptr = reinterpret_cast<unsigned char *>(ptr);

        auto it = memory_map.find(cptr);
        assert(it != memory_map.end());

        // NOTE: no need to lock to access m_ct_flags/m_el_size while reallocing.

        // Handle the special case size == 0: following the
        // behaviour on Linux, we will call free() and return
        // nullptr.
        if (size == 0u) {
            numpy_custom_free_impl(it);
            return;
        }

        // Fetch the original size.
        const auto orig_size = it->second.m_tot_size;

        // Allocate the new buffer. Wrap it into a unique_ptr for automatic cleanup.
        auto free_helper = [](void *p) {
            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory,hicpp-no-malloc,cppcoreguidelines-no-malloc)
            std::free(p);
        };
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory,hicpp-no-malloc,cppcoreguidelines-no-malloc)
        std::unique_ptr<void, decltype(free_helper)> new_ptr(std::malloc(size), free_helper);
        if (!new_ptr) {
            // Allocation failed, return nullptr.
            return;
        }
        // Formally construct the storage array.
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        auto *cret = ::new (new_ptr.get()) unsigned char[size];

        // Copy over the elements.
        std::unique_ptr<bool[]> new_ct_ptr;
        if (it->second.m_ct_flags == nullptr) {
            // This memory buffer does not contain constructed C++
            // objects, which means that either the NumPy array is storing
            // C objects, or that it is an empty array of C++ objects. To ensure
            // proper realloc behaviour for the former case, we need
            // to memcpy().
            std::memcpy(cret, cptr, std::min(size, orig_size));
        } else {
            // This memory buffer may contain constructed
            // C++ objects, we need to move them over.
            assert(it->second.m_el_size != 0u);
            assert(orig_size % it->second.m_el_size == 0u);
            assert(size % it->second.m_el_size == 0u);
            assert(it->second.m_dtor_func != nullptr);
            assert(it->second.m_move_func != nullptr);
            assert(it->second.m_type);

            // Try to allocate the new ct_flags array.
            try {
                new_ct_ptr = std::make_unique<bool[]>(size / it->second.m_el_size);
            } catch (...) {
                // Allocation failed, just return nullptr.
                return;
            }

            // Move over the existing elements that need
            // to be preserved.
            const std::size_t n_ex_elems = std::min(size, orig_size) / it->second.m_el_size;
            for (std::size_t i = 0; i < n_ex_elems; ++i) {
                if (it->second.m_ct_flags[i]) {
                    auto *src_ptr = cptr + i * it->second.m_el_size;
                    auto *dst_ptr = cret + i * it->second.m_el_size;
                    it->second.m_move_func(dst_ptr, src_ptr);
                    // Mark the destination as constructed.
                    new_ct_ptr[i] = true;
                }
            }
        }

        // Register the new buffer in the map.
        // NOTE: in case of exceptions, the noexcept nature of ths
        // function will lead to program termination. This is ok,
        // as recovering from errors here seems quite hard.
        const auto [new_it, ins_flag]
            = memory_map.emplace(std::piecewise_construct, std::make_tuple(cret), std::make_tuple(size));
        assert(ins_flag);

        // Assign the metadata to the new buffer, if needed.
        if (it->second.m_ct_flags != nullptr) {
            // NOTE: ensure we properly release new_ct_ptr here.
            new_it->second.m_ct_flags = new_ct_ptr.release();
            new_it->second.m_el_size = it->second.m_el_size;
            new_it->second.m_dtor_func = it->second.m_dtor_func;
            new_it->second.m_move_func = it->second.m_move_func;
            new_it->second.m_type = it->second.m_type;
        }

        // Free the existing buffer.
        numpy_custom_free_impl(it);

        // Release new_ptr and assign retval before exiting.
        // NOTE: new_ct_ptr has either been released above, or
        // it contains nullptr and thus its dtor has no effect.
        retval = new_ptr.release();
    });

    return retval;
}

namespace
{

// The NumPy custom memory handler.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyDataMem_Handler npy_custom_mem_handler
    = {"npy_custom_allocator",
       1,
       {nullptr, numpy_custom_malloc, numpy_custom_calloc, numpy_custom_realloc, numpy_custom_free}};

// When our custom NumPy memory handler is installed, the original
// memory handler is saved in this variable.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::optional<py::object> numpy_orig_mem_handler;

} // namespace

} // namespace detail

// Helper to setup the custom memory handling functions.
// NOTE: these functions are NOT thread safe, this needs
// to be highlighted in the docs.
void setup_custom_numpy_mem_handler(py::module_ &m)
{
    m.def("install_custom_numpy_mem_handler", []() {
        if (detail::numpy_orig_mem_handler) {
            // Don't do anything if we have already overridden
            // the memory management functions.
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

        // Store the original memory handler.
        // NOTE: we use reinterpret_steal because PyDataMem_SetHandler() returned a new reference.
        detail::numpy_orig_mem_handler.emplace(py::reinterpret_steal<py::object>(py::handle(old)));
    });

    m.def("remove_custom_numpy_mem_handler", []() {
        if (!detail::numpy_orig_mem_handler) {
            // Don't do anything if we have not overridden
            // the memory management functions yet.
            return;
        }

        // Try to restore the original memory handler.
        auto *tmp = PyDataMem_SetHandler(detail::numpy_orig_mem_handler->ptr());
        if (tmp == nullptr) {
            // NOTE: if PyDataMem_SetHandler() fails, it already sets the error flag.
            throw pybind11::error_already_set();
        }

        // The original memory handler was restored successfully. As cleanup actions we need to:
        // - decrease the refcount of tmp, the handler that we just replaced, and
        // - destroy the content of numpy_orig_mem_handler (which includes decreasing the refcount).
        Py_DECREF(tmp);
        detail::numpy_orig_mem_handler.reset();
    });

    // NOTE: as usual, ensure that Pythonic global variables
    // are cleaned up before shutting down the interpreter.
    auto atexit = py::module_::import("atexit");
    atexit.attr("register")(py::cpp_function([]() {
#if !defined(NDEBUG)
        std::cout << "Cleaning up the custom NumPy memory management data" << std::endl;
#endif
        detail::numpy_orig_mem_handler.reset();
    }));
}

} // namespace heyoka_py
