// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PY_NUMPY_MEMORY_HPP
#define HEYOKA_PY_NUMPY_MEMORY_HPP

#include <cassert>
#include <cstddef>
#include <mutex>
#include <new>
#include <optional>
#include <typeindex>
#include <typeinfo>
#include <utility>

#include "common_utils.hpp"

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wcast-align"

#endif

namespace heyoka_py
{

namespace detail
{

void numpy_custom_free(void *, void *, std::size_t) noexcept;

} // namespace detail

// Metadata that will be associated to memory buffers
// allocated and managed by NumPy to store the
// contents of ndarrays.
struct numpy_mem_metadata {
    // Total size in bytes of the memory buffer
    // associated to this metadata instance.
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes,misc-non-private-member-variables-in-classes)
    const std::size_t m_tot_size;

private:
    // The function to be used to destroy the array
    // elements when the array is garbage collected.
    using dtor_func_t = void (*)(unsigned char *) noexcept;

    // Mutex to synchronise access from multiple threads.
    std::mutex m_mut;
    // NOTE: keep these private so we are sure
    // that we always interact with them in a
    // thread-safe manner.
    bool *m_ct_flags = nullptr;
    std::size_t m_el_size = 0;
    dtor_func_t m_dtor_func = nullptr;
    std::optional<std::type_index> m_type;

    // NOTE: numpy_custom_free requires access to ct_ptr/el_size
    // without having to go through the mutex.
    friend void detail::numpy_custom_free(void *, void *, std::size_t) noexcept;

    bool *ensure_ct_flags_inited_impl(std::size_t, dtor_func_t, const std::type_index &) noexcept;

public:
    // The only meaningful ctor.
    explicit numpy_mem_metadata(std::size_t) noexcept;

    // Defaulted dtor.
    // NOTE: ct_flags will be cleaned up by numpy_custom_free(),
    // no need for special actions in the dtor.
    ~numpy_mem_metadata() = default;

    // Delete everything else.
    numpy_mem_metadata() = delete;
    numpy_mem_metadata(const numpy_mem_metadata &) = delete;
    numpy_mem_metadata(numpy_mem_metadata &&) noexcept = delete;
    numpy_mem_metadata &operator=(const numpy_mem_metadata &) = delete;
    numpy_mem_metadata &operator=(numpy_mem_metadata &&) noexcept = delete;

    template <typename T>
    bool *ensure_ct_flags_inited() noexcept
    {
        return ensure_ct_flags_inited_impl(
            sizeof(T), [](unsigned char *ptr) noexcept { std::launder(reinterpret_cast<T *>(ptr))->~T(); }, typeid(T));
    }
};

std::pair<unsigned char *, numpy_mem_metadata *> get_memory_metadata(const void *) noexcept;

void install_custom_numpy_mem_handler();

// This function accepts in input the memory address of an element
// in a NumPy array of T, and either:
//
// - returns ptr in case the memory address contains a properly constructed T, or
// - returns nullptr in case the memory address contains an uninitialised T.
//
// If ptr belongs to a memory area not allocated by NumPy, this function will
// assume that someone took care of constructing a T in ptr.
template <typename T>
const T *numpy_check_cted(const void *ptr) noexcept
{
    assert(ptr != nullptr);

    // Try to locate ptr in the memory map.
    const auto [base_ptr, meta_ptr] = get_memory_metadata(ptr);

    if (base_ptr == nullptr) {
        assert(meta_ptr == nullptr);

        // The memory area is not managed by NumPy, assume that a T
        // has been constructed in ptr by someone else.
        return std::launder(reinterpret_cast<const T *>(ptr));
    }

    assert(meta_ptr != nullptr);

    // The memory area is managed by NumPy.
    // Fetch the array of construction flags.
    const auto *ct_flags = meta_ptr->ensure_ct_flags_inited<T>();

    // Compute the position of ptr in the memory area.
    const auto bytes_idx = reinterpret_cast<const unsigned char *>(ptr) - base_ptr;
    assert(bytes_idx >= 0);

    // Compute the position in the array.
    auto idx = static_cast<std::size_t>(bytes_idx);
    assert(idx % sizeof(T) == 0u);
    idx /= sizeof(T);

    if (ct_flags[idx]) {
        // A constructed T exists, fetch it.
        return std::launder(reinterpret_cast<const T *>(ptr));
    } else {
        // No constructed T exists.
        return nullptr;
    }
}

// This function accepts in input the memory address of an element
// in a NumPy array of T. If a constructed T exists
// at the memory address, then ptr will be returned without taking
// further actions. Otherwise, a T will be default-constructed
// at the memory address, and ptr will then be returned.
//
// If ptr belongs to a memory area not allocated by NumPy, this function
// will assume that someone took care of constructing an T in ptr.
// If the default construction of a T throws, the error flag will be set
// and an empty optional will be returned.
template <typename T, typename... Args>
std::optional<T *> numpy_ensure_cted(void *ptr, Args &&...args) noexcept
{
    assert(ptr != nullptr);

    // Try to locate ptr in the memory map.
    const auto [base_ptr, meta_ptr] = get_memory_metadata(ptr);

    if (base_ptr == nullptr) {
        assert(meta_ptr == nullptr);

        // The memory area is not managed by NumPy, assume that a T
        // has been constructed in ptr by someone else.
        return std::launder(reinterpret_cast<T *>(ptr));
    }

    assert(meta_ptr != nullptr);

    // The memory area is managed by NumPy.
    // Fetch the array of construction flags.
    auto *ct_flags = meta_ptr->ensure_ct_flags_inited<T>();

    // Compute the position of ptr in the memory area.
    auto bytes_idx = reinterpret_cast<unsigned char *>(ptr) - base_ptr;
    assert(bytes_idx >= 0);

    // Compute the position of in the array.
    auto idx = static_cast<std::size_t>(bytes_idx);
    assert(idx % sizeof(T) == 0u);
    idx /= sizeof(T);

    if (!ct_flags[idx]) {
        // No T exists, construct it.
        const auto err = with_pybind11_eh([&]() { ::new (ptr) T(std::forward<Args>(args)...); });

        if (err) {
            return {};
        }

        // Signal that a new T was constructed.
        ct_flags[idx] = true;
    }

    return std::launder(reinterpret_cast<T *>(ptr));
}

} // namespace heyoka_py

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

#endif
