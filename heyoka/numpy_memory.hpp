// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#include <pybind11/pybind11.h>

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

void *numpy_custom_realloc(void *, void *, std::size_t) noexcept;
template <typename It>
void numpy_custom_free_impl(It) noexcept;

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
    // The function to be used to move the array
    // elements during realloc.
    using move_func_t = void (*)(unsigned char *, unsigned char *) noexcept;

    // Mutex to synchronise access from multiple threads.
    std::mutex m_mut;
    // NOTE: keep these private so we are sure
    // that we always interact with them in a
    // thread-safe manner.
    // NOTE: all these data members are always modified together.
    bool *m_ct_flags = nullptr;
    std::size_t m_el_size = 0;
    dtor_func_t m_dtor_func = nullptr;
    move_func_t m_move_func = nullptr;
    std::optional<std::type_index> m_type;

    // NOTE: numpy_custom_free/realloc require access to ct_ptr/el_size
    // without having to go through the mutex.
    friend void *detail::numpy_custom_realloc(void *, void *, std::size_t) noexcept;
    template <typename It>
    friend void detail::numpy_custom_free_impl(It) noexcept;

    bool *ensure_ct_flags_inited_impl(std::size_t, dtor_func_t, move_func_t, const std::type_index &) noexcept;

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
            sizeof(T), [](unsigned char *ptr) noexcept { std::launder(reinterpret_cast<T *>(ptr))->~T(); },
            [](unsigned char *dst, unsigned char *src) noexcept {
                ::new (dst) T(std::move(*std::launder(reinterpret_cast<T *>(src))));
            },
            typeid(T));
    }
};

std::pair<unsigned char *, numpy_mem_metadata *> get_memory_metadata(const void *) noexcept;

void setup_custom_numpy_mem_handler(pybind11::module_ &);

} // namespace heyoka_py

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

#endif
