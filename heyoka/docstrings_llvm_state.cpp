// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <string>

#include "docstrings.hpp"

namespace heyoka_py::docstrings
{

std::string llvm_state()
{
    return R"(LLVM state class.

This class encapsulates a single LLVM IR module together with the options used to compile it.

Static methods are provided to interact with both the in-memory and on-disk compilation caches, which are used by both
:py:class:`~heyoka.llvm_state` and :py:class:`~heyoka.llvm_multi_state`. All static methods are thread-safe,
and the static methods interacting with the on-disk cache can be invoked concurrently from multiple processes.

A :ref:`tutorial <tut_jit_caching>` illustrating the use the caches is available.

)";
}

std::string llvm_state_ir()
{
    return R"(The intermediate representation (IR) of the module.

:type: str

)";
}

std::string llvm_state_bc()
{
    return R"(The bitcode of the module.

:type: bytes

)";
}

std::string llvm_state_object_code()
{
    return R"(The object code of the module.

:type: bytes

)";
}

std::string llvm_state_opt_level()
{
    return R"(The optimisation level employed during compilation.

The returned value is in the [0, 3] range.

:type: int

)";
}

std::string llvm_state_fast_math()
{
    return R"(The fast math setting employed during compilation.

This flags indicates if optimisations which may improve floating-point performance at the expense of
accuracy and/or strict conformance to the IEEE 754 standard were employed during compilation.

:type: bool

)";
}

std::string llvm_state_force_avx512()
{
    return R"(Flag indicating whether the use of AVX-512 registers was forced during compilation.

Currently heyoka.py's default is to *disable* the use of `AVX-512 <https://en.wikipedia.org/wiki/AVX-512>`__
registers on all Intel processors and to *enable* it on AMD Zen 4 and later processors. This flag
indicates whether the default heuristic was overridden, forcing the use of AVX-512 registers.
On processors without AVX-512 instructions, this flag has no effect.

:type: bool

)";
}

std::string llvm_state_slp_vectorize()
{
    return R"(Flag indicating whether the LLVM `SLP vectorizer <https://llvm.org/docs/Vectorizers.html#the-slp-vectorizer>`__ was enabled during compilation.

The SLP vectorizer can improve performance in some situations, but it results in longer compilation times.

:type: bool

)";
}

std::string llvm_state_code_model()
{
    return R"(The code model used during compilation.

:type: code_model

)";
}

std::string llvm_state_get_memcache_size()
{
    return R"(get_memcache_size() -> int

Get the current size (in bytes) of the in-memory cache.

:rtype: int

)";
}

std::string llvm_state_get_memcache_limit()
{
    return R"(get_memcache_limit() -> int

Get the size limit (in bytes) of the in-memory cache.

When the cache size exceeds this limit, the least recently used entries are evicted.
A value of 0 disables the cache.

:rtype: int

)";
}

std::string llvm_state_set_memcache_limit()
{
    return R"(set_memcache_limit(limit: int) -> None

Set the size limit (in bytes) of the in-memory cache.

When the cache size exceeds *limit*, the least recently used entries are evicted.
A value of 0 disables the cache.

:param limit: the new size limit.

:raises ValueError: if *limit* is negative.

)";
}

std::string llvm_state_clear_memcache()
{
    return R"(clear_memcache() -> None

Clear the in-memory cache.

All entries in the in-memory cache are removed.

)";
}

std::string llvm_state_get_diskcache_path()
{
    return R"(get_diskcache_path() -> pathlib.Path

Get the path to the on-disk cache directory.

:rtype: pathlib.Path

)";
}

std::string llvm_state_set_diskcache_path()
{
    return R"(set_diskcache_path(path: os.PathLike | str | bytes) -> None

Set the path to the on-disk cache directory.

:param path: the new cache path.

)";
}

std::string llvm_state_get_diskcache_enabled()
{
    return R"(get_diskcache_enabled() -> bool

Get the flag indicating whether the on-disk cache is enabled.

:rtype: bool

)";
}

std::string llvm_state_set_diskcache_enabled()
{
    return R"(set_diskcache_enabled(flag: bool) -> None

Enable or disable the on-disk cache.

:param flag: ``True`` to enable, ``False`` to disable.

)";
}

std::string llvm_state_get_diskcache_limit()
{
    return R"(get_diskcache_limit() -> int

Get the size limit (in bytes) of the on-disk cache.

When the cache size exceeds *limit*, the least recently used entries are evicted.
A value of 0 disables the cache.

:rtype: int

)";
}

std::string llvm_state_set_diskcache_limit()
{
    return R"(set_diskcache_limit(limit: int) -> None

Set the size limit (in bytes) of the on-disk cache.

When the cache size exceeds *limit*, the least recently used entries are evicted.
A value of 0 disables the cache.

:param limit: the new size limit.

:raises ValueError: if *limit* is negative.

)";
}

std::string llvm_state_get_diskcache_size()
{
    return R"(get_diskcache_size() -> int

Get the current size (in bytes) of the on-disk cache.

:rtype: int

)";
}

std::string llvm_state_clear_diskcache()
{
    return R"(clear_diskcache() -> None

Clear the on-disk cache.

All entries in the on-disk cache are removed.

)";
}

std::string llvm_multi_state()
{
    return R"(LLVM multi state class.

This class encapsulates a set of LLVM IR modules together with the options used to compile them.
All modules in an :py:class:`~heyoka.llvm_multi_state` are compiled with the same set of options.

The compilation caches are managed via the static methods of :py:class:`~heyoka.llvm_state`.

)";
}

std::string llvm_multi_state_ir()
{
    return R"(The intermediate representations (IR) of the modules.

:type: list[str]

)";
}

std::string llvm_multi_state_bc()
{
    return R"(The bitcode of the modules.

:type: list[bytes]

)";
}

std::string llvm_multi_state_object_code()
{
    return R"(The object code of the modules.

:type: list[bytes]

)";
}

std::string llvm_multi_state_parjit()
{
    return R"(Flag indicating whether parallel JIT compilation was enabled.

:type: bool

)";
}

} // namespace heyoka_py::docstrings
