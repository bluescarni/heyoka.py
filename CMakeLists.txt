# NOTE: this is the minimum version currently
# required by heyoka.
cmake_minimum_required(VERSION 3.18.0)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.31.0)
    # NOTE: adopt the new behaviour for this policy:
    # https://cmake.org/cmake/help/latest/policy/CMP0177.html
    cmake_policy(SET CMP0177 NEW)
endif()

# Set default build type to "Release".
# NOTE: this should be done before the project command since the latter can set
# CMAKE_BUILD_TYPE itself (it does so for nmake).
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING
		"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
	FORCE)
endif()

project(heyoka.py VERSION 7.0.1 LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/yacma")

message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")
message(STATUS "heyoka.py version: ${heyoka.py_VERSION}")

option(HEYOKA_PY_ENABLE_IPO "Enable IPO (requires compiler support)." OFF)
mark_as_advanced(HEYOKA_PY_ENABLE_IPO)

# Run the YACMA compiler setup.
include(YACMACompilerLinkerSettings)

# Assemble the flags.
set(HEYOKA_PY_CXX_FLAGS_DEBUG ${YACMA_CXX_FLAGS} ${YACMA_CXX_FLAGS_DEBUG})
set(HEYOKA_PY_CXX_FLAGS_RELEASE ${YACMA_CXX_FLAGS})

if(YACMA_COMPILER_IS_MSVC)
    include(CheckCXXCompilerFlag)
    # Disable the idiotic minmax macros on MSVC (both cl and clang-cl).
    # Also, enable the bigobj flag and the WIN32_LEAN_AND_MEAN definitions:
    # https://stackoverflow.com/questions/11040133/what-does-defining-win32-lean-and-mean-exclude-exactly
    list(APPEND HEYOKA_PY_CXX_FLAGS_DEBUG "-DNOMINMAX" "/bigobj" "-DWIN32_LEAN_AND_MEAN")
    list(APPEND HEYOKA_PY_CXX_FLAGS_RELEASE "-DNOMINMAX" "/bigobj" "-DWIN32_LEAN_AND_MEAN")
    # Enable strict conformance mode, if supported.
    set(CMAKE_REQUIRED_QUIET TRUE)
    check_cxx_compiler_flag("/permissive-" _HEYOKA_PY_MSVC_SUPPORTS_STRICT_CONFORMANCE)
    unset(CMAKE_REQUIRED_QUIET)
    if(_HEYOKA_PY_MSVC_SUPPORTS_STRICT_CONFORMANCE)
        message(STATUS "The '/permissive-' flag is supported, enabling it.")
        list(APPEND HEYOKA_PY_CXX_FLAGS_DEBUG "/permissive-")
        list(APPEND HEYOKA_PY_CXX_FLAGS_RELEASE "/permissive-")
    endif()
    unset(_HEYOKA_PY_MSVC_SUPPORTS_STRICT_CONFORMANCE)
    if(YACMA_COMPILER_IS_CLANGXX)
        # clang-cl emits various warnings from GMP/MPFR, let's just silence them.
        # NOTE: at one point in the recent past, MSVC added an options similar to GCC's isystem:
        # https://blogs.msdn.microsoft.com/vcblog/2017/12/13/broken-warnings-theory/
        # We probably just need to wait for this to be picked up by CMake/clang-cl. Let's
        # revisit the issue in the future.
        list(APPEND _HEYOKA_PY_CLANG_CL_DISABLED_WARNINGS
            "-Wno-unused-variable"
            "-Wno-inconsistent-dllimport"
            "-Wno-unknown-pragmas"
            "-Wno-unused-parameter"
            "-Wno-sign-compare"
            "-Wno-deprecated-declarations"
            "-Wno-deprecated-dynamic-exception-spec"
            "-Wno-old-style-cast"
            "-Wno-sign-conversion"
            "-Wno-non-virtual-dtor"
            "-Wno-deprecated"
            "-Wno-shadow"
            "-Wno-shorten-64-to-32"
            "-Wno-reserved-id-macro"
            "-Wno-undef"
            "-Wno-c++98-compat-pedantic"
            "-Wno-documentation-unknown-command"
            "-Wno-zero-as-null-pointer-constant"
            "-Wno-language-extension-token"
            "-Wno-gnu-anonymous-struct"
            "-Wno-nested-anon-types"
            "-Wno-documentation"
            "-Wno-comma"
            "-Wno-nonportable-system-include-path"
            "-Wno-global-constructors"
            "-Wno-redundant-parens"
            "-Wno-exit-time-destructors"
            "-Wno-missing-noreturn"
            "-Wno-switch-enum"
            "-Wno-covered-switch-default"
            "-Wno-float-equal"
            "-Wno-double-promotion"
            "-Wno-microsoft-enum-value"
            "-Wno-missing-prototypes"
            "-Wno-implicit-fallthrough"
            "-Wno-format-nonliteral"
            "-Wno-cast-qual"
            "-Wno-disabled-macro-expansion"
            "-Wno-unused-private-field"
            "-Wno-unused-template"
            "-Wno-unused-macros"
            "-Wno-extra-semi-stmt"
            "-Wno-c++98-compat")
        list(APPEND HEYOKA_PY_CXX_FLAGS_DEBUG ${_HEYOKA_PY_CLANG_CL_DISABLED_WARNINGS})
        list(APPEND HEYOKA_PY_CXX_FLAGS_RELEASE ${_HEYOKA_PY_CLANG_CL_DISABLED_WARNINGS})
        unset(_HEYOKA_PY_CLANG_CL_DISABLED_WARNINGS)
    else()
        # Problematic MSVC cl warnings.
        list(APPEND HEYOKA_PY_CXX_FLAGS_DEBUG "/wd4459")
        list(APPEND HEYOKA_PY_CXX_FLAGS_RELEASE "/wd4459")
    endif()
endif()
if(MINGW)
    # In MinGW some tests generate big object files.
    message(STATUS "Enabling the '-Wa,-mbig-obj' flag for MinGW.")
    list(APPEND HEYOKA_PY_CXX_FLAGS_DEBUG "-Wa,-mbig-obj")
    list(APPEND HEYOKA_PY_CXX_FLAGS_RELEASE "-Wa,-mbig-obj")
endif()

# Find the dependencies.

# fmt.
find_package(fmt REQUIRED CONFIG)
message(STATUS "fmt version: ${fmt_VERSION}")

# heyoka.
find_package(heyoka 7.0.0 REQUIRED CONFIG)

# Python.

# Setup the list of Python components we need to locate.
# NOTE: NumPy is used for float32, long double, real and real128 support.
set(_HEYOKA_PY_PYTHON3_COMPONENTS Interpreter NumPy Development.Module)

# NOTE: strictly speaking, we should need only the Interpreter and Development.Module
# components to build heyoka.py. However, in certain setups (e.g., skbuild), it looks
# like if we specify only Development.Module CMake is unable to correctly locate
# the Python installation. Hence, as a workaround, if the following line fails
# and Python3 is *not* found, we try again, this time with the full Development
# component (rather than only Development.Module). This seems to work around the
# issue, for now at least.
find_package(Python3 QUIET COMPONENTS ${_HEYOKA_PY_PYTHON3_COMPONENTS})

if(NOT Python3_FOUND)
    list(POP_BACK _HEYOKA_PY_PYTHON3_COMPONENTS)
    list(APPEND _HEYOKA_PY_PYTHON3_COMPONENTS Development)
endif()

find_package(Python3 QUIET REQUIRED COMPONENTS ${_HEYOKA_PY_PYTHON3_COMPONENTS})

message(STATUS "Python3 interpreter: ${Python3_EXECUTABLE}")
message(STATUS "Python3 installation directory: ${Python3_SITEARCH}")
# NOTE: Python 3.9 seems like the absolute minimum for NumPy 2.
if(${Python3_VERSION_MINOR} LESS 9)
    message(FATAL_ERROR "The minimum supported version of Python is 3.9, but version 3.${Python3_VERSION_MINOR} was found instead")
endif()
# We support only NumPy>=2.
if((heyoka_WITH_REAL OR heyoka_WITH_REAL128) AND ${Python3_NumPy_VERSION} VERSION_LESS 2)
    message(FATAL_ERROR "NumPy>=2 is required when building with support for quadruple-precision or arbitrary-precision computations")
endif()
message(STATUS "NumPy version: ${Python3_NumPy_VERSION}")
unset(_HEYOKA_PY_PYTHON3_COMPONENTS)

if (DEFINED SKBUILD)
    # If we're using scikit-build-core, set install path to the current
    # directory since skbuild will be handling the installation.
    set(HEYOKA_PY_INSTALL_PATH "." CACHE STRING "heyoka module installation path")

    # Also ensure shared libraries can be found in RPATH.
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
else()
    set(HEYOKA_PY_INSTALL_PATH "" CACHE STRING "heyoka module installation path")
endif()
mark_as_advanced(HEYOKA_PY_INSTALL_PATH)

# pybind11.
find_package(pybind11 REQUIRED CONFIG)
if(${pybind11_VERSION} VERSION_LESS "2.10")
    message(FATAL_ERROR "pybind11 >= 2.10 is required, but version ${pybind11_VERSION} was found instead.")
endif()

# Boost.
# NOTE: we look for Boost in CONFIG mode first, as that has become the official supported way
# of locating Boost in recent Boost/CMake versions. If we fail, we try again in
# MODULE mode as last resort.
# NOTE: need 1.69 for safe numerics.
find_package(Boost 1.69 QUIET COMPONENTS serialization CONFIG)
if(NOT ${Boost_FOUND})
    message(STATUS "Boost not found in CONFIG mode, retrying in MODULE mode.")
    find_package(Boost 1.69 QUIET MODULE COMPONENTS serialization)
endif()
if(NOT ${Boost_FOUND})
    message(FATAL_ERROR "Could not locate Boost in either CONFIG or MODULE mode.")
endif()
message(STATUS "Found Boost version ${Boost_VERSION}.")

# Mandatory dependency on TBB.
find_package(TBB REQUIRED CONFIG)

# NOTE: we make direct use of the mp++ headers
# in the heyoka.py source code if real/real128 support is enabled.
if(heyoka_WITH_REAL128 OR heyoka_WITH_REAL)
    # NOTE: sync up the required mp++ version
    # with heyoka for now, not sure if there is
    # a better solution.
    find_package(mp++ ${heyoka_mp++_VERSION} REQUIRED CONFIG)
endif()

# Add the module directory.
add_subdirectory(heyoka)
