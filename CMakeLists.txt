# NOTE: this is the minimum version currently
# required by heyoka.
cmake_minimum_required(VERSION 3.18.0)

# Set default build type to "Release".
# NOTE: this should be done before the project command since the latter can set
# CMAKE_BUILD_TYPE itself (it does so for nmake).
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING
		"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
	FORCE)
endif()

project(heyoka.py VERSION 0.21.8 LANGUAGES CXX C)

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
# NOTE: put the minimum version in a variable
# so that we can re-use it below.
set(_HEYOKA_PY_MIN_HEYOKA_VERSION 0.21.0)
find_package(heyoka REQUIRED CONFIG)
if(${heyoka_VERSION} VERSION_LESS ${_HEYOKA_PY_MIN_HEYOKA_VERSION})
    message(FATAL_ERROR "The minimum heyoka version required by heyoka.py is ${_HEYOKA_PY_MIN_HEYOKA_VERSION}, but version ${heyoka_VERSION} was found instead.")
endif()
unset(_HEYOKA_PY_MIN_HEYOKA_VERSION)

# Python.

# Setup the list of Python components we need to locate.
# NOTE: NumPy is used for long double, real and real128 support.
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
if(${Python3_VERSION_MINOR} LESS 5)
    message(FATAL_ERROR "The minimum supported version of Python is 3.5, but version 3.${Python3_VERSION_MINOR} was found instead")
endif()
# NOTE: mppp::real support requires NumPy 1.22 at least.
if(heyoka_WITH_REAL AND ${Python3_NumPy_VERSION} LESS 1.22)
    message(FATAL_ERROR "The minimum version of NumPy required for supporting arbitrary-precision computations is 1.22, but version ${Python3_NumPy_VERSION} was found instead")
endif()
# Extract the major.minor version of NumPy for use in the wheel configuration.
string(REGEX MATCH "([0-9]+)\.([0-9]+)" _HEYOKA_PY_NPY_MAJOR_MINOR ${Python3_NumPy_VERSION})
message(STATUS "NumPy major.minor version: ${_HEYOKA_PY_NPY_MAJOR_MINOR}")

set(HEYOKA_PY_INSTALL_PATH "" CACHE STRING "heyoka module installation path")
mark_as_advanced(HEYOKA_PY_INSTALL_PATH)
unset(_HEYOKA_PY_PYTHON3_COMPONENTS)

# pybind11.
find_package(pybind11 REQUIRED CONFIG)
if(${pybind11_VERSION} VERSION_LESS "2.10")
    message(FATAL_ERROR "pybind11 >= 2.10 is required, but version ${pybind11_VERSION} was found instead.")
endif()

# Boost.
# NOTE: need 1.69 for safe numerics.
find_package(Boost 1.69 REQUIRED serialization)

# Mandatory dependency on TBB.
find_package(TBB REQUIRED CONFIG)

# NOTE: we make direct use of the mp++ headers
# in the heyoka.py source code if real/real128 support is enabled.
if(heyoka_WITH_REAL128 OR heyoka_WITH_REAL)
    find_package(mp++ REQUIRED CONFIG)

    # NOTE: not sure which minimum version we need here,
    # for the moment let's just grab the minimal requirement
    # from current heyoka.
    set(_HEYOKA_PY_MIN_MPPP_VERSION "0.27")
    if(${mp++_VERSION} VERSION_LESS ${_HEYOKA_PY_MIN_MPPP_VERSION})
        message(FATAL_ERROR "mp++ >= ${_HEYOKA_PY_MIN_MPPP_VERSION} is required.")
    endif()
    unset(_HEYOKA_PY_MIN_MPPP_VERSION)
endif()

# Add the module directory.
add_subdirectory(heyoka)

# Cleanup.
unset(_HEYOKA_PY_NPY_MAJOR_MINOR)
