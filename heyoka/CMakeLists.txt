# Configure the version file.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/_version.py.in" "${CMAKE_CURRENT_BINARY_DIR}/_version.py" @ONLY)

# The list of heyoka.py's Python files.
set(HEYOKA_PY_PYTHON_FILES __init__.py)

# Copy the python files in the current binary dir,
# so that we can import heyoka from the build dir.
# NOTE: importing from the build dir will work
# only on single-configuration generators.
foreach(HEYOKA_PY_PYTHON_FILE ${HEYOKA_PY_PYTHON_FILES})
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${HEYOKA_PY_PYTHON_FILE}"
        "${CMAKE_CURRENT_BINARY_DIR}/${HEYOKA_PY_PYTHON_FILE}" COPYONLY)
endforeach()

# Core module.
Python3_add_library(core MODULE WITH_SOABI
    core.cpp
    common_utils.cpp
)

target_link_libraries(core PRIVATE heyoka::heyoka Boost::boost)
target_include_directories(core SYSTEM PRIVATE "${pybind11_INCLUDE_DIR}")
target_compile_definitions(core PRIVATE "${pybind11_DEFINITIONS}")
target_compile_options(core PRIVATE
    "$<$<CONFIG:Debug>:${HEYOKA_PY_CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${HEYOKA_PY_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:RelWithDebInfo>:${HEYOKA_PY_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:MinSizeRel>:${HEYOKA_PY_CXX_FLAGS_RELEASE}>"
)
set_target_properties(core PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(core PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE)
if(NOT CMAKE_CXX_STANDARD)
    # The user did not provide the CMAKE_CXX_STANDARD variable,
    # go with the default (C++17).
    message(STATUS "Setting the C++ standard version to the default value (17).")
    set_property(TARGET core PROPERTY CXX_STANDARD 17)
else()
    message(STATUS "Using the manually-specified value for the C++ standard version (${CMAKE_CXX_STANDARD}).")
endif()
set_property(TARGET core PROPERTY CXX_STANDARD_REQUIRED YES)
set_property(TARGET core PROPERTY CXX_EXTENSIONS NO)

if (HEYOKA_PY_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _HEYOKA_PY_IPO_RESULT OUTPUT _HEYOKA_PY_IPO_OUTPUT)
    if (_HEYOKA_PY_IPO_RESULT)
        message(STATUS "IPO requested and supported, enabling.")
        set_property(TARGET core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "IPO requested, but it is not supported by the compiler:\n${_HEYOKA_PY_IPO_OUTPUT}")
    endif()
    unset(_HEYOKA_PY_IPO_RESULT)
    unset(_HEYOKA_PY_IPO_OUTPUT)
endif()

# Setup the installation path.
# set(HEYOKA_PY_INSTALL_PATH "${YACMA_PYTHON_MODULES_INSTALL_PATH}/heyoka")

# Install the core module.
# install(TARGETS core
#     RUNTIME DESTINATION ${HEYOKA_PY_INSTALL_PATH}
#     LIBRARY DESTINATION ${HEYOKA_PY_INSTALL_PATH}
# )

# Add the Python files.
# install(FILES ${HEYOKA_PY_PYTHON_FILES} "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
#     DESTINATION ${HEYOKA_PY_INSTALL_PATH})
