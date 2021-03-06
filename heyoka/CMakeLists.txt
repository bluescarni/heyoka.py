# Configure the version file.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/_version.py.in" "${CMAKE_CURRENT_BINARY_DIR}/_version.py" @ONLY)

# The list of heyoka.py's Python files.
set(HEYOKA_PY_PYTHON_FILES __init__.py test.py _sympy_utils.py)

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
    taylor_add_jet.cpp
    taylor_expose_integrator.cpp
    taylor_expose_events.cpp
    logging.cpp
    setup_sympy.cpp
)

target_link_libraries(core PRIVATE heyoka::heyoka fmt::fmt spdlog::spdlog Boost::boost Boost::serialization)
if(heyoka_WITH_MPPP)
    target_link_libraries(core PRIVATE mp++::mp++)
endif()
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
target_compile_features(core PRIVATE cxx_std_17)
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

# Installation setup.
if(HEYOKA_PY_INSTALL_PATH STREQUAL "")
    message(STATUS "heyoka.py will be installed in the default location: ${Python3_SITEARCH}")
    set(_HEYOKA_PY_INSTALL_DIR "${Python3_SITEARCH}/heyoka")
else()
    message(STATUS "heyoka.py will be installed in the custom location: ${HEYOKA_PY_INSTALL_PATH}")
    set(_HEYOKA_PY_INSTALL_DIR "${HEYOKA_PY_INSTALL_PATH}/heyoka")
endif()

# Install the core module.
install(TARGETS core
    RUNTIME DESTINATION ${_HEYOKA_PY_INSTALL_DIR}
    LIBRARY DESTINATION ${_HEYOKA_PY_INSTALL_DIR}
)

# Add the Python files.
install(FILES ${HEYOKA_PY_PYTHON_FILES} "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
    DESTINATION ${_HEYOKA_PY_INSTALL_DIR})

unset(_HEYOKA_PY_INSTALL_DIR)
