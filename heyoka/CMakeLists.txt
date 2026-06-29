# Configure the version file.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/_version.py.in" "${CMAKE_CURRENT_BINARY_DIR}/_version.py" @ONLY)

# The list of heyoka.py's Python files.
set(HEYOKA_PY_PYTHON_FILES
    __init__.py
    _sympy_utils.py
    _generic_wrappers.py
    _s11n.py
    _ensemble_impl.py
    _real_reduce_factory.py
    _test_scalar_integrator.py
    _test_sympy.py
    _test_vsop2013.py
    _test_var_ode_sys.py
    _test_var_integrator.py
    _test_sgp4_propagator.py
    _test_sw_data.py
    model/__init__.py
    model/_sgp4_propagator.py
    callback/__init__.py
    test/__init__.py
    test/_utils.py
    test/_sgp4_test_data.py
    test/test_c_output.py
    test/test_events.py
    test/test_llvm_state.py
    test/test_package_layout.py
    test/test_recommended_simd_size.py
    test/test_s11n_backend.py
    test/test_batch_integrator.py
    test/test_celmec.py
    test/test_cfunc.py
    test/test_diskcache.py
    test/test_dtens.py
    test/test_elp2000.py
    test/test_ensemble.py
    test/test_eop_data.py
    test/test_expression.py
    test/test_lagham.py
    test/test_memcache.py
    test/test_model.py
    test/test_mp.py
    test/test_real128.py
    test/test_real.py
)

# Copy the python files in the current binary dir,
# so that we can import heyoka from the build dir.
# NOTE: importing from the build dir will work
# only on single-configuration generators.
foreach(HEYOKA_PY_PYTHON_FILE ${HEYOKA_PY_PYTHON_FILES})
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${HEYOKA_PY_PYTHON_FILE}"
        "${CMAKE_CURRENT_BINARY_DIR}/${HEYOKA_PY_PYTHON_FILE}" COPYONLY)
endforeach()

# Core module.
set(_HEYOKA_PY_CORE_SOURCES
    core.cpp
    taylor_expose_integrator.cpp
    taylor_expose_events.cpp
    taylor_expose_c_output.cpp
    common_utils.cpp
    logging.cpp
    setup_sympy.cpp
    cfunc.cpp
    expose_real128.cpp
    expose_real.cpp
    dtypes.cpp
    custom_casters.cpp
    expose_expression.cpp
    expose_var_ode_sys.cpp
    expose_batch_integrators.cpp
    numpy_memory.cpp
    expose_models.cpp
    expose_sgp4_propagators.cpp
    expose_callbacks.cpp
    expose_eop_data.cpp
    expose_sw_data.cpp
    step_cb_utils.cpp
    docstrings.cpp
    docstrings_llvm_state.cpp
    docstrings_models.cpp
)

Python3_add_library(_core MODULE WITH_SOABI ${_HEYOKA_PY_CORE_SOURCES})

unset(_HEYOKA_PY_CORE_SOURCES)

target_link_libraries(_core PRIVATE heyoka::heyoka fmt::fmt Boost::boost Boost::serialization TBB::tbb Python3::NumPy)
target_link_libraries(_core PRIVATE "${pybind11_LIBRARIES}")
if(heyoka_WITH_REAL128 OR heyoka_WITH_REAL)
    target_link_libraries(_core PRIVATE mp++::mp++)
endif()
target_include_directories(_core SYSTEM PRIVATE "${pybind11_INCLUDE_DIR}" "${Python3_INCLUDE_DIRS}")
target_compile_definitions(_core PRIVATE "${pybind11_DEFINITIONS}")
target_compile_options(_core PRIVATE
    "$<$<CONFIG:Debug>:${HEYOKA_PY_CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${HEYOKA_PY_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:RelWithDebInfo>:${HEYOKA_PY_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:MinSizeRel>:${HEYOKA_PY_CXX_FLAGS_RELEASE}>"
)
set_target_properties(_core PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(_core PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE)
target_compile_features(_core PRIVATE cxx_std_23)
set_property(TARGET _core PROPERTY CXX_EXTENSIONS NO)

if (HEYOKA_PY_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _HEYOKA_PY_IPO_RESULT OUTPUT _HEYOKA_PY_IPO_OUTPUT)
    if (_HEYOKA_PY_IPO_RESULT)
        message(STATUS "IPO requested and supported, enabling.")
        set_property(TARGET _core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
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
install(TARGETS _core
    RUNTIME DESTINATION ${_HEYOKA_PY_INSTALL_DIR}
    LIBRARY DESTINATION ${_HEYOKA_PY_INSTALL_DIR}
)

# Add the Python files.
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_version.py" DESTINATION ${_HEYOKA_PY_INSTALL_DIR})
foreach(_HEYOKA_PY_CUR_FILE ${HEYOKA_PY_PYTHON_FILES})
    get_filename_component(_HEYOKA_PY_CUR_DIR ${_HEYOKA_PY_CUR_FILE} DIRECTORY)
    install(FILES ${_HEYOKA_PY_CUR_FILE} DESTINATION "${_HEYOKA_PY_INSTALL_DIR}/${_HEYOKA_PY_CUR_DIR}")
endforeach()

unset(_HEYOKA_PY_INSTALL_DIR)
