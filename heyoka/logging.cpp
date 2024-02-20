// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <pybind11/pybind11.h>

#include <heyoka/logging.hpp>

#include "logging.hpp"

namespace heyoka_py
{

namespace py = pybind11;

void expose_logging_setters(py::module_ &m)
{
    namespace hey = heyoka;

    m.def("set_logger_level_trace", &hey::set_logger_level_trace);

    m.def("set_logger_level_debug", &hey::set_logger_level_debug);

    m.def("set_logger_level_info", &hey::set_logger_level_info);

    m.def("set_logger_level_warning", &hey::set_logger_level_warn);

    m.def("set_logger_level_error", &hey::set_logger_level_err);

    m.def("set_logger_level_critical", &hey::set_logger_level_critical);
}

} // namespace heyoka_py
