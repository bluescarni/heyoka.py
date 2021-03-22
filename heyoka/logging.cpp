// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <memory>
#include <mutex>
#include <utility>

#include <fmt/core.h>

#include <spdlog/sinks/base_sink.h>
#include <spdlog/spdlog.h>

#include <pybind11/pybind11.h>

#include <heyoka/logging.hpp>

#include "logging.hpp"

namespace heyoka_py
{

namespace py = pybind11;
namespace hey = heyoka;

namespace detail
{

namespace
{

template <typename Mutex>
class py_sink : public spdlog::sinks::base_sink<Mutex>
{
protected:
    void sink_it_(const spdlog::details::log_msg &msg) override
    {
        // NOTE: grab the raw message and convert it to string
        // without applying any spdlog-specific formatting.
        const auto str = fmt::to_string(msg.payload);

        // Make sure we lock the GIL before calling into the
        // interpreter, as log messages may be produced by event
        // callbacks in long running integrations while the GIL
        // has been temporarily released.
        py::gil_scoped_acquire acquire;

        // Fetch the Python logger.
        auto py_logger = py::module_::import("logging").attr("getLogger")("heyoka");

        switch (msg.level) {
            case spdlog::level::trace:
                [[fallthrough]];
            case spdlog::level::debug:
                py_logger.attr("debug")(str);
                break;
            case spdlog::level::info:
                py_logger.attr("info")(str);
                break;
            case spdlog::level::warn:
                py_logger.attr("warning")(str);
                break;
            case spdlog::level::err:
                py_logger.attr("error")(str);
                break;
            case spdlog::level::critical:
                py_logger.attr("critical")(str);
                break;
            default:;
        }
    }

    void flush_() override {}
};

// Utility helper to synchronize the logging levels
// of the heyoka C++ logger and the Python one.
void log_sync_levels()
{
    // Fetch the C++ logger.
    auto logger = spdlog::get("heyoka");
    assert(logger);

    // Fetch the Python logger.
    auto log_mod = py::module_::import("logging");
    auto py_logger = log_mod.attr("getLogger")("heyoka");

    // Do the matching.
    switch (logger->level()) {
        case spdlog::level::trace:
            [[fallthrough]];
        case spdlog::level::debug:
            py_logger.attr("setLevel")(log_mod.attr("DEBUG"));
            break;
        case spdlog::level::info:
            py_logger.attr("setLevel")(log_mod.attr("INFO"));
            break;
        case spdlog::level::warn:
            py_logger.attr("setLevel")(log_mod.attr("WARNING"));
            break;
        case spdlog::level::err:
            py_logger.attr("setLevel")(log_mod.attr("ERROR"));
            break;
        case spdlog::level::critical:
            py_logger.attr("setLevel")(log_mod.attr("CRITICAL"));
            break;
        default:;
    }
}

} // namespace

} // namespace detail

void enable_logging()
{
    // Force the creation of the heyoka logger.
    hey::create_logger();

    // Fetch it.
    auto logger = spdlog::get("heyoka");
    assert(logger);

    // Initial creation of the heyoka logger on the
    // Python side.
    auto log_mod = py::module_::import("logging");
    auto py_logger = log_mod.attr("getLogger")("heyoka");

    // Set the initial logging level.
    detail::log_sync_levels();

    // Add the Python sink to the heyoka logger.
    auto sink = std::make_shared<detail::py_sink<std::mutex>>();
    logger->sinks().push_back(std::move(sink));
}

void test_debug_msg()
{
    auto logger = spdlog::get("heyoka");
    assert(logger);

    logger->debug("This is a test debug message");
}

void test_info_msg()
{
    auto logger = spdlog::get("heyoka");
    assert(logger);

    logger->info("This is a test info message");
}

void test_warning_msg()
{
    auto logger = spdlog::get("heyoka");
    assert(logger);

    logger->warn("This is a test warning message {}, {}, {}", 1, 2, 3);
}

void test_error_msg()
{
    auto logger = spdlog::get("heyoka");
    assert(logger);

    logger->error("This is a test error message");
}

void test_critical_msg()
{
    auto logger = spdlog::get("heyoka");
    assert(logger);

    logger->critical("This is a test critical message");
}

void expose_logging_setters(py::module_ &m)
{
    m.def("set_logger_level_debug", []() {
        hey::set_logger_level_debug();
        detail::log_sync_levels();
    });

    m.def("set_logger_level_info", []() {
        hey::set_logger_level_info();
        detail::log_sync_levels();
    });

    m.def("set_logger_level_warning", []() {
        hey::set_logger_level_warn();
        detail::log_sync_levels();
    });

    m.def("set_logger_level_error", []() {
        hey::set_logger_level_err();
        detail::log_sync_levels();
    });

    m.def("set_logger_level_critical", []() {
        hey::set_logger_level_critical();
        detail::log_sync_levels();
    });
}

} // namespace heyoka_py
