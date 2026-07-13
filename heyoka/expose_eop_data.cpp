// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <optional>
#include <string>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fmt/core.h>

#include <heyoka/eop_data.hpp>

#include "common_utils.hpp"
#include "docstrings.hpp"
#include "expose_eop_data.hpp"
#include "pickle_wrappers.hpp"

namespace heyoka_py
{

namespace py = pybind11;

void expose_eop_data(py::module_ &m)
{
    namespace hy = heyoka;
    using namespace pybind11::literals;

    // Add the eop_data_row dtype as a module attribute.
    m.attr("eop_data_row") = make_aligned_dtype<hy::eop_data_row>();

    // Expose the eop_data class.
    py::class_<hy::eop_data> eop_data_class(m, "eop_data", py::dynamic_attr{}, docstrings::eop_data().c_str());
    eop_data_class.def(py::init([](const std::optional<py::array> &data, const std::optional<std::string> &timestamp,
                                   const std::optional<std::string> &identifier) {
                           return eop_sw_ctor<hy::eop_data>("EOP", data, timestamp, identifier);
                       }),
                       py::kw_only(), "data"_a = py::none{}, "timestamp"_a = py::none{}, "identifier"_a = py::none{},
                       docstrings::eop_data_init().c_str());
    eop_data_class.def_property_readonly(
        "table",
        [](const py::object &o) {
            auto *edata = py::cast<const hy::eop_data *>(o);
            const auto &table = edata->get_table();

            auto ret = py::array(make_aligned_dtype<hy::eop_data_row>(), boost::numeric_cast<py::ssize_t>(table.size()),
                                 table.data(), o);

            // Ensure the returned array is read-only.
            ret.attr("flags").attr("writeable") = false;

            return ret;
        },
        docstrings::eop_data_table().c_str());
    eop_data_class.def_property_readonly("timestamp", &hy::eop_data::get_timestamp,
                                         docstrings::eop_data_timestamp().c_str());
    eop_data_class.def_property_readonly("identifier", &hy::eop_data::get_identifier,
                                         docstrings::eop_data_identifier().c_str());
    eop_data_class.def_static(
        "fetch_latest_iers_rapid",
        [](const std::string &origin, const std::string &filename) {
            // NOTE: release the GIL during download.
            py::gil_scoped_release release;

            return hy::eop_data::fetch_latest_iers_rapid(origin, filename);
        },
        "origin"_a = "usno", "filename"_a = "finals2000A.all", docstrings::eop_data_fetch_latest_iers_rapid().c_str());
    eop_data_class.def_static(
        "fetch_latest_iers_long_term",
        []() {
            // NOTE: release the GIL during download.
            py::gil_scoped_release release;

            return hy::eop_data::fetch_latest_iers_long_term();
        },
        docstrings::eop_data_fetch_latest_iers_long_term().c_str());
    eop_data_class.def_static(
        "fetch_latest_celestrak",
        [](const bool long_term) {
            // NOTE: release the GIL during download.
            py::gil_scoped_release release;

            return hy::eop_data::fetch_latest_celestrak(long_term);
        },
        "long_term"_a = false, docstrings::eop_data_fetch_latest_celestrak().c_str());
    // Repr.
    eop_data_class.def("__repr__", [](const hy::eop_data &data) {
        return fmt::format("N of rows : {}\nTimestamp : {}\nIdentifier: {}\n", data.get_table().size(),
                           data.get_timestamp(), data.get_identifier());
    });
    // Copy/deepcopy.
    eop_data_class.def("__copy__", copy_wrapper<hy::eop_data>);
    eop_data_class.def("__deepcopy__", deepcopy_wrapper<hy::eop_data>, "memo"_a);
    // Pickle support.
    eop_data_class.def(py::pickle(&pickle_getstate_wrapper<hy::eop_data>, &pickle_setstate_wrapper<hy::eop_data>));
}

} // namespace heyoka_py
