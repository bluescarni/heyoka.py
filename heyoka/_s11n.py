# Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import cloudpickle
from threading import Lock


# Helper to create dicts mapping a name to a serialization backend
# and vice-versa.
def _make_s11n_backend_maps():
    import pickle

    ret = {"cloudpickle": cloudpickle, "pickle": pickle}

    try:
        import dill

        ret["dill"] = dill
    except ImportError:
        pass

    inv = dict([(ret[_], _) for _ in ret])

    return ret, inv


_s11n_backend_map, _s11n_backend_inv_map = _make_s11n_backend_maps()

# The currently active s11n backend.
_s11n_backend = cloudpickle

# Lock to protect access to _s11n_backend.
_s11n_backend_mutex = Lock()


def set_serialization_backend(name):
    global _s11n_backend

    if not isinstance(name, str):
        raise TypeError(
            "The serialization backend must be specified as a string, but an object of"
            " type {} was provided instead".format(type(name))
        )

    if name not in _s11n_backend_map:
        raise ValueError(
            "The serialization backend '{}' is not valid. The valid backends are: {}".format(
                name, list(_s11n_backend_map.keys())
            )
        )

    new_backend = _s11n_backend_map[name]

    with _s11n_backend_mutex:
        _s11n_backend = new_backend


def get_serialization_backend():
    with _s11n_backend_mutex:
        return _s11n_backend
