# Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest


class package_layout_test_case(unittest.TestCase):
    # Names exposed by the compiled _core module that are intentionally *not*
    # re-exported by any public (sub)package. This should normally be empty;
    # add an entry here (with a justification) if a public _core name must
    # remain internal.
    _allowed_unexported = set()

    def test_core_names_are_partitioned(self):
        # Every public (non-underscore) name provided by the compiled _core
        # module must be re-exported - by identity and via __all__ - by exactly
        # one of the public packages. This guards against two failure modes:
        # a new _core binding that nobody re-exports (orphan), and the same
        # name leaking out of more than one package (duplicate).
        import heyoka
        from heyoka import _core

        packages = {
            "heyoka": heyoka,
            "heyoka.model": heyoka.model,
            "heyoka.callback": heyoka.callback,
        }

        orphaned = []
        duplicated = []
        for name in dir(_core):
            if name.startswith("_") or name in self._allowed_unexported:
                continue
            obj = getattr(_core, name)
            homes = [
                pkg_name
                for pkg_name, pkg in packages.items()
                if name in getattr(pkg, "__all__", ())
                and getattr(pkg, name, None) is obj
            ]
            if len(homes) == 0:
                orphaned.append(name)
            elif len(homes) > 1:
                duplicated.append((name, homes))

        self.assertEqual(
            orphaned,
            [],
            msg="public _core names re-exported by no package: {}".format(orphaned),
        )
        self.assertEqual(
            duplicated,
            [],
            msg="public _core names re-exported by multiple packages: {}".format(
                duplicated
            ),
        )

    def test_all_entries_are_bound(self):
        # Everything advertised in a package's __all__ must actually be an
        # attribute of that package, otherwise 'from <pkg> import *' is broken.
        import heyoka

        for pkg in (heyoka, heyoka.model, heyoka.callback):
            for name in pkg.__all__:
                self.assertTrue(
                    hasattr(pkg, name),
                    msg="{}.__all__ lists '{}', which is not bound".format(
                        pkg.__name__, name
                    ),
                )
