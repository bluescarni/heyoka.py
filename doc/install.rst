Installation
============

.. _installation_deps:

Dependencies
------------

heyoka.py has several Python and C++ dependencies. On the C++ side, heyoka.py depends on:

* the `heyoka C++ library <https://github.com/bluescarni/heyoka>`__,
  version 0.14.0 or later (**mandatory**),
* the `Boost <https://www.boost.org/>`__ C++ libraries (**mandatory**),
* the `{fmt} <https://fmt.dev/latest/index.html>`__ library (**mandatory**),
* the `spdlog <https://github.com/gabime/spdlog>`__ library (**mandatory**),
* the `mp++ <https://github.com/bluescarni/mppp>`__ library (**mandatory** if the
  heyoka C++ library was compiled with the ``HEYOKA_WITH_MPPP`` option on - see the
  :ref:`heyoka C++ installation instructions <hy:installation>`).

On the Python side, heyoka.py requires at least Python 3.4
(Python 2.x is **not** supported) and depends on:

* `NumPy <https://numpy.org/>`__ (**mandatory**),
* `cloudpickle <https://github.com/cloudpipe/cloudpickle>`__ (**mandatory**),
* `mpmath <https://mpmath.org/>`__ (**mandatory** if the
  heyoka C++ library was compiled with the ``HEYOKA_WITH_MPPP`` option on - see the
  :ref:`heyoka C++ installation instructions <hy:installation>`),
* `SymPy <https://www.sympy.org/en/index.html>`__ (*optional*, for converting heyoka.py
  expressions to/from SymPy expressions).

The tested and supported CPU architectures at this time are x86-64 and 64-bit ARM.

Packages
--------

Conda
^^^^^

heyoka.py is available via the `conda <https://conda.io/docs/>`__
package manager for Linux, OSX and Windows
thanks to the infrastructure provided by `conda-forge <https://conda-forge.org/>`__.
In order to install heyoka.py via conda, you just need to add ``conda-forge``
to the channels, and then we can immediately install heyoka.py:

.. code-block:: console

   $ conda config --add channels conda-forge
   $ conda config --set channel_priority strict
   $ conda install heyoka.py

The conda packages for heyoka.py are maintained by the core development team,
and they are regularly updated when new heyoka.py versions are released.

Please refer to the `conda documentation <https://conda.io/docs/>`__
for instructions on how to setup and manage
your conda installation.

Installation from source
------------------------

heyoka.py is written in modern C++, and it requires a compiler able to understand
at least C++17. The library is regularly tested on
a continuous integration pipeline which currently includes:

* GCC 9 on Linux,
* Clang 11 on OSX,
* MSVC 2017 + ``clang-cl`` on Windows.

In addition to the dependencies enumerated :ref:`earlier <installation_deps>`,
installation from source requires also:

* `pybind11 <https://github.com/pybind/pybind11>`__ (version >= 2.6),
* `CMake <https://cmake.org/>`__, version 3.8 or later.

After making sure the dependencies are installed on your system, you can
download the heyoka.py source code from the
`GitHub release page <https://github.com/bluescarni/heyoka.py/releases>`__. Alternatively,
and if you like living on the bleeding edge, you can get the very latest
version of heyoka.py via ``git``:

.. code-block:: console

   $ git clone https://github.com/bluescarni/heyoka.py.git

We follow the usual PR-based development workflow, thus heyoka.py's ``main``
branch is normally kept in a working state.

After downloading and/or unpacking heyoka.py's
source code, go to heyoka.py's
source tree, create a ``build`` directory and ``cd`` into it. E.g.,
on a Unix-like system:

.. code-block:: console

   $ cd /path/to/heyoka.py
   $ mkdir build
   $ cd build

Once you are in the ``build`` directory, you must configure your build
using ``cmake``. There are various useful CMake variables you can set,
such as:

* ``CMAKE_BUILD_TYPE``: the build type (``Release``, ``Debug``, etc.),
  defaults to ``Release``.
* ``CMAKE_PREFIX_PATH``: additional paths that will be searched by CMake
  when looking for dependencies.
* ``HEYOKA_PY_INSTALL_PATH``: the path into which the heyoka.py module
  will be installed. If left empty (the default), heyoka.py will be installed
  in the global modules directory of your Python installation.
* ``HEYOKA_PY_ENABLE_IPO``: set this flag to ``ON`` to compile heyoka.py
  with link-time optimisations. Requires CMake >= 3.9 and compiler support,
  defaults to ``OFF``.

Please consult `CMake's documentation <https://cmake.org/cmake/help/latest/>`_
for more details about CMake's variables and options.

The ``HEYOKA_PY_INSTALL_PATH`` option is particularly important. If you
want to install heyoka.py locally instead of globally (which is in general
a good idea), you can set this variable to the output of
``python -m site --user-site``.

After configuring the build with CMake, we can then proceed to actually
building heyoka.py:

.. code-block:: console

   $ cmake --build .

Finally, we can install heyoka.py with the command:

.. code-block:: console

   $ cmake  --build . --target install

Verifying the installation
--------------------------

You can verify that heyoka.py was successfully compiled and
installed by running the test suite. From a
Python session, run the following commands:

.. code-block:: python

   >>> import heyoka
   >>> heyoka.test.run_test_suite()

If these commands execute without any error, then
your heyoka.py installation is ready for use.

Getting help
------------

If you run into troubles installing heyoka.py, please do not hesitate
to contact us by opening an issue report on `github <https://github.com/bluescarni/heyoka.py/issues>`__.
