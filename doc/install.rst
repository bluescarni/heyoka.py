Installation
============

Dependencies
------------

heyoka.py has the following **mandatory** runtime dependencies:

* `Python <https://www.python.org/>`__ 3.4 or later (Python 2.x is
  **not** supported),
* the `heyoka C++ library <https://github.com/bluescarni/heyoka>`__,
  version 0.10.0 or later,
* the `{fmt} <https://fmt.dev/latest/index.html>`__ library,
* the `spdlog <https://github.com/gabime/spdlog>`__ library,
* `NumPy <https://numpy.org/>`__.

Additionally, heyoka.py has the following **optional** runtime
dependencies:

* `mpmath <https://mpmath.org/>`__ (necessary if the heyoka C++ library
  was compiled with support for quadruple-precision computations),
* the `mp++ library <https://github.com/bluescarni/mppp>`__ (necessary if the heyoka C++ library
  was compiled with support for quadruple-precision computations),
* `SymPy <https://www.sympy.org/en/index.html>`__ (for converting heyoka.py
  expressions to/from SymPy expressions).

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

In order to install heyoka.py from source, you will need:

* a C++17 capable compiler (recent versions of GCC,
  Clang or MSVC should do),
* a `Python <https://www.python.org/>`__ installation,
* `pybind11 <https://github.com/pybind/pybind11>`__ (version >= 2.6),
* the `heyoka C++ library <https://github.com/bluescarni/heyoka>`__,
  version 0.10.0 or later,
* the `{fmt} <https://fmt.dev/latest/index.html>`__ library,
* the `spdlog <https://github.com/gabime/spdlog>`__ library,
* the `Boost libraries <https://www.boost.org/>`__,
* the `mp++ library <https://github.com/bluescarni/mppp>`__ (optional,
  necessary only if the heyoka C++ library
  was compiled with support for quadruple-precision computations),
* `CMake <https://cmake.org/>`__, version 3.8 or later.

After making sure the dependencies are installed on your system, you can
download the heyoka.py source code from the
`GitHub release page <https://github.com/bluescarni/heyoka.py/releases>`__. Alternatively,
and if you like living on the bleeding edge, you can get the very latest
version of heyoka.py via ``git``:

.. code-block:: console

   $ git clone https://github.com/bluescarni/heyoka.py.git

We follow the usual PR-based development workflow, thus heyoka.py's ``master``
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
