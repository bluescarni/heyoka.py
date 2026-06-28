# Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the heyoka.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from copy import deepcopy, copy
from ._s11n import _s11n_backend_map, get_serialization_backend, _s11n_backend_inv_map
import multiprocessing as mp
from collections.abc import Iterable


# NOTE: this is a small helper to splat a 1D grid (i.e., a grid for
# a scalar integrator) into the appropriate shape for a batch integrator.
# If ta is a scalar integrator, the original grid will be returned unchanged.
def _splat_grid(arg, ta):
    if hasattr(ta, "batch_size"):
        return np.repeat(arg, ta.batch_size).reshape((-1, ta.batch_size))
    else:
        return arg


# Thread-based implementation.
def _ensemble_propagate_thread(tp, ta, arg, n_iter, gen, **kwargs):
    # Pop the multithreading options from kwargs.
    max_workers = kwargs.pop("max_workers", None)

    # Make deep copies of the callback argument, if present.
    if "callback" in kwargs:
        kwargs_list = []

        for i in range(n_iter):
            # Make a shallow copy of the original kwargs.
            # new_kwargs will be a new dict containing
            # references to the objects stored in kwargs.
            new_kwargs = copy(kwargs)

            # Update the callback argument in new_kwargs
            # with a deep copy of the original callback object in
            # kwargs.
            new_kwargs.update(callback=deepcopy(kwargs["callback"]))

            kwargs_list.append(new_kwargs)
    else:
        kwargs_list = [kwargs] * n_iter

    # The worker function.
    def func(i):
        # Create the local integrator.
        local_ta = gen(deepcopy(ta), i)

        # Run the propagation.
        if tp == "until":
            loc_ret = local_ta.propagate_until(arg, **kwargs_list[i])
        elif tp == "for":
            loc_ret = local_ta.propagate_for(arg, **kwargs_list[i])
        else:
            loc_ret = local_ta.propagate_grid(_splat_grid(arg, ta), **kwargs_list[i])

        # Return the results.
        return (local_ta,) + loc_ret

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        ret = list(executor.map(func, range(n_iter)))

    return ret


# The worker function used in the multiprocessing implementation.
def _mp_propagate(tup):
    tp, ta, gen, arg, kwargs, i, s11n_str = tup

    # Fetch the s11n backend from its
    # str representation.
    s11n_be = _s11n_backend_map[s11n_str]

    # Unpickle the other arguments.
    ta = s11n_be.loads(ta)
    gen = s11n_be.loads(gen)
    kwargs = s11n_be.loads(kwargs)

    # Create the local integrator.
    local_ta = gen(ta, i)

    # Run the propagation.
    if tp == "until":
        loc_ret = local_ta.propagate_until(arg, **kwargs)
    elif tp == "for":
        loc_ret = local_ta.propagate_for(arg, **kwargs)
    else:
        loc_ret = local_ta.propagate_grid(_splat_grid(arg, ta), **kwargs)

    # Return the results.
    return s11n_be.dumps((local_ta,) + loc_ret)


# Process-based implementation.
def _ensemble_propagate_process(tp, ta, arg, n_iter, gen, **kwargs):
    # Fetch the currently active s11n backend.
    s11n_be = get_serialization_backend()

    # Fetch its string counterpart.
    s11n_str = _s11n_backend_inv_map[s11n_be]

    # NOTE: ensure the processes are started with
    # the 'spawn' method.
    ctx = mp.get_context("spawn")

    # Pop the multiprocessing options from kwargs.
    max_workers = kwargs.pop("max_workers", None)
    chunksize = kwargs.pop("chunksize", 1)

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        ret = list(
            executor.map(
                _mp_propagate,
                zip(
                    [tp] * n_iter,
                    [s11n_be.dumps(ta)] * n_iter,
                    [s11n_be.dumps(gen)] * n_iter,
                    [arg] * n_iter,
                    [s11n_be.dumps(kwargs)] * n_iter,
                    range(n_iter),
                    [s11n_str] * n_iter,
                ),
                chunksize=chunksize,
            )
        )

    return [s11n_be.loads(_) for _ in ret]


# Ensemble propagations.
def _ensemble_propagate_generic(tp, ta, arg, n_iter, gen, **kwargs):
    if not isinstance(n_iter, int):
        raise TypeError(
            "The n_iter parameter must be an integer, but an object of type {} was"
            " provided instead".format(type(n_iter))
        )

    if n_iter < 0:
        raise ValueError(
            "The n_iter parameter must be non-negative, but it is {} instead".format(
                n_iter
            )
        )

    # Validate arg and max_delta_t, if present.
    def is_iterable(x):
        return isinstance(x, Iterable)

    if tp == "until" or tp == "for":
        if is_iterable(arg):
            raise TypeError(
                "Cannot perform an ensemble propagate_until/for(): the final epoch/time"
                " interval must be a scalar, not an iterable object"
            )
    else:
        arg = np.array(arg)

        if arg.ndim != 1:
            raise ValueError(
                "Cannot perform an ensemble propagate_grid(): the input time grid must"
                " be one-dimensional, but instead it has {} dimensions".format(arg.ndim)
            )

    if "max_delta_t" in kwargs and is_iterable(kwargs["max_delta_t"]):
        raise TypeError(
            'Cannot perform an ensemble propagate_until/for/grid(): the "max_delta_t"'
            " argument must be a scalar, not an iterable object"
        )

    # Parallelisation algorithm.
    algo = kwargs.pop("algorithm", "thread")
    allowed_algos = ["thread", "process"]

    if algo == "thread":
        return _ensemble_propagate_thread(tp, ta, arg, n_iter, gen, **kwargs)

    if algo == "process":
        return _ensemble_propagate_process(tp, ta, arg, n_iter, gen, **kwargs)

    raise ValueError(
        "The parallelisation algorithm must be one of {}, but '{}' was provided instead".format(
            allowed_algos, algo
        )
    )


def ensemble_propagate_until(ta, t, n_iter, gen, **kwargs):
    return _ensemble_propagate_generic("until", ta, t, n_iter, gen, **kwargs)


def ensemble_propagate_for(ta, delta_t, n_iter, gen, **kwargs):
    return _ensemble_propagate_generic("for", ta, delta_t, n_iter, gen, **kwargs)


def ensemble_propagate_grid(ta, grid, n_iter, gen, **kwargs):
    return _ensemble_propagate_generic("grid", ta, grid, n_iter, gen, **kwargs)


def ensemble_propagate_until_batch(ta, t, n_iter, gen, **kwargs):
    return _ensemble_propagate_generic("until", ta, t, n_iter, gen, **kwargs)


def ensemble_propagate_for_batch(ta, delta_t, n_iter, gen, **kwargs):
    return _ensemble_propagate_generic("for", ta, delta_t, n_iter, gen, **kwargs)


def ensemble_propagate_grid_batch(ta, grid, n_iter, gen, **kwargs):
    return _ensemble_propagate_generic("grid", ta, grid, n_iter, gen, **kwargs)
