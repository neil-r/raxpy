"""
The following optimziation code was copied from scipy to explore
with raxpy.

Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

from typing import Optional, Tuple
import numpy as np

from scipy.stats.qmc import discrepancy

try:
    from numpy.random import Generator as Generator
except ImportError:

    class Generator:  # type: ignore[no-redef]
        pass


def _perturb_discrepancy(
    sample: np.ndarray, i1: int, i2: int, k: int, disc: float
):
    """Centered discrepancy after an elementary perturbation of a LHS.

    An elementary perturbation consists of an exchange of coordinates between
    two points: ``sample[i1, k] <-> sample[i2, k]``. By construction,
    this operation conserves the LHS properties.

    Parameters
    ----------
    sample : array_like (n, d)
        The sample (before permutation) to compute the discrepancy from.
    i1 : int
        The first line of the elementary permutation.
    i2 : int
        The second line of the elementary permutation.
    k : int
        The column of the elementary permutation.
    disc : float
        Centered discrepancy of the design before permutation.

    Returns
    -------
    discrepancy : float
        Centered discrepancy of the design after permutation.

    References
    ----------
    .. [1] Jin et al. "An efficient algorithm for constructing optimal design
       of computer experiments", Journal of Statistical Planning and
       Inference, 2005.

    """
    n = sample.shape[0]

    z_ij = sample - 0.5

    # Eq (19)
    c_i1j = (
        1.0
        / n**2.0
        * np.prod(
            0.5
            * (2.0 + abs(z_ij[i1, :]) + abs(z_ij) - abs(z_ij[i1, :] - z_ij)),
            axis=1,
        )
    )
    c_i2j = (
        1.0
        / n**2.0
        * np.prod(
            0.5
            * (2.0 + abs(z_ij[i2, :]) + abs(z_ij) - abs(z_ij[i2, :] - z_ij)),
            axis=1,
        )
    )

    # Eq (20)
    c_i1i1 = 1.0 / n**2 * np.prod(1 + abs(z_ij[i1, :])) - 2.0 / n * np.prod(
        1.0 + 0.5 * abs(z_ij[i1, :]) - 0.5 * z_ij[i1, :] ** 2
    )
    c_i2i2 = 1.0 / n**2 * np.prod(1 + abs(z_ij[i2, :])) - 2.0 / n * np.prod(
        1.0 + 0.5 * abs(z_ij[i2, :]) - 0.5 * z_ij[i2, :] ** 2
    )

    # Eq (22), typo in the article in the denominator i2 -> i1
    num = (
        2 + abs(z_ij[i2, k]) + abs(z_ij[:, k]) - abs(z_ij[i2, k] - z_ij[:, k])
    )
    denum = (
        2 + abs(z_ij[i1, k]) + abs(z_ij[:, k]) - abs(z_ij[i1, k] - z_ij[:, k])
    )
    gamma = num / denum

    # Eq (23)
    c_p_i1j = gamma * c_i1j
    # Eq (24)
    c_p_i2j = c_i2j / gamma

    alpha = (1 + abs(z_ij[i2, k])) / (1 + abs(z_ij[i1, k]))
    beta = (2 - abs(z_ij[i2, k])) / (2 - abs(z_ij[i1, k]))

    g_i1 = np.prod(1.0 + abs(z_ij[i1, :]))
    g_i2 = np.prod(1.0 + abs(z_ij[i2, :]))
    h_i1 = np.prod(1.0 + 0.5 * abs(z_ij[i1, :]) - 0.5 * (z_ij[i1, :] ** 2))
    h_i2 = np.prod(1.0 + 0.5 * abs(z_ij[i2, :]) - 0.5 * (z_ij[i2, :] ** 2))

    # Eq (25), typo in the article g is missing
    c_p_i1i1 = (g_i1 * alpha) / (n**2) - 2.0 * alpha * beta * h_i1 / n
    # Eq (26), typo in the article n ** 2
    c_p_i2i2 = (g_i2 / ((n**2) * alpha)) - (2.0 * h_i2 / (n * alpha * beta))

    # Eq (26)
    sum_ = c_p_i1j - c_i1j + c_p_i2j - c_i2j

    mask = np.ones(n, dtype=bool)
    mask[[i1, i2]] = False
    sum_ = sum(sum_[mask])

    disc_ep = disc + c_p_i1i1 - c_i1i1 + c_p_i2i2 - c_i2i2 + 2 * sum_

    return disc_ep


def rng_integers(
    gen, low, high=None, size=None, dtype="int64", endpoint=False
):
    """
    Return random integers from low (inclusive) to high (exclusive), or if
    endpoint=True, low (inclusive) to high (inclusive). Replaces
    `RandomState.randint` (with endpoint=False) and
    `RandomState.random_integers` (with endpoint=True).

    Return random integers from the "discrete uniform" distribution of the
    specified dtype. If high is None (the default), then results are from
    0 to low.

    Parameters
    ----------
    gen : {None, np.random.RandomState, np.random.Generator}
        Random number generator. If None, then the np.random.RandomState
        singleton is used.
    low : int or array-like of ints
        Lowest (signed) integers to be drawn from the distribution (unless
        high=None, in which case this parameter is 0 and this value is used
        for high).
    high : int or array-like of ints
        If provided, one above the largest (signed) integer to be drawn from
        the distribution (see above for behavior if high=None). If array-like,
        must contain integer values.
    size : array-like of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k
        samples are drawn. Default is None, in which case a single value is
        returned.
    dtype : {str, dtype}, optional
        Desired dtype of the result. All dtypes are determined by their name,
        i.e., 'int64', 'int', etc, so byteorder is not available and a specific
        precision may have different C types depending on the platform.
        The default value is 'int64'.
    endpoint : bool, optional
        If True, sample from the interval [low, high] instead of the default
        [low, high) Defaults to False.

    Returns
    -------
    out: int or ndarray of ints
        size-shaped array of random integers from the appropriate distribution,
        or a single such random int if size not provided.
    """
    if isinstance(gen, Generator):
        return gen.integers(
            low, high=high, size=size, dtype=dtype, endpoint=endpoint
        )
    else:
        if gen is None:
            # default is RandomState singleton used by np.random.
            gen = np.random.mtrand._rand
        if endpoint:
            # inclusive of endpoint
            # remember that low and high can be arrays, so don't modify in
            # place
            if high is None:
                return gen.randint(low + 1, size=size, dtype=dtype)
            if high is not None:
                return gen.randint(low, high=high + 1, size=size, dtype=dtype)

        # exclusive
        return gen.randint(low, high=high, size=size, dtype=dtype)


def random_cd(
    best_sample: np.ndarray,
    n_iters: int,
    n_nochange: int,
    column_bounds: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Optimal LHS on CD.

    Create a base LHS and do random permutations of coordinates to
    lower the centered discrepancy.
    Because it starts with a normal LHS, it also works with the
    `scramble` keyword argument.

    Two stopping criterion are used to stop the algorithm: at most,
    `n_iters` iterations are performed; or if there is no improvement
    for `n_nochange` consecutive iterations.
    """

    rng = np.random.default_rng()

    n, d = best_sample.shape

    if d == 0 or n == 0:
        return np.empty((n, d))

    if d == 1 or n == 1:
        # discrepancy measures are invariant under permuting single factors and runs
        return best_sample

    best_disc = discrepancy(best_sample)

    if column_bounds is None:
        column_bounds = (0, d - 1)

    bounds = (column_bounds, (0, n - 1), (0, n - 1))

    n_nochange_ = 0
    n_iters_ = 0
    while n_nochange_ < n_nochange and n_iters_ < n_iters:
        n_iters_ += 1

        col = rng_integers(rng, *bounds[0], endpoint=True)  # type: ignore[misc]
        row_1 = rng_integers(rng, *bounds[1], endpoint=True)  # type: ignore[misc]
        row_2 = rng_integers(rng, *bounds[2], endpoint=True)  # type: ignore[misc]
        disc = _perturb_discrepancy(best_sample, row_1, row_2, col, best_disc)
        if disc < best_disc:
            best_sample[row_1, col], best_sample[row_2, col] = (
                best_sample[row_2, col],
                best_sample[row_1, col],
            )

            best_disc = disc
            n_nochange_ = 0
        else:
            n_nochange_ += 1

    return best_sample
