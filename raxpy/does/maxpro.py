"""
    Provides extensions to the MaxPro optmization algorithm 
    to support design optimizations.

    See "Designing Computer Experiments with
    Multiple Types of Factors: The MaxPro Approach"
    for more details about MaxPro.

    https://par.nsf.gov/servlets/purl/10199193
"""

import random
import numpy as np
import math
from ..spaces.complexity import estimate_complexity

from .. import spaces as s
from .doe import DesignOfExperiment, EncodingEnum


def _create_max_pro_dist_func(dim):
    level_factor = 1.0 / estimate_complexity(dim)

    def f(x1_value, x2_value):
        if not np.isnan(x1_value) and not np.isnan(x2_value):
            return abs(x1_value - x2_value) + level_factor
        else:
            if np.isnan(x1_value) and np.isnan(x2_value):
                return level_factor
            else:
                return 1.0 + level_factor

    return f


def optimize_design(
    base_design: DesignOfExperiment,
    encoding: EncodingEnum = EncodingEnum.ZERO_ONE_NULL_ENCODING,
) -> DesignOfExperiment:
    opt_design = base_design.copy()

    # initalize data structures
    d_best = opt_design.get_data_points(encoding)

    n = opt_design.point_count
    point_comps = np.zeros((n, n))

    d = opt_design.dim_specification_count

    index_dim_map = opt_design.index_dim_id_map

    dim_map = opt_design.input_space.create_dim_map()

    root_dim_ids = {
        dim.id: True
        for dim in s.create_level_iterable(opt_design.input_space.children)
    }

    dist_funcs = []
    # algorithm avoids swapping values for dimensions that define heirarchy
    column_indices_to_swap = []
    # algorithm avoids swapping null values to retain original's design optionality features
    active_row_indicies = {}

    for k in range(d):
        dim_id = index_dim_map[k]
        dim = dim_map[dim_id]
        if not dim.has_child_dimensions():
            column_indices_to_swap.append(k)
            active_row_indicies[k] = list(
                i for i in range(n) if not np.isnan(d_best[i, k])
            )

            if len(active_row_indicies[k]) <= 1:
                # avoid considering this column since not enough values to swap
                del active_row_indicies[k]
                column_indices_to_swap.pop()

        if (
            dim.portion_null > 0
            or dim_id not in root_dim_ids
            or dim.has_finite_values()
        ):
            dist_funcs.append(_create_max_pro_dist_func(dim))
        else:
            dist_funcs.append(lambda x1_value, x2_value: x1_value - x2_value)

    for i in range(n - 1):
        for j in range(i + 1, n):
            m = 1.0
            for k in range(0, d):
                m *= dist_funcs[k](d_best[i, k], d_best[j, k]) ** 2

            point_comps[i, j] = 1.0 / m

    d_try = np.copy(d_best)
    d_try_point_comps = np.copy(point_comps)
    d_best_value = np.sum(np.triu(point_comps, -1))

    # Optimization Stage: Iteratively search in the design space to
    # optimize the criterion (9) using a version of the simulated annealing
    # algorithm (Morris and Mitchell 1995).
    temp = 0
    max_temp = 4
    attempt_from_best_count = 0
    max_attempt_from_best_count = 100000
    maxiter = 100000

    for _ in range(maxiter):
        attempt_from_best_count += 1
        if attempt_from_best_count > max_attempt_from_best_count:
            break
        temp += 1
        # Step 5. Denote the current design matrix as D = [Dx, Du, Dv]. Randomly
        # choose a column from the [Dx, Du] components, and interchange two
        # randomly chosen elements within the selected column. Denote the new
        # design matrix as Dtry.
        c_i = random.choice(column_indices_to_swap)
        row_indices = active_row_indicies[c_i]

        r_1 = random.choice(row_indices)
        r_2 = random.choice(row_indices)
        # Step 6. If Dtry = D, repeat Step (5).
        while r_1 == r_2:
            r_2 = random.choice(row_indices)

        d_try[r_1, c_i], d_try[r_2, c_i] = d_try[r_2, c_i], d_try[r_1, c_i]

        # adjust values that changed
        for i in [r_1, r_2]:
            # up
            for j in range(0, i):
                m = 1.0
                for k in range(0, d):
                    m *= dist_funcs[k](d_try[i, k], d_try[j, k]) ** 2

                d_try_point_comps[j, i] = 1.0 / m

            # over
            for j in range(i + 1, n):
                m = 1.0
                for k in range(0, d):
                    m *= dist_funcs[k](d_try[i, k], d_try[j, k]) ** 2

                d_try_point_comps[i, j] = 1.0 / m

        # Step 7. If ψ(Dtry) < ψ(D), replace the current design D with Dtry;
        # otherwise, replace the current design D with Dtry with probability
        # π = exp{−[ψ(Dtry) − ψ(D)]/T }, where T is a preset parameter
        # known as “temperature”.
        d_try_value = np.sum(np.triu(d_try_point_comps, -1))
        if d_try_value < d_best_value:
            print("Found new best!")
            d_best[:, :] = d_try
            point_comps[:, :] = d_try_point_comps
            d_best_value = d_try_value
            attempt_from_best_count = 0
        elif temp >= max_temp:
            # reset
            temp = 0
            d_try[:, :] = d_best
            d_try_point_comps[:, :] = point_comps

            # Step 8. Repeat Step (5) to Step (7) until some convergence requirements
            # are met. Report the design matrix with the smallest ψ(D) value as the
            # optimal design with respect to criterion (9).
    opt_design.input_sets[:, :] = d_best
    return opt_design


def optimize_design_with_sa(
    base_design: DesignOfExperiment,
    encoding: EncodingEnum = EncodingEnum.ZERO_ONE_NULL_ENCODING,
    maxiter: int = 10000,
    max_rabbit_whole_threshold: int = 1,
) -> DesignOfExperiment:
    opt_design = base_design.copy()

    # initalize data structures
    d_best = opt_design.get_data_points(encoding)

    n = opt_design.point_count
    point_comps = np.zeros((n, n))

    d = opt_design.dim_specification_count

    index_dim_map = opt_design.index_dim_id_map

    dim_map = opt_design.input_space.create_dim_map()

    root_dim_ids = {
        dim.id: True
        for dim in s.create_level_iterable(opt_design.input_space.children)
    }

    dist_funcs = []
    # algorithm avoids swapping values for dimensions that define heirarchy
    column_indices_to_swap = []
    # algorithm avoids swapping null values to retain original's design optionality features
    active_row_indicies = {}

    for k in range(d):
        dim_id = index_dim_map[k]
        dim = dim_map[dim_id]
        if not dim.has_child_dimensions():
            column_indices_to_swap.append(k)
            active_row_indicies[k] = list(
                i for i in range(n) if not np.isnan(d_best[i, k])
            )

            if len(active_row_indicies[k]) <= 1:
                # avoid considering this column since not enough values to swap
                del active_row_indicies[k]
                column_indices_to_swap.pop()

        if (
            dim.portion_null > 0
            or dim_id not in root_dim_ids
            or dim.has_finite_values()
        ):
            dist_funcs.append(_create_max_pro_dist_func(dim))
        else:
            dist_funcs.append(lambda x1_value, x2_value: x1_value - x2_value)

    for i_a in range(n - 1):
        for j_a in range(i_a + 1, n):
            m = 1.0
            for k in range(0, d):
                m *= dist_funcs[k](d_best[i_a, k], d_best[j_a, k]) ** 2

            point_comps[i_a, j_a] = 1.0 / m

    d_try = np.copy(d_best)
    d_try_point_comps = np.copy(point_comps)
    d_best_value = np.sum(np.triu(point_comps, -1))

    # Optimization Stage: Iteratively search in the design space to
    # optimize the criterion (9) using a version of the simulated annealing
    # algorithm (Morris and Mitchell 1995).
    dig_count = 0
    attempt_from_best_count = 0
    max_attempt_from_best_count = 100000

    def revise_temp(i):
        return 1.0 - (i / maxiter)

    for i_iteration in range(maxiter):
        t = revise_temp(i_iteration)
        dig_count += 1

        attempt_from_best_count += 1
        if attempt_from_best_count > max_attempt_from_best_count:
            break
        # Step 5. Denote the current design matrix as D = [Dx, Du, Dv]. Randomly
        # choose a column from the [Dx, Du] components, and interchange two
        # randomly chosen elements within the selected column. Denote the new
        # design matrix as Dtry.
        k = random.choice(column_indices_to_swap)
        row_indices = active_row_indicies[k]

        i = random.choice(row_indices)
        j = random.choice(row_indices)
        # Step 6. If Dtry = D, repeat Step (5).
        while i == j:
            j = random.choice(row_indices)

        d_try[i, k], d_try[j, k] = d_try[j, k], d_try[i, k]

        # adjust values that changed
        for i_a in [i, j]:
            # up
            for j_a in range(0, i_a):
                m = 1.0
                for k in range(0, d):
                    m *= dist_funcs[k](d_try[i_a, k], d_try[j_a, k]) ** 2

                d_try_point_comps[j_a, i_a] = 1.0 / m

            # over
            for j_a in range(i_a + 1, n):
                m = 1.0
                for k in range(0, d):
                    m *= dist_funcs[k](d_try[i_a, k], d_try[j_a, k]) ** 2

                d_try_point_comps[i_a, j_a] = 1.0 / m

        # Step 7. If ψ(Dtry) < ψ(D), replace the current design D with Dtry;
        # otherwise, replace the current design D with Dtry with probability
        # π = exp{−[ψ(Dtry) − ψ(D)]/T }, where T is a preset parameter
        # known as “temperature”.
        d_try_value = np.sum(np.triu(d_try_point_comps, -1))
        p_threshold = math.e ** (-(d_try_value - d_best_value) / t)
        if d_try_value < d_best_value or p_threshold > random.random():
            print("Found new best!")
            d_best[:, :] = d_try
            point_comps[:, :] = d_try_point_comps
            d_best_value = d_try_value
            attempt_from_best_count = 0
            dig_count = 0
        elif dig_count >= max_rabbit_whole_threshold:
            # reset
            dig_count = 0
            d_try[:, :] = d_best
            d_try_point_comps[:, :] = point_comps

        # Step 8. Repeat Step (5) to Step (7) until some convergence requirements
        # are met. Report the design matrix with the smallest ψ(D) value as the
        # optimal design with respect to criterion (9).
    opt_design.input_sets[:, :] = d_best
    return opt_design
