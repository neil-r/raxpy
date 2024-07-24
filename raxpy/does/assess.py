"""
    This modules provides logic to compute assessments of an experiment design.
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import math
import itertools

import numpy as np
from scipy.stats.qmc import discrepancy
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from .doe import DesignOfExperiment
from ..spaces import root as s


@dataclass
class SubSpaceMetricComputeContext:
    """
    A class used to gather the values needed to compute metrics for a
    subspace design.
    """

    space: s.InputSpace
    whole_doe: DesignOfExperiment
    sub_space_doe: DesignOfExperiment


# DOE Metrics
METRIC_PORTION_SUBSPACES_INCLUDED = "portion_of_subspaces_included"

METRIC_TARGET_PORTION_OFFSET = "target_portion_offset"

METRIC_AVG_AVG_PORTION_OF_SUBSPACE_LEVELS = "avg_avg_subspace_levels"

# the following map is used in the assess function to discover the metrics to compute
doe_metric_computation_map = {}

# Complete SubDesign Metrics
METRIC_AVG_PORTION_LEVELS_INCLUDED = "avg_portion_of_levels_included"

METRIC_PORTION_OF_TOTAL = "portion_of_total"

METRIC_DISCREPANCY = "discrepancy"

METRIC_MIN_POINT_DISTANCE = "max_min_point_distance"


def compute_min_point_distance(context: SubSpaceMetricComputeContext) -> float:
    """
    Computes and returns the minimum-interpoint-distance (MIPD) among every
    pair of points

    Returns:
    float: the minimum-interpoint-distance
    """
    points = context.sub_space_doe.input_sets
    # compute the distances for each point combination
    dm = distance_matrix(points, points)
    np.fill_diagonal(dm, np.inf)

    # find the min distances to the each other point
    return np.min(dm)


def compute_average_reciprocal_distance_projection(
    context: SubSpaceMetricComputeContext, lambda_hp=2, z_hp=2
):
    """
    Implementation of the Average reciprocal distance projection metric as
    denoted in: Draguljić, Santner, and Dean, “Noncollapsing Space-Filling
    Designs for Bounded Nonrectangular Regions.”
    """
    n = context.sub_space_doe.point_count
    p = context.sub_space_doe.dim_specification_count
    big_p = list(range(0, p))
    big_j = list(range(1, p + 1))

    comb_count = 0

    # compute the reciprocal sum for every subspace projection
    running_sum = 0.0
    for j in big_j:
        index_combinations = itertools.combinations(big_p, j)

        max_j_distance = j ** (1.0 / z_hp)

        max_j_distance_m = np.ones((n, n)) * max_j_distance

        for index_combination in index_combinations:

            x_projection = context.sub_space_doe.input_sets[:, index_combination]

            dm = distance_matrix(x_projection, x_projection, p=z_hp)
            # fill diagonal to avoid divide by zero
            # The equation does not need the diagonal elements anyway
            np.fill_diagonal(dm, 1)
            reciprocal_distances = (max_j_distance_m / dm) ** lambda_hp

            # sum the upper triangle matrix elements, excluding the diagonal elements
            reciprocal_distances_sum = np.sum(np.tril(reciprocal_distances, k=1))

            running_sum += reciprocal_distances_sum
            comb_count += 1

    return (running_sum / (math.comb(n, 2) * comb_count)) ** (1.0 / lambda_hp)


def compute_mst_stats(context: SubSpaceMetricComputeContext) -> Tuple[float, float]:
    """
    Computes and returns the mean and standard deviation of the edge-values of
    a minimum spanning tree (MST) of the design points. The edge-values
    represent the distances between design points.

    For more information see https://doi.org/10.1016/j.chemolab.2009.03.011

    Returns:
    Tuple[float, float]: a tuple of the mean and standard deviation of the MST
        edges
    """
    points = context.sub_space_doe.input_sets
    # compute the distances for each point combination
    dm = distance_matrix(points, points)

    mst = minimum_spanning_tree(dm)

    edge_matrix = mst.toarray()
    included_edge_idxs = np.flatnonzero(edge_matrix)
    included_edges = edge_matrix.ravel()[included_edge_idxs]

    mst_mean = np.mean(included_edges)
    mst_std = np.std(included_edges)
    return mst_mean, mst_std


def compute_discrepancy(context: SubSpaceMetricComputeContext) -> float:
    return discrepancy(context.sub_space_doe.input_sets)


def compute_portion_of_total(context: SubSpaceMetricComputeContext) -> float:
    return context.sub_space_doe.point_count / context.whole_doe.point_count


def compute_avg_portion_of_levels(context: SubSpaceMetricComputeContext) -> float:
    pass


subspace_metric_computation_map = {
    METRIC_DISCREPANCY: compute_discrepancy,
    METRIC_PORTION_OF_TOTAL: compute_portion_of_total,
    METRIC_AVG_PORTION_LEVELS_INCLUDED: compute_avg_portion_of_levels,
    METRIC_MIN_POINT_DISTANCE: compute_min_point_distance,
}


@dataclass
class CompleteSubDesignAssessment:
    point_count: int
    active_dimensions: List[str]
    measurements: Dict[str, float]
    space_attributes: Set[str]


@dataclass
class DoeAssessment:
    total_point_count: int
    full_sub_set_assessments: List[CompleteSubDesignAssessment]
    measurements: Dict[str, float]


def assess(space: s.InputSpace, doe: DesignOfExperiment) -> DoeAssessment:
    """
    Assesses the experiment design for the given input space.

    Returns:
    DoeAssessment: results from an assessment for the whole design and sub-designs.
    """
    # determine every full-combination of input dimensions that could be defined in this space
    sub_spaces = space.derive_full_subspaces()

    # assign a id/int to each sub space
    sub_space_index_map = {}
    for i, sub_space in enumerate(sub_spaces):
        sub_space.sort()
        standardized_tuple_key = tuple(sub_space)
        sub_space_index_map[standardized_tuple_key] = i

    # determine the sub-space each data-point belongs to
    def map_point(point):
        active_dim_ids = []

        for dim_id, column_index in doe.input_set_map.items():
            if ~np.isnan(point[column_index]):
                active_dim_ids.append(dim_id)

        active_dim_ids.sort()
        return sub_space_index_map[tuple(active_dim_ids)]

    # compute the subspace each point belongs to
    mapped_values = [map_point(point) for point in doe.input_sets]

    # prepare data structures for returned assessment structure
    total_point_count = len(mapped_values)

    full_sub_set_assessments = []
    total_measurements = []

    ################################
    # full-sub-design metrics
    ################################

    for i, sub_space in enumerate(sub_spaces):
        point_row_mask = [v == i for v in mapped_values]
        sub_space_doe = doe.extract_points_and_dimensions(point_row_mask, sub_space)
        measurements = {}
        space_attributes = set()

        subspace_context = SubSpaceMetricComputeContext(space, doe, sub_space_doe)

        for m_id, compute in subspace_metric_computation_map.items():
            try:
                value = compute(subspace_context)
                measurements[m_id] = value
            except Exception as e:
                print(
                    f"WARNING: failed to compute metric {m_id} given this error {e}; skipping"
                )

        full_sub_set_assessments.append(
            CompleteSubDesignAssessment(
                point_count=sub_space_doe.point_count,
                active_dimensions=sub_space,
                measurements=measurements,
                space_attributes=space_attributes,
            )
        )

    ################################
    # DOE metrics
    ################################
    # TODO compute portion of complete sub-spaces sampled

    return DoeAssessment(
        total_point_count=total_point_count,
        full_sub_set_assessments=full_sub_set_assessments,
        measurements=total_measurements,
    )
