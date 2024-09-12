"""
    This modules provides logic to compute
    assessments of an experiment design.
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
import math
import itertools

import numpy as np
from scipy.stats.qmc import discrepancy
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from .doe import DesignOfExperiment, Encoding, EncodingEnum
from ..spaces import root as s
from ..spaces import dimensions


@dataclass
class SubSpaceMetricComputeContext:
    """
    A class used to gather the values needed to compute metrics for a
    subspace design.
    """

    whole_doe: DesignOfExperiment
    sub_space_doe: DesignOfExperiment


# Whole DOE Metrics
METRIC_PORTION_SUBSPACES_INCLUDED = "portion_of_subspaces_included"

METRIC_TARGET_PORTION_OFFSET = "target_portion_offset"

METRIC_AVG_AVG_PORTION_OF_SUBSPACE_LEVELS = "avg_avg_subspace_levels"

METRIC_WEIGHTED_DISCREPANCY = "weighted_discrepancy"

METRIC_WEIGHTED_MIPD = "weighted_mipd"

METRIC_WEIGHTED_MST_STATS = "weighted_mst_stats"

METRIC_WHOLE_MIN_POINT_DISTANCE = "max_whole_min_point_distance"

METRIC_WHOLE_MIN_PROJECTED_DISTANCE = "max_whole_min_projected_distance"

METRIC_WHOLE_STAR_DISCREPANCY = "star_discrepancy"

# FullSubDesign Metrics
METRIC_AVG_PORTION_LEVELS_INCLUDED = "avg_portion_of_levels_included"

METRIC_PORTION_OF_TOTAL = "portion_of_total"

METRIC_DISCREPANCY = "discrepancy"

METRIC_MIN_POINT_DISTANCE = "max_min_point_distance"


def determine_relevant_dimensions(
    design: DesignOfExperiment, point
) -> List[str]:
    relevant_dims = []

    children_sets = [design.input_space.children]
    while len(children_sets) > 0:
        children_set = children_sets.pop(0)
        for dim in s.create_level_iterable(children_set):
            if dim.id in design.input_set_map:
                if isinstance(dim, dimensions.Variant):
                    raise NotImplementedError(
                        "Relevant dimensions given Variant not implemented"
                    )
                d_index = design.input_set_map[dim.id]
                relevant_dims.append(dim.id)

                if (
                    dim.has_child_dimensions()
                    and not np.isnan(point[d_index])
                    and point[d_index] > 0.0
                ):
                    children_sets.append(dim.children)

    return relevant_dims


def compute_star_discrepancy(
    design: DesignOfExperiment,
    p=np.inf,
    encoding=EncodingEnum.ZERO_ONE_NULL_ENCODING,
) -> float:

    x = design.get_data_points(encoding)

    # determine projections for each releval dimension set
    relevant_dim_sets_proj_map = {}
    point_relevant_dims = []

    # determine which heirachiral dimensions are active for each point
    # determine which vairant dimensions are active for each point
    for point in x:
        # determine relevant dimensions for point
        relevant_dims_set = tuple(determine_relevant_dimensions(design, point))

        if relevant_dims_set not in relevant_dim_sets_proj_map:
            projection_sets = []
            l = len(relevant_dims_set)
            # determine projects to consider for this set of dimensions
            for i in range(l):
                rhs_dim_id = relevant_dims_set[i]
                projection_sets.append((rhs_dim_id,))
                for j in range(i + 1, l):
                    lhs_dim_id = relevant_dims_set[j]
                    projection_sets.append((rhs_dim_id, lhs_dim_id))

            relevant_dim_sets_proj_map[relevant_dims_set] = projection_sets

        point_relevant_dims.append(relevant_dims_set)

    local_discrepancies = []
    n = design.point_count
    dim_map = design.input_space.create_dim_map()
    # use each point for region discrepancy sample
    for relevant_dims_set, point in zip(point_relevant_dims, x):

        projection_sets = relevant_dim_sets_proj_map[relevant_dims_set]

        point_projection_discrepancies = []
        for u in projection_sets:
            col_indexes = list(design.input_set_map[dim_id] for dim_id in u)
            point_count_in_projection_region = 0
            for c_relevant_dims_set, c_point in zip(point_relevant_dims, x):
                relevant_point = True
                in_region = True
                for col_index, dim_id in zip(col_indexes, u):
                    if dim_id not in c_relevant_dims_set:
                        relevant_point = False
                        break
                    else:
                        if np.isnan(point[col_index]):
                            if np.isnan(c_point[col_index]):
                                pass
                            else:
                                in_region = False
                                break
                        else:
                            if np.isnan(c_point[col_index]):
                                pass
                            elif c_point[col_index] > point[col_index]:
                                in_region = False
                                break
                if relevant_point and in_region:
                    point_count_in_projection_region += 1

            region_volumn_percent = 1.0
            for col_index, dim_id in zip(col_indexes, u):
                dim = dim_map[dim_id]
                # compute dimension's culmative distribution for point value
                v = point[col_index]
                if np.isnan(v):
                    region_volumn_percent *= dim.portion_null
                else:
                    region_volumn_percent *= (
                        dim.portion_null + (1 - dim.portion_null) * v
                    )

            portion_of_points_in_region = point_count_in_projection_region / n

            ppd = abs(portion_of_points_in_region - region_volumn_percent)

            point_projection_discrepancies.append(ppd)

        if p == np.inf:
            local_discrepancies.append(max(point_projection_discrepancies))
        else:
            local_discrepancies.append(
                sum(ppd ** (p) for ppd in point_projection_discrepancies)
            )

    if p == np.inf:
        return max(local_discrepancies)
    else:
        return sum(local_discrepancies) ** (1 / p)


def compute_min_point_distance(context: SubSpaceMetricComputeContext) -> float:
    """
    Computes and returns the minimum-interpoint-distance (MIPD)
    among every pair of points

    Arguments
    ---------
    context : SubSpaceMetricComputeContext
        **Explanation**

    Returns
    -------
    np.min(dm) : float
        The minimum-interpoint-distance
    """
    if context.sub_space_doe.point_count <= 1:
        raise ValueError("Not enough points to compute min point distance")
    points = context.sub_space_doe.input_sets
    # compute the distances for each point combination
    dm = distance_matrix(points, points)
    # the diagonals will be zero since a point's distance to itself is 0,
    # we don't care about these distances
    np.fill_diagonal(dm, np.inf)

    # find the min distances to the each other point
    return np.min(dm)


def compute_average_reciprocal_distance_projection(
    context: SubSpaceMetricComputeContext, lambda_hp=2, z_hp=2
) -> float:
    """
    Implementation of the Average reciprocal distance projection
    metric as denoted in: Draguljić, Santner, and Dean, “Noncollapsing
    Space-Filling Designs for Bounded Nonrectangular Regions.”

    Arguments
    ---------
    context : SubSpaceMetricComputeContext
        **Explanation**
    lambda_hp=2 : int
        **Explanation**
    z_hp=2 : int
        **Explanation**

    Returns
    -------
    TODO **Explanation**
    """
    if context.sub_space_doe.point_count <= 1:
        raise ValueError("Not enough points to compute ard")
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

            x_projection = context.sub_space_doe.input_sets[
                :, index_combination
            ]

            dm = distance_matrix(x_projection, x_projection, p=z_hp)
            # fill diagonal to avoid divide by zero
            # The equation does not need the diagonal elements anyway
            np.fill_diagonal(dm, 1)
            reciprocal_distances = (max_j_distance_m / dm) ** lambda_hp

            # sum the upper triangle matrix elements,
            # excluding the diagonal elements
            reciprocal_distances_sum = np.sum(
                np.tril(reciprocal_distances, k=1)
            )

            running_sum += reciprocal_distances_sum
            comb_count += 1

    return (running_sum / (math.comb(n, 2) * comb_count)) ** (1.0 / lambda_hp)


def compute_mst_stats(
    context: SubSpaceMetricComputeContext,
) -> Tuple[float, float]:
    """
    Computes and returns the mean and standard deviation of the edge-values of
    a minimum spanning tree (MST) of the design points. The edge-values
    represent the distances between design points.

    For more information see https://doi.org/10.1016/j.chemolab.2009.03.011

    Arguments
    ---------
    context : SubSpaceMetricComputeContext
        **Explanation**

    Returns
    -------
    mst_mean, mst_std : Tuple[float, float]
        a tuple of the mean and standard deviation of the MST edges
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
    """
    TODO Explain the Function

    Arguments
    ---------
    context : SubSpaceMetricComputeContext
        **Explanation**

    Returns
    -------
    discrepancy : float
        **Explanation**

    """
    return discrepancy(context.sub_space_doe.input_sets, method="MD")


def compute_portion_of_total(context: SubSpaceMetricComputeContext) -> float:
    """
    TODO Explain the Function

    Arguments
    ---------
    context : SubSpaceMetricComputeContext
        **Explanation**

    Returns
    -------
    Portion of whole DoE as a float

    """
    return context.sub_space_doe.point_count / context.whole_doe.point_count


def compute_avg_portion_of_levels(
    context: SubSpaceMetricComputeContext,
) -> float:
    """
    TODO Explain the Function

    Arguments
    ---------
    context : SubSpaceMetricComputeContext
        **Explanation**

    Returns
    -------
    Not Implemented

    """
    pass


subspace_metric_computation_map = {
    METRIC_DISCREPANCY: compute_discrepancy,
    METRIC_PORTION_OF_TOTAL: compute_portion_of_total,
    METRIC_AVG_PORTION_LEVELS_INCLUDED: compute_avg_portion_of_levels,
    METRIC_MIN_POINT_DISTANCE: compute_min_point_distance,
}


@dataclass
class FullSubDesignAssessment:
    """
    TODO Explain Class
    """

    point_count: int
    active_dimensions: List[str]
    measurements: Dict[str, float]
    space_attributes: Set[str]

    def compare_dimensions(self, dim_list) -> bool:
        return len(dim_list) == len(self.active_dimensions) and all(
            d in self.active_dimensions for d in dim_list
        )


@dataclass
class DoeAssessment:
    """
    TODO Explain Class
    """

    total_point_count: int
    full_sub_set_assessments: List[FullSubDesignAssessment]
    measurements: Dict[str, float]

    def get_full_sub_design_assessment(
        self, active_dimensions: List[str]
    ) -> Optional[FullSubDesignAssessment]:
        for assessment in self.full_sub_set_assessments:
            if assessment.compare_dimensions(active_dimensions):
                return assessment
        return None


def compute_whole_design_star_discrepancy(
    design: DesignOfExperiment,
    _: List[FullSubDesignAssessment],
    encoding: Encoding,
) -> float:
    if encoding == EncodingEnum.ZERO_ONE_RAW_ENCODING:
        encoding = EncodingEnum.ZERO_ONE_NULL_ENCODING
    return compute_star_discrepancy(design, encoding=encoding)


def compute_weighted_discrepancy(
    _doe: DesignOfExperiment,
    full_design_assessments: List[FullSubDesignAssessment],
    _encoding: Encoding,
    weighting_metric=METRIC_PORTION_OF_TOTAL,
) -> float:
    """
    TODO Explain the Function

    Arguments
    ---------
    doe : DesignOfExperiment
        **Explanation**
    full_design_assessments : List[CompleteSubDesignAssessment]
        **Explanation**
    _encoding: Encoding
    weighting_metric=METRIC_PORTION_OF_TOTAL
        **Explanation**

    Returns
    -------
        **Explanation**

    """
    discrepancies = []
    weights = []
    for full_design in full_design_assessments:
        if METRIC_DISCREPANCY in full_design.measurements:
            discrepancies.append(full_design.measurements[METRIC_DISCREPANCY])
            weights.append(full_design.measurements[weighting_metric])

    weight_sum = sum(weights)

    return sum(d * w / weight_sum for d, w in zip(discrepancies, weights))


def compute_weighted_mipd(
    _doe: DesignOfExperiment,
    full_design_assessments: List[FullSubDesignAssessment],
    _encoding: Encoding,
    weighting_metric=METRIC_PORTION_OF_TOTAL,
) -> float:
    """
    TODO Explain the Function

    Arguments
    ---------
    doe : DesignOfExperiment
        **Explanation**
    full_design_assessments : List[CompleteSubDesignAssessment]
        **Explanation**
    _: Encoding
    weighting_metric=METRIC_PORTION_OF_TOTAL
        **Explanation**

    Returns
    -------
        **Explanation**

    """
    discrepancies = []
    weights = []
    for full_design in full_design_assessments:
        if METRIC_MIN_POINT_DISTANCE in full_design.measurements:
            discrepancies.append(
                full_design.measurements[METRIC_MIN_POINT_DISTANCE]
            )
            weights.append(full_design.measurements[weighting_metric])

    weight_sum = sum(weights)

    return sum(d * w / weight_sum for d, w in zip(discrepancies, weights))


def _compute_nan_distance(row1, row2, p=2):
    """
    Computes the p-norm distance between two rows,
    For comparisons where one point has a nan value for a
    dimension comparison, the distance is projected to 1 for that dimension.
    For comparisons where both points have a nan value for a dimension
    comparison, the distance is projected to 0 for that dimension.

    Arguments
    ---------
    row1 : np.array
        a row to consider for the distance computation
    row2 : np.array
        another row to consider for the distance computation
    p:int=2
        Which Minkowski p-norm to use in the distance computation

    Returns
    -------
        The distance between the two rows
    """
    # Identify which elements are not NaN in both rows
    mask = ~np.isnan(row1) | ~np.isnan(row2)

    # If no valid comparisons, return a distance of 0
    if not np.any(mask):
        return 0.0

    # Compute Euclidean distance only for non-NaN elements for both elements
    parts = (np.abs(row1[mask] - row2[mask])) ** p

    # Replace nan's differences to 1 to represent the maximin difference
    parts = np.nan_to_num(parts, nan=1)

    distance = np.sqrt(np.sum(parts))

    return distance


def _compute_nan_distance_matrix(matrix: np.array, p=2):
    """
    Computes the pairwise distances between all rows in a matrix.
    For comparisons where one point has a nan value for a
    dimension comparison, the distance is projected to 1 for that dimension.
    For comparisons where both points have a nan value for a dimension
    comparison, the distance is projected to 0 for that dimension.

    Arguments
    ---------
    matrix : np.array
        The design to assess
    p:int=2
        Which Minkowski p-norm to use in the distance computation

    Returns
    -------
        The distance matrix for the rows in the matrix
    """
    num_rows = matrix.shape[0]
    dm = np.zeros((num_rows, num_rows))

    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            dist = _compute_nan_distance(matrix[i], matrix[j], p=p)
            dm[i, j] = dist
            dm[j, i] = dist

    return dm


def compute_whole_min_point_distance(
    doe: DesignOfExperiment,
    _: List[FullSubDesignAssessment],
    encoding: Encoding,
    p: int = 2,
) -> float:
    """
    Computes the minimum distance between two points in the DOE with nan
    projections. For comparisons where one point has a nan value for a
    dimension comparison, the distance is projected to 1 for that dimension.
    For comparisons where both points have a nan value for a dimension
    comparison, the distance is projected to 0 for that dimension.

    Arguments
    ---------
    doe : DesignOfExperiment
        The design to assess
    _ : List[CompleteSubDesignAssessment]
        The sub-design assessments (not-used, specified to support common
        metric computation interface)
    encoding: Encoding

    p:int=2
        Which Minkowski p-norm to use in the distance computation

    Returns
    -------
        A float value representing the minimum distance
    """
    dm = _compute_nan_distance_matrix(doe.get_data_points(encoding), p=p)
    np.fill_diagonal(dm, np.inf)
    return np.min(dm)


def compute_min_projected_distance(
    doe: DesignOfExperiment,
    _: List[FullSubDesignAssessment],
    encoding: Encoding,
) -> float:
    """
    Computes the minimum projected distance between any two values within the
    same a dimension, ignoring distances of values with that are nan

    Arguments
    ---------
    doe : DesignOfExperiment
        The design to assess
    _ : List[CompleteSubDesignAssessment]
        The sub-design assessments (not-used, specified to support common
        metric computation interface)
    encoding: Encoding
    Returns
    -------
        A float value representing the minimum projected distance
    """
    data_points = doe.get_data_points(encoding)
    dim_map = doe.input_space.create_dim_map()
    min_so_far = np.inf
    for dim_id, column_i in doe.input_set_map.items():
        dim = dim_map[dim_id]
        # ignore dimensions that have children dimensions since the projected
        # should not apply to these dimensions
        if dim.has_child_dimensions():
            continue

        for row_1i in range(doe.point_count):
            lhs = data_points[row_1i][column_i]
            for row_2i in range(row_1i + 1, doe.point_count):
                d = np.abs(lhs - data_points[row_2i][column_i])
                if d != np.nan and d < min_so_far:
                    min_so_far = d

    return min_so_far


# the following map is used in the assess function
# to discover the metrics to compute
doe_metric_computation_map = {
    METRIC_WEIGHTED_DISCREPANCY: compute_weighted_discrepancy,
    METRIC_WEIGHTED_MIPD: compute_weighted_mipd,
    METRIC_WHOLE_MIN_POINT_DISTANCE: compute_whole_min_point_distance,
    METRIC_WHOLE_MIN_PROJECTED_DISTANCE: compute_min_projected_distance,
    METRIC_WHOLE_STAR_DISCREPANCY: compute_whole_design_star_discrepancy,
}


def assess_with_all_metrics(
    doe: DesignOfExperiment,
    encoding: Encoding = EncodingEnum.ZERO_ONE_NULL_ENCODING,
) -> DoeAssessment:
    """
    Assesses the experiment design for the given input space.

    Arguments
    ---------
    doe : DesignOfExperiment
        **Explanation**
    encoding: Encoding

    Returns
    -------
    DoeAssessment
        Results from an assessment for the whole design and sub-designs.
    """
    # determine every full-combination of input dimensions
    # that could be defined in this space
    sub_spaces = doe.input_space.derive_full_subspaces()

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
    mapped_values = [map_point(point) for point in doe.decoded_input_sets]

    # prepare data structures for returned assessment structure
    total_point_count = len(mapped_values)

    full_sub_set_assessments: List[FullSubDesignAssessment] = []

    ################################
    # full-sub-design metrics
    ################################

    for i, sub_space in enumerate(sub_spaces):
        point_row_mask = [v == i for v in mapped_values]
        sub_space_doe = doe.extract_points_and_dimensions(
            point_row_mask, sub_space, encoding
        )
        measurements = {}
        space_attributes = set()

        subspace_context = SubSpaceMetricComputeContext(doe, sub_space_doe)

        for m_id, compute in subspace_metric_computation_map.items():
            try:
                value = compute(subspace_context)
                if isinstance(value, Tuple):
                    for i, v in enumerate(value):
                        measurements[f"{m_id}-{i}"] = v
                else:
                    measurements[m_id] = value
            except Exception as e:
                print(
                    f"WARNING: failed to compute metric {m_id} "
                    f"given this error {e}; skipping"
                )

        full_sub_set_assessments.append(
            FullSubDesignAssessment(
                point_count=sub_space_doe.point_count,
                active_dimensions=sub_space,
                measurements=measurements,
                space_attributes=space_attributes,
            )
        )

    ################################
    # DOE metrics
    ################################
    total_measurements = {}

    for m_id, compute in doe_metric_computation_map.items():

        value = compute(doe, full_sub_set_assessments, encoding)
        if isinstance(value, Tuple):
            for i, v in enumerate(value):
                total_measurements[f"{m_id}-{i}"] = v
        else:
            total_measurements[m_id] = value

    return DoeAssessment(
        total_point_count=total_point_count,
        full_sub_set_assessments=full_sub_set_assessments,
        measurements=total_measurements,
    )
