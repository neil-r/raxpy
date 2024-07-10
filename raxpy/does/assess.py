"""
    This modules provides logic to compute assessments of an experiment design.
"""

from dataclasses import dataclass
from typing import List, Dict, Set

import numpy as np
from scipy.stats.qmc import discrepancy

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

METRIC_MAX_MIN_POINT_DISTANCE = "max_min_point_distance"


def compute_max_min_point_distance(context: SubSpaceMetricComputeContext) -> float:
    """
    Computes and returns the maximum of the minimum-point-distance-for-each-point

    Returns:
    float: the maximum of the minimum-point-distance-for-each-point
    """
    # TODO: compute the distances for each point combination
    # TODO: find the min for each point
    # TODO: take the max from all of these min distances
    return 0.0


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
    METRIC_MAX_MIN_POINT_DISTANCE: compute_max_min_point_distance,
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
    sub_spaces = space.derive_subspaces()

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
