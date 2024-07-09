from dataclasses import dataclass
from typing import List, Dict, Set
import numpy as np

from .doe import DesignOfExperiment
from ..spaces import root as s

# DOE Metrics

METRIC_PORTION_SUBSPACES_INCLUDED = "portion_of_subspaces_included"

METRIC_TARGET_PORTION_OFFSET = "target_portion_offset"

METRIC_AVG_AVG_PORTION_OF_SUBSPACE_LEVELS = "avg_avg_subspace_levels"

# Complete SubDesign Metrics
METRIC_AVG_PORTION_LEVELS_INCLUDED = "avg_portion_of_levels_included"

METRIC_PORTION_OF_TOTAL = "portion_of_total"

METRIC_DISCREPANCY = "discrepancy"

doe_metric_computation_map = {

}

subspace_metric_computation_map = {
  
}

@dataclass
class CompleteSubDesignAssessment:
  point_count:int
  active_dimensions:List[str]
  measurements:Dict[str, float]
  space_attributes:Set[str]


@dataclass
class DoeAssessment:
  total_point_count:int
  full_sub_set_assessments:List[CompleteSubDesignAssessment]
  measurements:Dict[str, float]
  pass


def assess(space:s.InputSpace, doe:DesignOfExperiment) -> DoeAssessment:

  # determine every full-combination of input dimensions that could be defined in this space
  
  sub_spaces = space.derive_subspaces()

  sub_space_index_map = {}
  for i, sub_space in enumerate(sub_spaces):
    sub_space.sort()
    standardized_tuple_key = tuple(sub_space)
    sub_space_index_map[standardized_tuple_key] = i

  def map_point(point):
    active_dim_ids = []
    
    for dim_id, column_index in doe.input_set_map.items():
      if ~np.isnan(point[column_index]):
        active_dim_ids.append(dim_id)

    active_dim_ids.sort()
    return sub_space_index_map[tuple(active_dim_ids)]

  # determine the sub-space each data-point belongs to
  mapped_values = [map_point(point) for point in doe.input_sets]

  total_point_count = len(mapped_values)
  full_sub_set_assessments = []
  total_measurements = []

  ################################
  # full-sub-design metrics
  ################################
  for i, sub_space in enumerate(sub_spaces):
    measurements = {}
    sub_space_point_count = sum(v == i for v in mapped_values)
    space_attributes = set()

    # TODO compute max min point distance

    # TODO compute discrepancy

    # TODO compute correlation

    # TODO compute portion of levels included

    full_sub_set_assessments.append(
      CompleteSubDesignAssessment(
        point_count=sub_space_point_count,
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