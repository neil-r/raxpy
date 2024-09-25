from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np

from .doe import DesignOfExperiment


@dataclass
class TargetTracker:
    fullsubspace_ids: Tuple[str]
    count: int = 0
    point_idxs: List = []

    def add_point(self, point_index):
        self.count += 1
        self.point_idxs(point_index)


def adjust_to_meet_targets(doe: DesignOfExperiment, targets: Dict[Tuple, int]):
    # determine actual
    actual_counts = {key: TargetTracker(fullsubspace_ids=key) for key in targets.keys()}

    # determine the sub-space each data-point belongs to
    def map_point(point_index, point):
        active_dim_ids = []

        for dim_id, column_index in doe.input_set_map.items():
            if ~np.isnan(point[column_index]):
                active_dim_ids.append(dim_id)

        active_dim_ids.sort()
        actual_counts[tuple(a for a in active_dim_ids)].add_point(point_index)

    # compute the subspace each point belongs to
    for i, point in enumerate(doe.input_sets):
        map_point(i, point)

    overs:List[TargetTracker] = []
    unders:List[TargetTracker] = []

    for key in targets.keys():
        if targets[key][1] > actual_counts[key].count:
            unders.append(actual_counts[key])
        elif targets[key][1] < actual_counts[key].count:
            overs.append(actual_counts[key])

    dim_value_map: Dict[str,List[float]] = {}
    value_supplier: Dict[str, List[TargetTracker]] = {}

    for over in overs:
        for dim_id in over.fullsubspace_ids:
            if dim_id not in value_supplier:
                value_supplier[dim_id] = []    
            value_supplier[dim_id].append(over)

    for under_



    for over in overs:
        offset = targets[over.fullsubspace_ids] - over.count
        while offset > 0:
            # find point to move
            for distance_away in range(1,4):

                for under in unders:
                    # check if candiate move
            
`

    return True
