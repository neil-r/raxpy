"""
    This module defines the data structures used to represent designs of experiments.
"""

from typing import Dict, List
from dataclasses import dataclass

import numpy as np

from ..spaces.root import InputSpace


@dataclass
class DesignOfExperiment:
    """
    A class used to represent a design of an experiment.
    """

    input_space: InputSpace
    input_sets: np.array
    input_set_map: Dict[str, int]
    encoded_flag: bool = False

    def __post_init__(self):
        """
        Post-initialization processing to validate the fields of the dataclass.

        Raises
        ------
        ValueError
            If the input_sets column count does not match the number of
            dimension mappings.
        """
        if self.dim_specification_count != len(self.input_set_map):
            raise ValueError(
                "Invalid inputs: number of columns of input sets, "
                + f"{self.dim_specification_count}, does not match the number"
                + " of dimension id mappings provided, {len(self.input_set_map)}"
            )
        # ensure no duplicate index specifications
        reverse_mapping = {}
        highest_column_index = self.dim_specification_count - 1
        for dim_id, dim_index in self.input_set_map.items():
            if dim_index in reverse_mapping:
                raise ValueError("Duplicate column indexes specified in input set map")
            else:
                if dim_index >= 0 and dim_index <= highest_column_index:
                    reverse_mapping[dim_index] = dim_id
                else:
                    raise ValueError(
                        f"Invalid column index, out-of-bounds: {dim_id}:{dim_index}"
                    )

    def extract_points_and_dimensions(
        self, point_row_mask, dim_set: List[str]
    ) -> "DesignOfExperiment":

        column_indexes = [self.input_set_map[dim_id] for dim_id in dim_set]

        return DesignOfExperiment(
            input_space=self.input_space,
            input_sets=self.input_sets[point_row_mask][:, column_indexes],
            input_set_map={dim_id: i for i, dim_id in enumerate(dim_set)},
            encoded_flag=self.encoded_flag,
        )

    @property
    def point_count(self) -> int:
        """
        Provides the number of points/rows within the experiment design.

        Returns:
        int: the count of points/rows the design provides values.
        """
        return np.size(self.input_sets, axis=0)

    @property
    def dim_specification_count(self) -> int:
        """
        Provides the number of dimensions/columns within the experiment design.

        Returns:
        int: the count of dimensions/columns the design provides values.
        """
        return np.size(self.input_sets, axis=1)
