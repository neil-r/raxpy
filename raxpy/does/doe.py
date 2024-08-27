"""
    This module defines the data structures
    used to represent designs of experiments.
"""

from typing import Dict, List, Literal, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..spaces.root import InputSpace


class EncodingEnum(str, Enum):
    ZERO_ONE_RAW_ENCODING = "0-1-raw"
    ZERO_ONE_NULL_ENCODING = "0-1-null"
    NONE = "decoded"


Encoding = Literal[
    EncodingEnum.ZERO_ONE_RAW_ENCODING,
    EncodingEnum.ZERO_ONE_NULL_ENCODING,
    EncodingEnum.NONE,
]


@dataclass
class DesignOfExperiment:
    """
    A class used to represent a design of an experiment.
    """

    input_space: InputSpace
    input_sets: np.array
    input_set_map: Dict[str, int]
    encoding: Encoding

    _decoded_cache: Optional[np.array] = None

    def __post_init__(self):
        """
        Post-initialization processing to validate
        the fields of the dataclass.

        Raises
        ------
        ValueError
            If the input_sets column count does not match the number of
            dimension mappings.
        """
        if self.dim_specification_count != len(self.input_set_map):
            raise ValueError(
                f"Invalid inputs: number of columns of input sets, "
                f"{self.dim_specification_count}, does not match the number "
                f"of dimension id mappings provided, "
                f"{len(self.input_set_map)}"
            )
        # ensure no duplicate index specifications
        reverse_mapping = {}
        highest_column_index = self.dim_specification_count - 1
        for dim_id, dim_index in self.input_set_map.items():
            if dim_index in reverse_mapping:
                raise ValueError(
                    "Duplicate column indexes specified in input set map"
                )
            else:
                if dim_index >= 0 and dim_index <= highest_column_index:
                    reverse_mapping[dim_index] = dim_id
                else:
                    raise ValueError(
                        f"Invalid column index, "
                        f"out-of-bounds: {dim_id}:{dim_index}"
                    )

    @property
    def decoded_input_sets(self):
        if self._decoded_cache is None:
            if self.encoding == EncodingEnum.ZERO_ONE_NULL_ENCODING:
                self._decoded_cache = self.input_space.decode_zero_one_matrix(
                    self.input_sets,
                    self.input_set_map,
                    map_null_to_children_dim=False,
                    utilize_null_portitions=False,
                )
            elif self.encoding == EncodingEnum.ZERO_ONE_RAW_ENCODING:
                self._decoded_cache = self.input_space.decode_zero_one_matrix(
                    self.input_sets,
                    self.input_set_map,
                    map_null_to_children_dim=True,
                    utilize_null_portitions=True,
                )
            else:
                self._decoded_cache = self.input_sets

        return self._decoded_cache

    def extract_points_and_dimensions(
        self, point_row_mask, dim_set: List[str]
    ) -> "DesignOfExperiment":
        """
        Extracts a sub-design given a row mask and a subset of dimensions

        Arguments
        ---------
        self
            the parent design
        point_row_mask
            the row mask, true if the row should be included in extracted design
        dim_set : List[str]
            the id of the columns that should be included in the extracted design

        Returns
        -------
        DesignOfExperiment
            a sub design given the row mask and the column list
        """

        column_indexes = [self.input_set_map[dim_id] for dim_id in dim_set]

        return DesignOfExperiment(
            input_space=self.input_space,
            input_sets=self.input_sets[point_row_mask][:, column_indexes],
            input_set_map={dim_id: i for i, dim_id in enumerate(dim_set)},
            encoding=self.encoding,
        )

    @property
    def point_count(self) -> int:
        """
        Provides the number of points/rows
        within the experiment design.

        Returns
        -------
        np.size : int
            The count of points/rows the design provides values.
        """
        return np.size(self.input_sets, axis=0)

    @property
    def dim_specification_count(self) -> int:
        """
        Provides the number of dimensions/columns within
        the experiment design.

        Returns
        -------
        np.size : int
            The count of dimensions/columns the design provides values.
        """
        return np.size(self.input_sets, axis=1)
