""" TODO Explain Module """

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from .dimensions import Dimension, Variant


def _generate_combinations(base_list: list) -> list:
    """
    TODO Explain the Function

    Arguments
    ---------
    base_list : list
        **Explanation**

    Returns
    -------
    all_combinations : list
        **Explanation**
    """
    # Create an empty list to store all combinations
    all_combinations = []

    # Iterate through all possible lengths of combinations
    for r in range(1, len(base_list) + 1):
        # Generate combinations of length r
        combinations = itertools.combinations(base_list, r)
        # Add the generated combinations to the list
        all_combinations.extend(combinations)

    return all_combinations


def derive_subspaces(level: Iterable[Dimension]) -> List[List[str]]:
    """
    TODO Explain the Function

    Arguments
    ---------
    level : Interable[Dimension]
        **Explanation**

    Returns
    -------
    condensed_lists : List[List[str]]
        **Explanation**
    """

    optional_dims = []
    expansion_map = {}
    required_dims = []

    for dim in level:
        if dim.nullable:
            if dim.has_child_dimensions():
                if isinstance(dim, Variant):
                    children_subspaces = []
                    for o in dim.children:
                        children_subspaces += derive_subspaces(
                            create_level_iterable([o])
                        )
                else:
                    children_subspaces = derive_subspaces(
                        create_level_iterable(dim.children)
                    )
                expansion_map[dim.local_id] = children_subspaces
                optional_dims.append(dim.local_id)
            else:
                optional_dims.append(dim.local_id)
        else:
            if isinstance(dim, Variant):
                optional_dims.append(dim.local_id)
                children_subspaces = []
                for o in dim.children:
                    children_subspaces += derive_subspaces(
                        create_level_iterable([o])
                    )
                expansion_map[dim.local_id] = children_subspaces
            else:
                required_dims.append(dim.local_id)

    base_dim_combinations = _generate_combinations(optional_dims)

    # expand base-level combinations with children subspace combinations
    condensed_lists = []
    for base_dim_combination in base_dim_combinations:
        expanded_dim_list: List[List[str]] = []
        for base_dim_id in base_dim_combination:
            # if dim does not have children subspace combinations
            if base_dim_id not in expansion_map:
                # simply add to existing combinations
                if len(expanded_dim_list) > 0:
                    for b_list in expanded_dim_list:
                        b_list.append(base_dim_id)
                else:
                    expanded_dim_list.append([base_dim_id])
            else:
                # dim has children subspace combinations
                children_spaces = expansion_map[base_dim_id]
                base_list_len = len(expanded_dim_list)
                if base_list_len == 0:
                    # must create base combinations with child combination
                    for css in children_spaces:
                        expanded_dim_list.append([base_dim_id] + css)
                else:
                    # must mix each existing combination
                    # with each child combination
                    for css in children_spaces:
                        i = 0
                        while i < base_list_len:
                            expanded_dim_list.append(
                                expanded_dim_list[i] + [base_dim_id] + css
                            )
                            i += 1
                    # remove incomplete sets without any child combination
                    for _ in range(base_list_len):
                        expanded_dim_list.pop(0)
        # add required dims to every combination
        for dim_list in expanded_dim_list:
            condensed_lists.append(list(required_dims) + dim_list)

    condensed_lists.append(required_dims)

    return condensed_lists


def create_level_iterable(
    base_dimensions: List[Dimension],
) -> Iterable[Dimension]:
    """
    TODO Explain the Function

    Arguments
    ---------
    base_dimensions : List[Dimension]
        **Explanation**

    Returns
    -------
    resolved_dimension_list : Iterable[Dimension]
        **Explanation**

    """
    resolved_dimension_list: List[Dimension] = []

    dimension_stack = list(base_dimensions)

    while len(dimension_stack) > 0:
        dim1 = dimension_stack.pop(0)

        if dim1.has_child_dimensions():
            if dim1.only_supports_spec_structure():
                # insert children dimensions to stack
                # to be processed in same level
                for child_dim in reversed(dim1.children):
                    dimension_stack.insert(0, child_dim)
            else:
                resolved_dimension_list.append(dim1)
        else:
            resolved_dimension_list.append(dim1)

    return resolved_dimension_list


def create_all_iterable(
    base_dimensions: List[Dimension],
) -> Iterable[Dimension]:
    """
    TODO Explain the Function

    Arguments
    ---------
    base_dimensions : List[Dimension]
        **Explanation**

    Returns
    -------
    resolved_dimension_list : Iterable[Dimension]
        **Explanation**
    """
    resolved_dimension_list: List[Dimension] = []

    dimension_stack = list(base_dimensions)

    while len(dimension_stack) > 0:
        dim1 = dimension_stack.pop(0)
        resolved_dimension_list.append(dim1)
        if dim1.has_child_dimensions():
            for child_dim in reversed(dim1.children):
                dimension_stack.insert(0, child_dim)

    return resolved_dimension_list


def _create_dict_from_flat_values(
    dimensions: List[Dimension], inputs, dim_to_index_mapping
) -> Dict:
    """
    TODO Explain the Function

    Arguments
    ---------
    dimensions : List[Dimension]
        **Explanation**
    inputs
        **Explanation**
    dim_to_index_mapping
        **Explanation**

    Returns
    -------
    dict_values : Dict
        **Explanation**
    """
    dict_values = {}
    for dim in dimensions:

        if dim.has_child_dimensions():
            input_is_null = False
            if dim.local_id in dim_to_index_mapping:
                # check to see if the input is marked as none
                dim_index = dim_to_index_mapping[dim.local_id]
                if inputs[dim_index] is None:
                    input_is_null = True

            if not input_is_null:
                children_dict = _create_dict_from_flat_values(
                    dim.children, inputs, dim_to_index_mapping
                )
                if len(children_dict) > 0:
                    dict_values[dim.local_id] = children_dict
        else:
            if dim.local_id in dim_to_index_mapping:
                dim_index = dim_to_index_mapping[dim.local_id]
                if np.isnan(inputs[dim_index]):
                    dict_values[dim.local_id] = None
                else:
                    dict_values[dim.local_id] = inputs[dim_index]

    return dict_values


@dataclass
class SubSpace:
    """
    TODO Explain Class
    """

    active_dimensions: Dict
    target_sample_count = 0

    pass


def _project_null(x1, x2):
    """
    TODO Explain the Function

    Arguments
    ---------
    x1
        **Explanation**
    x2
        **Explanation**

    Returns
    -------
    np.nan
        **Explanation**

    """
    return [(np.nan if np.isnan(xp2) else xp1) for xp1, xp2 in zip(x1, x2)]


@dataclass
class Space:
    """
    TODO Explain Class

    """

    dimensions: List[Dimension]

    @property
    def children(self) -> List[Dimension]:
        """
        TODO Explain the Function

        Arguments
        ---------
        self : Space
            **Explanation**

        Returns
        -------
        self.dimension : List[Dimension]
            **Explanation**
        """
        return self.dimensions

    def create_dim_map(self) -> Dict[str, Dimension]:
        """
        TODO Explain the Function

        Arguments
        ---------
        self : Space
            **Explanation**

        Returns
        -------
        dim_map : Dict[str, Dimension]
            **Explanation**
        """
        dim_map = {}

        for dim in create_all_iterable(self.dimensions):
            dim_map[dim.local_id] = dim

        return dim_map

    def derive_full_subspaces(self) -> List[List[str]]:
        """
        Discovers every combination of dimensions that are able to
        be specified togehter.
        We call these the possible full subspaces.
        A list is returned containing List of the dimensions'
        global identifer.
        Children dimensions are also analyzed.

        Returns
        -------
        derive_subspaces : List[List[str]]
            Every combination of dimensions that must be
            specified together
        """
        return derive_subspaces(create_level_iterable(self.children))

    def derive_spanning_subspaces(self) -> List[List[str]]:
        """
        Discovers every combination of dimensions that are always
        specified together. We call these the spanning subspaces.

        Returns
        -------
        spanning_subspaces : List[List[str]]
            Every combination of dimensions that must be
            specified together
        """
        subspaces = self.derive_full_subspaces()
        spanning_subspaces = {}

        dims = create_all_iterable(self.dimensions)
        for dim in dims:
            subspaces_key = ""
            for i, subspace in enumerate(subspaces):
                if dim.local_id in subspace:
                    subspaces_key += f"{i}_"
            if subspaces_key != "":
                if subspaces_key not in spanning_subspaces:
                    spanning_subspaces[subspaces_key] = []
                spanning_subspaces[subspaces_key].append(dim.id)

        return list(spanning_subspaces.values())

    def count_dimensions(self) -> int:
        """
        Counts and returns the dimensions of the space
        that are not only for structure. TODO Look at wording
        """
        return sum(
            [
                (
                    1
                    if not d.has_child_dimensions()
                    else (
                        (0 if d.only_supports_spec_structure() else 1)
                        + (d.count_children_dimensions())
                    )
                )
                for d in self.dimensions
            ]
        )

    def convert_flat_values_to_dict(
        self, input_sets, dim_index_mapping: Dict[str, int]
    ):
        """
        TODO Explain the Function

        Arguments
        ---------
        self : Space
            **Explanation**
        input_sets
            **Explanation**
        dim_index_mapping : Dict[str, int]
            **Explanation**

        Returns
        -------
        value_dicts : List
            **Explanation**
        """
        value_dicts = []

        for inputs in input_sets:
            value_map = _create_dict_from_flat_values(
                self.dimensions, inputs, dim_index_mapping
            )

            value_dicts.append(value_map)

        return value_dicts

    def decode_zero_one_matrix(
        self,
        x: np.array,
        dim_column_map: Dict[str, int],
        map_null_to_children_dim=False,
        utilize_null_portitions=True,
    ):
        """
        TODO Explain the Function

        Arguments
        ---------
        self : Space
            **Explanation**
        x : np.array
            **Explanation**
        dim_column_map : Dict[str, int]
            **Explanation**
        map_null_to_children_dim=False
            **Explanation**
        utilize_null_portitions=True
            **Explanation**

        Returns
        -------
        decoded_values : ndarray
            **Explanation**

        """

        decoded_values = np.array(x)
        flatted_dimensions = create_all_iterable(self.children)
        for dim in flatted_dimensions:

            if dim.id in dim_column_map:
                column_index = dim_column_map[dim.id]
                encoded_column = x[:, column_index]
                decoded_column = dim.collapse_uniform(
                    encoded_column, utilize_null_portitions
                )
                if map_null_to_children_dim and dim.has_child_dimensions():
                    nan_mask = np.isnan(decoded_column)
                    if nan_mask.sum() > 0:
                        children = create_all_iterable(dim.children)
                        non_nan_mask = ~nan_mask
                        for c_dim in children:
                            if c_dim.id in dim_column_map:
                                c_column_index = dim_column_map[c_dim.id]
                                x[:, c_column_index][non_nan_mask] = x[
                                    :, c_column_index
                                ][non_nan_mask]
                                x[:, c_column_index][nan_mask] = np.nan
                    if isinstance(dim, Variant):
                        # determine active child, null out others
                        for i, c_dim in enumerate(dim.children):
                            nan_mask = [x != i for x in decoded_column]

                            for c_c_dim in create_all_iterable([c_dim]):
                                if c_c_dim.id in dim_column_map:
                                    c_column_index = dim_column_map[c_c_dim.id]
                                    x[:, c_column_index][nan_mask] = np.nan

                decoded_values[:, column_index] = decoded_column
        return decoded_values


@dataclass
class InputSpace(Space):
    """
    TODO Explain Class
    """

    multi_dim_contraints: Optional[List] = None


@dataclass
class OutputSpace(Space):
    """
    TODO Explain Class
    """

    dimensions: List[Dimension]
