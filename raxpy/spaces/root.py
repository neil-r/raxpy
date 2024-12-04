""" 
    This modules implements the data structures to represent 
    compositions of dimensions and common functions working with
    a list of dimensions.
"""

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from .dimensions import Dimension, Variant


def _generate_combinations(base_list: list) -> list:
    """
    Helper function to combinations of the elements in base_list

    Arguments
    ---------
    base_list : list
        the list of elements to consider

    Returns
    -------
    all_combinations : list
        the list of combinations
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
    Dervies every possible sub-spaces. Each sub-space defines a set
    of dimensions that corrospond to a valid set of non-null
    inputs.

    Arguments
    ---------
    level : Interable[Dimension]
        The dimensions to consider

    Returns
    -------
    condensed_lists : List[List[str]]
        a list of lists containing dimension's ids
    """

    optional_dims = []
    expansion_map = {}
    required_variant_expansion_map = {}
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
                expansion_map[dim.id] = children_subspaces
                optional_dims.append(dim.id)
            else:
                optional_dims.append(dim.id)
        else:
            if isinstance(dim, Variant):
                required_dims.append(dim.id)
                children_subspaces = []
                for o in dim.children:
                    children_subspaces += derive_subspaces(
                        create_level_iterable([o])
                    )
                required_variant_expansion_map[dim.id] = children_subspaces
            else:
                required_dims.append(dim.id)

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

    if len(required_variant_expansion_map) > 0:
        condensed_lists_expanded = []
        # expand required variants
        for (
            _v_dim_id,
            children_spaces,
        ) in required_variant_expansion_map.items():

            # must mix each existing combination
            # with each child combination
            base_list_len = len(condensed_lists)
            for css in children_spaces:
                i = 0
                while i < base_list_len:
                    condensed_lists_expanded.append(condensed_lists[i] + css)
                    i += 1
        condensed_lists = condensed_lists_expanded

    return condensed_lists


def create_level_iterable(
    base_dimensions: List[Dimension],
) -> Iterable[Dimension]:
    """
    Creates a iterable over base_dimensions, including
    the children dimensions of dimensions that are only
    used for structure.

    Arguments
    ---------
    base_dimensions : List[Dimension]
        the dimensions to consider

    Returns
    -------
    resolved_dimension_list : Iterable[Dimension]
        the dimensions at the same level
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
    base_dimensions: List[Dimension], skip_structual_dims=False
) -> Iterable[Dimension]:
    """
    Creates an Iterable over all the dimensions in
    base_dimensions and their children.

    Arguments
    ---------
    base_dimensions : List[Dimension]
        the dimensions to consider

    Returns
    -------
    resolved_dimension_list : Iterable[Dimension]
        the flattened list of dimensions
    """
    resolved_dimension_list: List[Dimension] = []

    dimension_stack = list(base_dimensions)

    while len(dimension_stack) > 0:
        dim1 = dimension_stack.pop(0)
        if not skip_structual_dims or not dim1.only_supports_spec_structure():
            resolved_dimension_list.append(dim1)
        if dim1.has_child_dimensions():
            for child_dim in reversed(dim1.children):
                dimension_stack.insert(0, child_dim)

    return resolved_dimension_list


def _create_dict_from_flat_values(
    dimensions: List[Dimension], inputs, dim_to_index_mapping
) -> Dict:
    """
    Helper function that converts a flat array of values, inputs,
    to a dictionary with the keys representing the dimensions' ids
    and the values from inputs. Also converts np.nan values in input to None.

    Arguments
    ---------
    dimensions : List[Dimension]
        the list of dimensions
    inputs
        a flattened array of values
    dim_to_index_mapping
        the mapping of dimensions'' ids to their index of values in inputs

    Returns
    -------
    Dict
        the dict form of inputs
    """
    dict_values = {}
    for dim in dimensions:

        if dim.has_child_dimensions():
            input_is_null = False
            if dim.id in dim_to_index_mapping:
                # check to see if the input is marked as none
                dim_index = dim_to_index_mapping[dim.id]
                if np.isnan(inputs[dim_index]):
                    input_is_null = True

            if not input_is_null:
                if isinstance(dim, Variant):
                    # must determine which child is active
                    active_option_index = round(inputs[dim_index])
                    dict_values[dim.local_id] = _create_dict_from_flat_values(
                        [dim.children[active_option_index]],
                        inputs,
                        dim_to_index_mapping,
                    )
                else:
                    children_dict = _create_dict_from_flat_values(
                        dim.children, inputs, dim_to_index_mapping
                    )
                    if len(children_dict) > 0:
                        dict_values[dim.local_id] = children_dict
        else:
            if dim.id in dim_to_index_mapping:
                dim_index = dim_to_index_mapping[dim.id]
                if np.isnan(inputs[dim_index]):
                    dict_values[dim.local_id] = None
                else:
                    dict_values[dim.local_id] = dim.convert_to_argument(
                        inputs[dim_index]
                    )

    return dict_values


@dataclass
class Space:
    """
    Composition of dimensions that together define a domain
    of tuples with values from the dimensions.
    """

    dimensions: List[Dimension]

    @property
    def children(self) -> List[Dimension]:
        """
        Gets the list of root-dimensions that form this space.

        Arguments
        ---------
        self : Space
            Space

        Returns
        -------
        List[Dimension]
            a list of dimensions
        """
        return self.dimensions

    def create_dim_map(self) -> Dict[str, Dimension]:
        """
        Creates a dict of dimension's id as keys to
        to the Dimension objects as values.

        Arguments
        ---------
        self : Space
            self

        Returns
        -------
        Dict[str, Dimension]
            the resulting dictionary mapping of ids to Dimension objects
        """
        dim_map = {}

        for dim in create_all_iterable(self.dimensions):
            dim_map[dim.id] = dim

        return dim_map

    def find_parent(self, child_dim: Dimension):

        parents = [self]

        while len(parents) > 0:
            parent_dim = parents.pop()
            for c_dim in parent_dim.children:
                if c_dim.id == child_dim.id:
                    return parent_dim
                elif c_dim.has_child_dimensions():
                    parents.append(c_dim)
        return None

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
        that contribute to the domain of values (Dimensions
        that are only used for structure are not counted).

        Returns
        -------
        int: the number of dimensions, including all children dimensions
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

    def encode_to_zero_one_null_matrix(
        self,
        zero_one_encoded_values: np.array,
        dim_column_map: Dict[str, int],
    ):
        """
        Encodes a 0-1 matrix of values to a 0-1 and np.nan matrix
        given dimensions null_portion attributes

        Arguments
        ---------
        self : Space
            Space
        zero_one_encoded_values : np.array
            the 0-1 matrix with rows representing data-points and columns dimensions
        dim_column_map : Dict[str, int]
            the index of dimensions in zero_one_encoded_values given the dimensions' ids

        Returns
        -------
        np.array
            a 0-1 matrix with np.nan values
        """

        adjusted_encoded_values = np.array(zero_one_encoded_values)
        flatted_dimensions = create_all_iterable(self.children)
        for dim in flatted_dimensions:

            if dim.portion_null is not None and dim.portion_null > 0.0:
                if dim.id in dim_column_map:
                    column_index = dim_column_map[dim.id]
                    encoded_column = zero_one_encoded_values[:, column_index]
                    adjusted_encoded_column = [
                        (
                            (xp - dim.portion_null) / (1.0 - dim.portion_null)
                            if xp is not None and xp > dim.portion_null
                            else None
                        )
                        for xp in encoded_column
                    ]
                    decoded_column = dim.collapse_uniform(encoded_column, True)
                    # address broadcast any nan to children
                    if dim.has_child_dimensions():
                        nan_mask = np.isnan(decoded_column)
                        if nan_mask.sum() > 0:
                            children = create_all_iterable(dim.children)
                            non_nan_mask = ~nan_mask
                            for c_dim in children:
                                if c_dim.id in dim_column_map:
                                    c_column_index = dim_column_map[c_dim.id]
                                    zero_one_encoded_values[:, c_column_index][
                                        non_nan_mask
                                    ] = zero_one_encoded_values[
                                        :, c_column_index
                                    ][
                                        non_nan_mask
                                    ]
                                    zero_one_encoded_values[:, c_column_index][
                                        nan_mask
                                    ] = np.nan
                        if isinstance(dim, Variant):
                            # determine active child, null out others
                            for i, c_dim in enumerate(dim.children):
                                nan_mask = [x != i for x in decoded_column]

                                for c_c_dim in create_all_iterable([c_dim]):
                                    if c_c_dim.id in dim_column_map:
                                        c_column_index = dim_column_map[
                                            c_c_dim.id
                                        ]
                                        zero_one_encoded_values[
                                            :, c_column_index
                                        ][nan_mask] = np.nan

                    adjusted_encoded_values[:, column_index] = (
                        adjusted_encoded_column
                    )
        return adjusted_encoded_values

    def decode_zero_one_matrix(
        self,
        zero_one_encoded_values: np.array,
        dim_column_map: Dict[str, int],
        map_null_to_children_dim=False,
        utilize_null_portitions=True,
    ):
        """
        Decodes a 0-1 matrix given the dimensions specifications.

        Arguments
        ---------
        self : Space
            the specification of dimensions
        x : np.array
            the 0-1 matrix to consider
        dim_column_map : Dict[str, int]
            the index of dimensions given the dimensions' ids
        map_null_to_children_dim=False
            flag to include mapping null across the children cells given
            parents that are determined to be null
        utilize_null_portitions=True
            flag to indicate the activation of logic that addresses
            mapping null values given dimensions' null_portion attribute

        Returns
        -------
        np.array:
            matrix of decoded values with np.nan representing nulls
        """

        decoded_values = np.array(zero_one_encoded_values)
        flatted_dimensions = create_all_iterable(self.children)
        for dim in flatted_dimensions:

            if dim.id in dim_column_map:
                column_index = dim_column_map[dim.id]
                encoded_column = zero_one_encoded_values[:, column_index]
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
                                zero_one_encoded_values[:, c_column_index][
                                    non_nan_mask
                                ] = zero_one_encoded_values[:, c_column_index][
                                    non_nan_mask
                                ]
                                zero_one_encoded_values[:, c_column_index][
                                    nan_mask
                                ] = np.nan
                    if isinstance(dim, Variant):
                        # determine active child, null out others
                        for i, c_dim in enumerate(dim.children):
                            nan_mask = [x != i for x in decoded_column]

                            for c_c_dim in create_all_iterable([c_dim]):
                                if c_c_dim.id in dim_column_map:
                                    c_column_index = dim_column_map[c_c_dim.id]
                                    zero_one_encoded_values[:, c_column_index][
                                        nan_mask
                                    ] = np.nan

                decoded_values[:, column_index] = decoded_column
        return decoded_values


@dataclass
class InputSpace(Space):
    """
    A Space representing inputs into an experimentation subject
    """

    multi_dim_contraints: Optional[List] = None


@dataclass
class OutputSpace(Space):
    """
    A Space representing outputs returned from passing inputs
    through an experimentation subject
    """
