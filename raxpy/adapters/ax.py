from typing import Optional, List

import numpy as np

from ax import TParameterization
from ax.api.configs import RangeParameterConfig, ChoiceParameterConfig
from ax.generation_strategy.external_generation_node import (
    ExternalGenerationNode,
)
from ax.core.data import Data
from ax.core.experiment import Experiment

from raxpy.spaces import dim_tags
from raxpy.spaces.complexity import assign_null_portions
from raxpy.spaces.dimensions import convert_values_from_dict
from raxpy.spaces.root import create_all_iterable, create_level_iterable

from ..annotations import function_spec
from ..does.doe import DesignOfExperiment
from ..spaces import (
    InputSpace,
    OutputSpace,
    Dimension,
    Int,
    Float,
    Variant,
    Composite,
    Bool,
    Text,
)


def derive_parameter_configurations_and_constraints(input_space: InputSpace):
    parameters = []
    constraints = []

    # ax does not explicitly consider the conditional branchs of the input space
    # so we can flatten it
    dimensions = create_all_iterable(
        input_space.children, skip_structual_dims=True
    )

    for dim in dimensions:
        if dim.nullable and (
            dim.portion_null is not None and dim.portion_null > 0.0
        ):
            # create a custom parameter to indicate if a nullable parameter should be null
            parameters.append(
                ChoiceParameterConfig(
                    name=f"{dim.id}__nulled",
                    values=[False, True],
                    parameter_type="bool",
                    is_ordered=True,
                )
            )

        value_set = None
        is_ordered = False
        p_type = None
        bounds = None

        if isinstance(dim, Float):
            p_type = "float"
            if dim.value_set is not None:
                value_set = dim.value_set
                is_ordered = True
            elif dim.has_tag("log"):
                bounds = (dim.lb, dim.ub)
            else:
                bounds = (dim.lb, dim.ub)

        elif isinstance(dim, Int):
            p_type = "int"
            if dim.value_set is not None:
                value_set = dim.value_set
                is_ordered = True
            elif dim.has_tag("log"):
                bounds = (dim.lb, dim.ub)
            else:
                bounds = (dim.lb, dim.ub)
        elif isinstance(dim, Composite):
            pass
        elif isinstance(dim, Variant) and dim.options is not None:
            p_type = "int"
            value_set = list(i for i in range(len(dim.options)))
            is_ordered = True
        elif isinstance(dim, Text) and dim.value_set is not None:
            p_type = "str"
            value_set = list(i for i in range(len(dim.value_set)))
            is_ordered = False

        if value_set is not None:
            parameters.append(
                ChoiceParameterConfig(
                    name=f"{dim.id}",
                    values=[False, True],
                    parameter_type="bool",
                    is_ordered=is_ordered,
                )
            )
        elif bounds is not None:
            assert p_type is not None
            parameters.append(
                RangeParameterConfig(
                    name=f"{dim.id}",
                    bounds=bounds,
                    parameter_type=p_type,
                )
            )

    return parameters, constraints


class ExplicitPointsGenerationNode(ExternalGenerationNode):
    """A generation node that emits a predefined list of parameterizations."""

    def __init__(
        self,
        points: List[TParameterization],
        name: str = "ExplicitPoints",
        transition_criteria: Optional[List] = None,
    ) -> None:
        super().__init__(name=name, transition_criteria=transition_criteria)
        self.points: List[TParameterization] = list(points)
        self._current_idx: int = 0

    def update_generator_state(
        self, experiment: Experiment, data: Data
    ) -> None:
        # No model state to fit since candidates are predefined.
        pass

    def new_trial_limit(self, raise_generation_errors: bool = False) -> int:
        return len(self.points)

    def get_next_candidate(self, pending_parameters) -> TParameterization:
        if self._current_idx >= len(self.points):
            raise RuntimeError(
                f"All {len(self.points)} explicit points have already been generated."
            )

        candidate = self.points[self._current_idx]
        self._current_idx += 1
        return candidate


def convert_to_ax(
    f,
    input_space: Optional[InputSpace] = None,
    output_space: Optional[OutputSpace] = None,
):
    """
    Converts a raxpy annotated function f to a hyper-opt space
    and to a hyper-opt optimizable function.

    """
    if input_space is None:
        input_space = function_spec.extract_input_space(f)

    if output_space is None:
        output_space = function_spec.extract_output_space(f)

    # assign unassigned null poritions using complexity hueristic
    assign_null_portions(create_level_iterable(input_space.children))

    parameter_specs, constraints = (
        derive_parameter_configurations_and_constraints(input_space)
    )

    dimensions = create_all_iterable(
        input_space.children, skip_structual_dims=True
    )

    input_set_map = {}

    for i, dim in enumerate(dimensions):
        input_set_map[dim.id] = i

    def ax_callable(parameters):

        # convert ax-parameters to raxpy-compatable dict
        value_array = []

        for dim in dimensions:
            if (
                dim.nullable
                and (dim.portion_null is not None and dim.portion_null > 0.0)
                and parameters[f"{dim.id}__nulled"]
            ):
                value_array.append(np.nan)
            else:
                value_array.append(parameters[dim.id])

        value_dict = input_space.convert_flat_values_to_dict(
            np.array([value_array]), input_set_map
        )[0]

        # convert raxpy-compatable dict to data-structures specs in f
        arg_set = convert_values_from_dict(input_space.children, value_dict)

        outputs = f(**arg_set)

        output_dict = {}

        # check to see if only one output returned
        if len(output_space.children) == 1:
            output_dict[output_space.children[0].id] = outputs
        else:
            for dim, o_value in zip(output_space.children, outputs):
                output_dict[dim.id] = o_value

        return output_dict

    conf_experiment = {
        "parameters": parameter_specs,
        "constraints": constraints,
    }

    objective = None

    for dim in output_space.children:
        if dim.has_tag(dim_tags.MAXIMIZE):
            assert objective is None
            objective = f"-1 * {dim.id}"
        if dim.has_tag(dim_tags.MINIMIZE):
            assert objective is None
            objective = f"{dim.id}"

    assert objective is not None

    conf_optimization = {"objective": objective}

    return ax_callable, conf_experiment, conf_optimization


def convert_doe_to_ax_points(doe: DesignOfExperiment):
    parameterizations = []

    dim_map = doe.input_space.create_dim_map()
    for input_set in doe.decoded_input_sets:
        parameterization = {}

        for dim_index, value in enumerate(input_set):
            dim = dim_map[doe.index_dim_id_map[dim_index]]

            # must inject numeric values for nulled values so ax can still
            # process points
            if (
                dim.nullable
                and dim.portion_null is not None
                and dim.portion_null >= 0
            ):
                if np.isnan(value):
                    if dim.default_value is None:
                        parameterization[dim.id] = 0
                    else:
                        parameterization[dim.id] = dim.default_value
                    parameterization[f"{dim.id}__nulled"] = True
                else:
                    parameterization[f"{dim.id}__nulled"] = False
                    parameterization[dim.id] = value
            else:
                parameterization[dim.id] = value

        # for nullable dimensions, inject zero-one null flag dimension value

        parameterizations.append(parameterization)

    return parameterizations
