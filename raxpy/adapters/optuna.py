"""
Provides helper functions to simplify the use of raxpy annotated functions 
with the Optuna Hyper-parameter optmization library.
"""

from typing import Optional
import numpy as np
import optuna

from raxpy.spaces.complexity import assign_null_portions
from raxpy.spaces.root import create_level_iterable

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
)


def _create_argument(d: Dimension, trial: optuna.Trial):
    """
    Extracts a function argument represented by the provided dimension
    from an Optuna trial.
    """
    if d.nullable and d.portion_null > 0.0:
        null_sample = trial.suggest_float(f"{d.id}##NULL", low=0.0, high=1.0)

        if null_sample <= d.portion_null:
            return None

    if isinstance(d, Bool):
        if d.lb is not None and d.ub is not None:
            return (
                False
                if trial.suggest_int(
                    d.id, low=d.lb, high=d.ub, log=d.has_tag("log")
                )
                == 0
                else True
            )

    if isinstance(d, Int):
        if d.value_set is not None:
            return d.value_set[
                trial.suggest_int(d.id, low=0, high=len(d.value_set) - 1)
            ]
        elif d.lb is not None and d.ub is not None:
            return trial.suggest_int(
                d.id, low=d.lb, high=d.ub, log=d.has_tag("log")
            )

    elif isinstance(d, Float):
        if d.value_set is not None:
            return d.value_set[
                trial.suggest_int(d.id, low=0, high=len(d.value_set) - 1)
            ]
        elif d.lb is not None and d.ub is not None:
            return trial.suggest_float(
                d.id, low=d.lb, high=d.ub, log=d.has_tag("log")
            )

    elif isinstance(d, Variant):

        if len(d.options) > 2:
            c_value = trial.suggest_categorical(d.id, range(len(d.options)))
        else:
            c_value = trial.suggest_int(d.id, low=0, high=len(d.options) - 1)
        c_dim = d.options[c_value]
        return _create_argument(c_dim, trial)
    elif isinstance(d, Composite):
        args = {}

        for child in d.children:
            args[child.local_id] = _create_argument(child, trial)

        return d.type_class(**args)

    if d.specified_default:
        return d.default_value

    assert NotImplementedError(
        f"Conversion for dimension {d} not implemented."
    )


def call_raxpy_f(
    trial,
    f,
    input_space: InputSpace,
):
    """
    Calls a raxpy annotated function f with the arguments extracted
    from an Optuna trial.
    """
    args = {}

    for child in input_space.children:
        args[child.local_id] = _create_argument(child, trial)

    return f(**args)


def _add_to_optuna_trial_dict(
    dim: Dimension, point, doe: DesignOfExperiment, trial_dict: dict
):
    """
    Converts a dimension's decoded point from a design of experiment
    into a dict format for use to represent the point as a Optuna
    trial.
    """
    if dim.id in doe.input_set_map:
        c_add = True
        if dim.nullable and dim.portion_null > 0.0:
            if np.isnan(point[doe.input_set_map[dim.id]]):
                trial_dict[f"{dim.id}##NULL"] = 0.0
                c_add = False
            else:
                trial_dict[f"{dim.id}##NULL"] = 1.0
        if c_add:
            if isinstance(dim, Bool):
                trial_dict[dim.id] = point[doe.input_set_map[dim.id]]
            elif isinstance(dim, Int):
                if dim.value_set is not None:
                    value = int(point[doe.input_set_map[dim.id]])
                    v_index = dim.value_set.index(value)
                    trial_dict[dim.id] = float(v_index)
                else:
                    trial_dict[dim.id] = point[doe.input_set_map[dim.id]]
            elif isinstance(dim, Float):
                if dim.value_set is not None:
                    value = float(point[doe.input_set_map[dim.id]])
                    v_index = dim.value_set.index(value)
                    trial_dict[dim.id] = float(v_index)
                else:
                    trial_dict[dim.id] = point[doe.input_set_map[dim.id]]
            elif isinstance(dim, Composite):
                for child in dim.children:
                    _add_to_optuna_trial_dict(child, point, doe, trial_dict)
            elif isinstance(dim, Variant):
                option_index = int(point[doe.input_set_map[dim.id]])
                trial_dict[f"{dim.id}"] = option_index
                o_dim = dim.options[option_index]
                _add_to_optuna_trial_dict(o_dim, point, doe, trial_dict)
    elif isinstance(dim, Composite):
        for child in dim.children:
            _add_to_optuna_trial_dict(child, point, doe, trial_dict)


def enqueue_trials_from_doe_to_study(
    doe: DesignOfExperiment, study: optuna.Study
):
    """
    Maps points from a design of experiment and adds them to the
    Optuna study's trial queue.
    """
    for point in doe.decoded_input_sets:
        trial_dict = {}

        for root_dim in doe.input_space.children:
            _add_to_optuna_trial_dict(root_dim, point, doe, trial_dict)

        study.enqueue_trial(trial_dict)


def convert_to_optuna(
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

    def optuna_callable(trial):
        results = call_raxpy_f(trial, f, input_space)

        return results

    return optuna_callable
