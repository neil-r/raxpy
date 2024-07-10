"""
This modules provides logic to run an experiment by executing each data point sequentially on the local system.
"""

from typing import Callable

from ..does.doe import DesignOfExperiment
from ..annotations import function_spec


def execute(design: DesignOfExperiment, func: Callable):
    results = []

    arg_sets = function_spec.map_design_to_function_spec(design, func)

    for args in arg_sets:
        results.append(func(args))

    return results
