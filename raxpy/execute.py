""" 
    This module provides some capabilites that integrate
    the designing and execution of experiments.
"""

import sys

if sys.version_info >= (3, 10):
    from typing import Callable, TypeVar, List, ParamSpec, Tuple
else:
    from typing_extensions import Callable, TypeVar, List, ParamSpec, Tuple

from raxpy.spaces.dimensions import convert_values_from_dict
from raxpy.spaces.complexity import assign_null_portions
from raxpy.spaces import InputSpace, create_level_iterable
from raxpy.annotations import function_spec
from raxpy.does import lhs


T = TypeVar("T")
I = ParamSpec("I")


def _default_orchistrator(f: Callable[I, T], inputs: List[I]) -> List[T]:
    """
    Simply executes the function f sequentially,
    saving and returning the results

    Arguments
    ---------
    f (Function) : Callable[I, T]
        the function to execute
    inputs : List[I]
        the values to pass into the function

    Returns
    -------
    results : List[T]
        List of sequential results from input function
    """
    results = []

    for arg_set in inputs:
        results.append(f(**arg_set))

    return results


def _default_designer(
    input_space: InputSpace, target_number_of_runs: int
) -> List[I]:
    """
    Designs an experiment of size target_number_of_runs
    for the values varied in the input_space.

    Arguments
    ---------
    input_space : InputSpace
        the specification of the range of values to
        consider for the design
    target_number_of_runs : int
        The requested quantity of iterations
        of the experiment being designed

    Returns
    -------
    arg_set : List[I]
        a list of points (i.e., the design of the experiment)
    """

    # assign unassigned null poritions using complexity hueristic
    assign_null_portions(create_level_iterable(input_space.children))

    design = lhs.generate_design(input_space, target_number_of_runs)

    value_dicts = input_space.convert_flat_values_to_dict(
        design.decoded_input_sets, design.input_set_map
    )

    arg_set = list(
        convert_values_from_dict(input_space.dimensions, value_dict)
        for value_dict in value_dicts
    )

    return arg_set


def perform_experiment(
    f: Callable[I, T],
    target_number_of_runs: int,
    designer: Callable[[InputSpace, int], List[I]] = _default_designer,
    orchistrator: Callable[
        [Callable[I, T], List[I]], List[T]
    ] = _default_orchistrator,
) -> Tuple[List[I], List[T]]:
    """
    Executes a batch experiment for function f.
    Begins by inspecting the input space of f.
    Then calls designer to design an experiment
    given the input space specifications.
    Then calls the orchitstrator to orchistrate the
    calling of function f given the input sets.
    The orchistration captures the returned values
    with calling f and returns the input sets and the outputs.

    Arguments
    ---------
    f : Callable[I, T]
        The function to design an experiment with respect to
        and to execute this experiment by calling
    target_number_of_runs : int
        The maximum number of points to execute the function
        with.
    designer : Callable[[InputSpace, int], List[I]]
        A function that designs the experiment
    orchistrator : Callable[[Callable[I, T], List[I]], List[T]]
        A function that executes the experiment on f

    Returns
    -------
    arg_sets : List[I@perform_experiment]
        the points passed into f as inputs (i.e., the design of the experiment)
    results : List[T@perform_experiment]
        the values returned from f for each point
    """

    input_space = function_spec.extract_input_space(f)
    arg_sets = designer(input_space, target_number_of_runs)
    results = orchistrator(f, arg_sets)
    return arg_sets, results
