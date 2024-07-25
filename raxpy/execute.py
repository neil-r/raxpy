""" TODO """

from typing import Callable, TypeVar, List, ParamSpec, Tuple

from raxpy.spaces.dimensions import convert_values_from_dict
from raxpy.spaces.complexity import assign_null_portions
from raxpy.spaces.root import InputSpace, create_level_iterable
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
        TODO **Explanation**
    inputs : List[I]
        **Explanation**

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
    TODO Explain the Function

    Arguments
    ---------
    input_space : InputSpace
        **Explanation**
    target_number_of_runs : int
        **Explanation**

    Returns
    -------
    arg_set : List[I]
        **Explanation**
    """

    # assign unassigned null poritions using complexity hueristic
    assign_null_portions(create_level_iterable(input_space.children))

    design = lhs.generate_design(input_space, target_number_of_runs)

    value_dicts = input_space.convert_flat_values_to_dict(
        design.input_sets, design.input_set_map
    )

    arg_set = list(
        convert_values_from_dict(input_space.dimensions, value_dict)
        for value_dict in value_dicts
    )

    return arg_set


def perform_batch_experiment(
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
    """
    """
    TODO Explain the Function

    Arguments
    ---------
    f : Callable[I, T]
        **Explanation**
    target_number_of_runs : int
        **Explanation**
    designer : Callable[[InputSpace, int], List[I]] 
        Set to default= _default_designer TODO**Explanation**
    orchistrator : Callable[[Callable[I, T], List[I]], List[T]]
        Set to default= _default_orchistrator TODO**Explanation**

    Returns
    -------
    arg_sets : 
        **Explanation**
    results : 
        **Explanation**
    """

    input_space = function_spec.extract_input_space(f)
    arg_sets = designer(input_space, target_number_of_runs)
    results = orchistrator(f, arg_sets)
    return arg_sets, results
