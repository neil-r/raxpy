"""
This module provides some capabilites that integrate
the designing and execution of experiments.
"""

import sys
import os
import pickle
from typing import Optional, Union
from functools import partial
import numpy as np

if sys.version_info >= (3, 10):
    from typing import Callable, TypeVar, List, ParamSpec, Tuple, Dict
else:
    from typing_extensions import (
        Callable,
        TypeVar,
        List,
        ParamSpec,
        Tuple,
        Dict,
    )

from raxpy.does.doe import DesignOfExperiment
from raxpy.spaces.dimensions import convert_values_from_dict
from raxpy.spaces.complexity import assign_null_portions
from raxpy.spaces import InputSpace, create_level_iterable
from raxpy.annotations import function_spec
from raxpy.does import lhs
from raxpy.does import maxpro
from raxpy.does import random

T = TypeVar("T")
P = ParamSpec("P")


def _default_orchistrator(
    f: Callable[P, T],
    inputs: List[Dict],
    session_folder: Optional[str] = None,
) -> List[T]:
    """
    Simply executes the function f sequentially,
    saving and returning the results

    Arguments
    ---------
    f (Function) : Callable[I, T]
        the function to execute
    inputs : List[I]
        the values to pass into the function
    session_folder : Optional[str]
        If specified, the folder to save the results of the experiment

    Returns
    -------
    results : List[T]
        List of sequential results from input function
    """
    results = []

    for i, arg_set in enumerate(inputs):
        already_saved = False
        result = None
        result_file_path = ""
        if session_folder is not None:
            # Check to see if the results have already been saved for this
            # input set
            result_file_path = os.path.join(session_folder, f"result_{i}.pkl")
            if os.path.exists(result_file_path):
                with open(result_file_path, "rb") as fi:
                    result = pickle.load(fi)
                already_saved = True

        if not already_saved:
            # redirect stdout and stderr to files in the session folder
            if session_folder is not None:
                stdout_file_path = os.path.join(
                    session_folder, f"stdout_{i}.txt"
                )
                stderr_file_path = os.path.join(
                    session_folder, f"stderr_{i}.txt"
                )
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                with (
                    open(
                        stdout_file_path, "w", encoding="utf-8"
                    ) as stdout_file,
                    open(
                        stderr_file_path, "w", encoding="utf-8"
                    ) as stderr_file,
                ):
                    sys.stdout = stdout_file
                    sys.stderr = stderr_file
                    try:
                        result = f(**arg_set)  # type: ignore
                    except Exception as e:
                        # If an exception occurs, write it to the stderr file
                        print(
                            f"Exception occurred while executing f: {e}",
                            file=sys.stderr,
                        )
                        raise e
                    finally:
                        # Reset stdout and stderr to the values active
                        # before redirection (important for notebooks).
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr
            else:
                result = f(**arg_set)  # type: ignore
            if session_folder is not None:
                # Save the results to the specified folder
                with open(result_file_path, "wb") as fi:
                    pickle.dump(result, fi)
        results.append(result)  # type: ignore

    return results


def _default_designer(
    input_space: InputSpace, n_points: int, seed: Optional[int] = None
) -> DesignOfExperiment:
    """
    Designs an experiment of size target_number_of_runs
    for the values varied in the input_space.

    Arguments
    ---------
    input_space : InputSpace
        the specification of the range of values to
        consider for the design
    n_points : int
        The requested quantity of iterations
        of the experiment being designed
    seed : Optional[int]
        If specified, seeds the random number generator(s) to ensure
        design is created the same way

    Returns
    -------
    design : DesignOfExperiment

    """

    design = design_experiment(input_space, n_points=n_points, seed=seed)

    return design


def perform_experiment(
    f: Callable[P, T],
    n_points: int,
    designer: Callable[
        [InputSpace, int, Optional[int]], DesignOfExperiment
    ] = _default_designer,
    orchistrator: Callable[
        [Callable[P, T], List[Dict], Optional[str]], List[T]
    ] = _default_orchistrator,
    session_folder: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[DesignOfExperiment, List[Dict], List[T]]:
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
    n_points : int
        The maximum number of points to execute the function
        with.
    designer : Callable[[InputSpace, int], List[I], Optional[int]]
        A function that designs the experiment
    session_folder : Optional[str]
        If specified, the folder to save the results of the experiment
    orchistrator : Callable[[Callable[I, T], List[I]], List[T]]
        A function that executes the experiment on f

    Returns
    -------
    arg_sets : List[I@perform_experiment]
        the points passed into f as inputs (i.e., the design of the experiment)
    results : List[T@perform_experiment]
        the values returned from f for each point
    """

    if session_folder is not None:
        if os.path.exists(session_folder):
            if not os.path.isdir(session_folder):
                raise ValueError(
                    f"session_folder {session_folder} exists "
                    "but is not a directory"
                )
        else:
            os.makedirs(session_folder, exist_ok=True)

        design_file_path = os.path.join(session_folder, "design.pkl")
        if os.path.exists(design_file_path):
            with open(design_file_path, "rb") as fi:
                design = pickle.load(fi)
                input_space = design.input_space
        else:
            input_space = function_spec.extract_input_space(f)
            design = designer(input_space, n_points, seed)
            with open(design_file_path, "wb") as fi:
                pickle.dump(design, fi)
    else:
        input_space = function_spec.extract_input_space(f)
        design = designer(input_space, n_points, seed)

    value_dicts = input_space.convert_flat_values_to_dict(
        design.decoded_input_sets, design.input_set_map
    )
    arg_sets = list(
        convert_values_from_dict(input_space.dimensions, value_dict)
        for value_dict in value_dicts
    )

    results = orchistrator(f, arg_sets, session_folder)
    return design, arg_sets, results


def design_experiment(
    subject: Union[
        InputSpace,
        Callable[P, T],
    ],
    n_points: int,
    design_algorithm=lhs.generate_seperate_designs_by_full_subspace_and_pool,
    optimize_projections: bool = True,
    seed: Optional[int] = None,
) -> DesignOfExperiment:
    """
    Designs a batch experiment for the subject; ensures that all optional
    dimensions have null poritions specifications.

    Arguments
    ---------
    subject : Callable[I, T]
        The function to design an experiment with respect to
        or the InputSpace specification
    n_points : int
        The maximum number of points to execute the function
        with.
    design_algorithm
        Generates an initial experiment that could be optimize further
    optimize_projections: bool
        If true, optimizes the design created by the design_algorithm to
        maximize the projections
    seed:Optional[int]
        If specified, seeds the random number generator(s) to ensure
        design is created the same way

    Returns
    -------
    DesignOfExperiment
        The designed experiment
    """
    if isinstance(subject, InputSpace):
        input_space = subject
    else:
        input_space = function_spec.extract_input_space(subject)

    # assign unassigned null poritions using complexity hueristic
    assign_null_portions(create_level_iterable(input_space.children))

    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()

    design = design_algorithm(input_space, n_points, rng=rng)
    if optimize_projections:
        design = maxpro.optimize_design_with_sa(
            design,
            # We want to ensure that the projections are optimized
            # with the encoded values not the decoded values
            # (Decoded values could be on different unit-of-measurement scales
            # and the optimization is more effective)
            encoding=maxpro.EncodingEnum.ZERO_ONE_NULL_ENCODING,
            rng=rng,
        )

    return design


design_simple_random_experiment = partial(
    design_experiment,
    design_algorithm=random.generate_seperate_designs_by_full_subspace,
    optimize_projections=False,
)

generate_random_design = partial(
    design_experiment,
    design_algorithm=random.generate_seperate_designs_by_full_subspace,
)
