"""
Test writing/retrieving experiment design and results to session folders.
"""

import os
from typing import Annotated
import random

import raxpy


def test_session_folders(tmp_path):
    """
    Tests that the session folders are created and used correctly.
    """

    # Define a simple function to test with
    def f(x: Annotated[float, raxpy.Float(lb=0.0, ub=1.0)]) -> float:
        return x**2

    # Use a per-test temporary directory managed by pytest
    session_folder = os.path.join(os.fspath(tmp_path), "session")

    # Perform the experiment with the session folder
    raxpy.perform_experiment(f, n_points=5, session_folder=session_folder)

    # Check that the session folder was created
    assert os.path.exists(session_folder)

    # Check that the results were saved in the session folder
    for i in range(5):
        result_file_path = os.path.join(session_folder, f"result_{i}.pkl")
        assert os.path.exists(result_file_path)

    # No cleanup needed; tmp_path is automatically removed by pytest


def test_session_folder_skips_existing_results(tmp_path):
    """
    Tests that the session folder skips existing results.
    """

    # Define a simple function to test with
    def f(x: Annotated[float, raxpy.Float(lb=0.0, ub=1.0)]) -> float:
        return x**2 * random.random()  # Introduce randomness to test skipping

    # Use a per-test temporary directory managed by pytest
    session_folder = os.path.join(os.fspath(tmp_path), "session")

    # Perform the experiment with the session folder
    _design_1, inputs_1, outputs_1 = raxpy.perform_experiment(
        f, n_points=5, session_folder=session_folder
    )

    # Check that the results were saved in the session folder
    for i in range(5):
        result_file_path = os.path.join(session_folder, f"result_{i}.pkl")
        assert os.path.exists(result_file_path)

    # Now perform the experiment again; it should skip existing results
    _design_2, inputs_2, outputs_2 = raxpy.perform_experiment(
        f, n_points=5, session_folder=session_folder
    )

    # Check that the results were not overwritten (i.e., they still exist)
    for i in range(5):
        result_file_path = os.path.join(session_folder, f"result_{i}.pkl")
        assert os.path.exists(result_file_path)
    assert (
        outputs_1 == outputs_2
    )  # The outputs should be the same since existing results were skipped
    assert inputs_1 == inputs_2  # The inputs should be the same since existing


def test_no_session_folder():
    """
    Tests that the experiment runs without a session folder.
    """

    # Define a simple function to test with
    def f(x: Annotated[float, raxpy.Float(lb=0.0, ub=1.0)]) -> float:
        return (
            x**2 + random.random()
        )  # Introduce randomness to test no session folder

    # Perform the experiment with the session folder
    _design_1, inputs_1, outputs_1 = raxpy.perform_experiment(f, n_points=5)

    # Now perform the experiment again; it should skip existing results
    _design_2, inputs_2, outputs_2 = raxpy.perform_experiment(f, n_points=5)

    # Check that the results were not persisted and reloaded (i.e., they are
    # different)
    assert outputs_1 != outputs_2
    assert inputs_1 != inputs_2
