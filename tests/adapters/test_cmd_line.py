"""
Test the command line adapter for the raxpy library.
This test simulates command line arguments and verifies that the command
line adapter correctly extracts and converts them to the appropriate types
before calling the target function.
"""

from typing import Annotated, Optional, Union
from dataclasses import dataclass
import sys
import raxpy
from raxpy.adapters import cmd_line


def test_basic_cmd_line_adapter():
    """
    Test the command line adapter by simulating command line arguments and calling a test function.
    """

    # Define a test function with type annotations
    def f(
        x1: float,
        x2: int,
    ):
        return (x1 - 2) ** 2 + (x2 - 3) ** 2

    # Simulate command line arguments
    test_args = ["program_name", "--x1", "5.0", "--x2", "7"]
    # Change sys.argv to simulate command line input, saving the original sys.argv to restore later
    original_argv = sys.argv
    try:

        sys.argv = test_args

        # Call the function using the command line adapter
        result = cmd_line.call(f)

        # Check if the result is as expected
        expected_result = (5.0 - 2) ** 2 + (7 - 3) ** 2
        assert (
            result == expected_result
        ), f"Expected {expected_result}, but got {result}"
    finally:
        sys.argv = original_argv  # Restore the original sys.argv


def test_adv_cmd_line_adapter():
    """
    Test the command line adapter by simulating command line arguments and calling a test function.
    """

    @dataclass
    class Branch1:
        x5: Annotated[float, raxpy.Float(lb=0.0, ub=10.0)]

        def value(self):
            return self.x5

    @dataclass
    class Branch2:
        x6: Annotated[float, raxpy.Float(lb=0.0, ub=10.0)]

        def value(self):
            return self.x6 * 2

    # Define a test function with type annotations
    def f(
        x1: float,
        x2: int,
        x3: Annotated[
            Optional[str], raxpy.Categorical(value_set=("option1", "option2"))
        ],
        x4: Optional[Union[Branch1, Branch2]],
    ):
        x4_value = x4.value() if x4 is not None else 1.0
        if x3 == "option1":
            return (x1 - 2) ** 2 + (x2 - 3) ** 2 * x4_value
        elif x3 == "option2":
            return (x1 + 2) ** 2 + (x2 + 3) ** 2 * x4_value
        else:
            if x4 is None:
                return 1.0
            else:
                return x4.value()

    # Simulate command line arguments
    # Change sys.argv to simulate command line input, saving the original sys.argv to restore later
    original_argv = sys.argv
    try:
        sys.argv = ["program_name", "--x1", "5.0", "--x2", "7"]

        # Call the function using the command line adapter
        result_1 = cmd_line.call(f)

        sys.argv = [
            "program_name",
            "--x1",
            "5.0",
            "--x2",
            "7",
            "--x3",
            "option1",
        ]

        # Call the function using the command line adapter
        result_2 = cmd_line.call(f)

        sys.argv = [
            "program_name",
            "--x1",
            "5.0",
            "--x2",
            "7",
            "--x3",
            "option2",
        ]

        # Call the function using the command line adapter
        result_3 = cmd_line.call(f)

        sys.argv = [
            "program_name",
            "--x1",
            "5.0",
            "--x2",
            "7",
            "--x4",
            "Branch1",
            "--x4##option_0##x5",
            "3.0",
        ]

        # Call the function using the command line adapter
        result_4 = cmd_line.call(f)

        sys.argv = [
            "program_name",
            "--x1",
            "5.0",
            "--x2",
            "7",
            "--x4",
            "Branch2",
            "--x4##option_1##x6",
            "3.0",
        ]

        # Call the function using the command line adapter
        result_5 = cmd_line.call(f)

        # Check if the result is as expected
        expected_result_1 = 1.0
        expected_result_2 = (5.0 - 2) ** 2 + (7 - 3) ** 2
        expected_result_3 = (5.0 + 2) ** 2 + (7 + 3) ** 2
        expected_result_4 = 3.0
        expected_result_5 = 6.0
        assert (
            result_1 == expected_result_1
        ), f"Expected {expected_result_1}, but got {result_1}"
        assert (
            result_2 == expected_result_2
        ), f"Expected {expected_result_2}, but got {result_2}"
        assert (
            result_3 == expected_result_3
        ), f"Expected {expected_result_3}, but got {result_3}"
        assert (
            result_4 == expected_result_4
        ), f"Expected {expected_result_4}, but got {result_4}"
        assert (
            result_5 == expected_result_5
        ), f"Expected {expected_result_5}, but got {result_5}"
    finally:
        sys.argv = original_argv  # Restore the original sys.argv
