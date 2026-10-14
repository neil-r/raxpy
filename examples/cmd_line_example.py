"""
This example demonstrates how to call an annotated function from the command line using the raxpy command-line adapter.
The command line adapter inspects the function signature, pulling command line arguments and converting them to the appropriate types. Then it calls the function with those arguments.

To run this example, use the following command in the terminal:
python examples/cmd_line_example.py --x1 5.0 --x2 7

"""

from typing import Annotated
import raxpy
import raxpy.adapters.cmd_line as cmd_line_adapter


def f(
    x1: Annotated[float, raxpy.Float(lb=-10.0, ub=10.0)],
    x2: Annotated[float, raxpy.Float(lb=-10.0, ub=10.0)],
):
    return (x1 - 2) ** 2 + (x2 - 3) ** 2


if __name__ == "__main__":
    print("Calling function f with arguments extracted from the command line:")
    print(cmd_line_adapter.call(f))
