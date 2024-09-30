"""
    This modules provides support to annotate and inspect
    function signatures.
"""

from .function_spec import extract_input_space, extract_output_space
from .values import Float, Integer, Categorical, CategorySpec, Binary
