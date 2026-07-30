"""
raxpy: a RApid eXperimentation tool
"""

from .annotations import function_spec  # noqa: F401
from .annotations.values import *  # noqa: F401
from .execute import (  # noqa: F401
    perform_experiment,
    design_experiment,
    generate_random_design,
    design_simple_random_experiment,
)
from .decorators import validate_at_runtime  # noqa: F401
from .spaces import dim_tags as tags  # noqa: F401
from . import spaces  # noqa: F401
from .does import measure  # noqa: F401
