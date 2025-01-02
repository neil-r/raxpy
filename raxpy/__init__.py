"""
raxpy: a RApid eXperimentation tool
"""

from .annotations import function_spec
from .annotations.values import *
from .annotations.mixture import *
from .execute import (
    perform_experiment,
    design_experiment,
    generate_random_design,
)
from .decorators import validate_at_runtime
from .spaces import dim_tags as tags
from . import spaces
from .does import measure
