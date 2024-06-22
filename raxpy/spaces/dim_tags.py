"""
  This module defines Dimension object tags that can be attached to dimensions
  to provide suggestions to algorithms in how to treat the dimensions.
"""

# Input Dimension Tags

FIXED = "fixed"
"""
Flag to indicate an input dimension as fixed or static (not to change and to
always use the default value during exploration)
"""

DEPENDENT = "dependent"
"""
Flag to indicate an input dimension as dependent on other inputs and should
be dynamically derived from the values of inputs; the expression attribute
specifies this computation
"""


# Ouptut Dimension Tags

MAXIMIZE = "maximize"
"""
Flag to indicate an output dimension as an target dimension to maximize during
optimization
"""

MINIMIZE = "minimize"
"""
Flag to indicate an output dimension as an target dimension to minimize during
optimization
"""

# Input or Output Dimension Tags

ORDINAL = "ordinal"
"""
Flag to indicate a categorial dimension as having a natural order
"""