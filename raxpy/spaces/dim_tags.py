"""
  This module defines Dimension object tags that can be attached
  to dimensions to provide suggestions to algorithms in how to
  treat the dimensions.
"""

# Input Dimension Tags

FIXED = "fixed"
"""
Flag to indicate an input dimension as fixed or static (not to change
and to always use the default value during exploration).
"""

DEPENDENT = "dependent"
"""
Flag to indicate an input dimension as dependent on other inputs and
should be dynamically derived from the values of inputs; the expression
attribute specifies this computation.
"""

EXPECT_INTERACTIONS = "expect_interactions"
"""
Flag to indicate a composite input dimension as having children
dimensions that cause significant interaction effects to the outputs
of interest.

This flag may be used in dimensional complexity hueristics to support
how to summarize the complexity of multiple children dimensions.
"""


# Ouptut Dimension Tags

MAXIMIZE = "maximize"
"""
Flag to indicate an output dimension as a target dimension to maximize
during optimization.
"""

MINIMIZE = "minimize"
"""
Flag to indicate an output dimension as a target dimension to minimize
during optimization.
"""

# Input or Output Dimension Tags

ORDINAL = "ordinal"
"""
Flag to indicate a categorial dimension as having a natural order.
"""
