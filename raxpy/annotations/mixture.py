""" TODO Explain Module"""

from dataclasses import dataclass
from . import values


@dataclass
class Mixture:
    """
    TODO Explain Class
    """
    label: str
    limit: float = 1.0

    def create_component_meta(self, component_label) -> float:
        """
        TODO Explain the Function

        Arguments
        ---------
        self
            **Explanation**
        component_label
            **Explanation**

        Returns
        -------
        **Labelled float value?**
        """
        return values.Float(component_label, self.label)
