from typing import Iterable, List

import math

from . import dimensions as d
from . import dim_tags
from . import root as s


def estimate_complexity(dim: d.Dimension) -> float:
    complexity_estimate = 1.0

    # default complexity heuristics
    if isinstance(dim, d.Float):
        if dim.lb is not None and dim.ub is not None:
            complexity_estimate = 9.0
        elif dim.value_set is not None:
            complexity_estimate = len(dim.value_set)
        else:
            # TODO issue warning about not supporting complexity of unbounded floats
            return 0.0
    elif isinstance(dim, d.Int):
        if dim.value_set is not None:
            complexity_estimate = len(dim.value_set)
        elif dim.lb is not None and dim.ub is not None:
            complexity_estimate = min(9.0, dim.ub - dim.lb + 1)
    elif isinstance(dim, d.Text):
        if dim.value_set is not None:
            complexity_estimate = len(dim.value_set)
    elif isinstance(dim, d.Variant):
        complexity_estimate = 0.0
        # summerize complexity of children
        for child in dim.children:
            complexity_estimate += estimate_complexity(child)
    elif isinstance(dim, d.Composite):
        complexity_estimate = 0.0
        expected_significant_interactions = dim.has_tag(
            dim_tags.EXPECT_INTERACTIONS
        )

        if expected_significant_interactions:
            complexity_estimate = 1.0
        # summerize complexity of children
        for child in dim.children:
            if expected_significant_interactions:
                complexity_estimate *= estimate_complexity(child)
            else:
                complexity_estimate += estimate_complexity(child)

    if dim.nullable:
        complexity_estimate += 1.0

    return complexity_estimate


def assign_null_portions(
    dimensions: Iterable[d.Dimension], complexity_estimator=estimate_complexity
):

    children_sets: List[Iterable[d.Dimension]] = []

    # compute porition for active dimensions
    for dim in dimensions:

        if dim.nullable:
            complexity_estimate = complexity_estimator(dim)

            dim.portion_null = 1.0 / complexity_estimate
        else:
            dim.portion_null = 0.0

        if (
            dim.has_child_dimensions()
            and not dim.only_supports_spec_structure()
        ):
            children_sets.append(s.create_level_iterable(dim.children))

    # compute portions for children set dimensions
    for children_set in children_sets:
        assign_null_portions(children_set, complexity_estimator)


def compute_subspace_portitions(
    space: s.Space, full_subspace_sets: List[List[str]]
) -> List[float]:
    portitions = []
    # compute portion of the n_points that each sub-design for each sub-space
    # should address
    for full_subspace in full_subspace_sets:
        portition_components = []

        l1 = s.create_level_iterable(space.children)

        levels_to_process = [l1]

        while len(levels_to_process) > 0:
            active_level = levels_to_process.pop(0)

            for dim in active_level:

                if dim.id in full_subspace:
                    if dim.nullable:
                        p = 1.0 - dim.portion_null
                    else:
                        p = 1.0

                    if dim.has_child_dimensions():
                        if isinstance(dim, d.Variant):

                            portition_components.append(
                                1.0 / len(dim.children)
                            )
                            # only add active child to be processed
                            for child_dim in dim.children:
                                if child_dim.id in full_subspace:
                                    levels_to_process.append(
                                        s.create_level_iterable([child_dim])
                                    )
                        else:
                            levels_to_process.append(
                                s.create_level_iterable(dim.children)
                            )
                else:
                    p = dim.portion_null

                portition_components.append(p)

        portitions.append(math.prod(portition_components))

    return portitions
