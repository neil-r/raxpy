"""
Provides adapters to use raxpy with ConfigSpace code and
experiment environments
"""
from typing import List

import numpy as np

import ConfigSpace
from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import (
    BetaFloatHyperparameter,
    BetaIntegerHyperparameter,
    CategoricalHyperparameter, Constant,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    UnParametrizedHyperparameter
)
from ConfigSpace.conditions import (
    AndConjunction,
    EqualsCondition,
    GreaterThanCondition,
    InCondition,
    LessThanCondition,
    NotEqualsCondition,
    OrConjunction,
)

from raxpy.does.doe import DesignOfExperiment
from raxpy.spaces.dimensions import Bool, Float, Int, Text, Variant, Composite
from raxpy.spaces.root import InputSpace
import raxpy.spaces.dim_tags as dim_tags



if ConfigSpace.__version__ < "1.0.0":
    def address_version_differences(config_space):
        class Wrapper:
            def __init__(self, config_space):
                self.config_space = config_space

            @property
            def unconditional_hyperparameters(self):
                return self.config_space.get_all_unconditional_hyperparameters()
            
            def get_hyperparameters(self):
                return self.config_space.get_hyperparameters()

            @property
            def conditional_hyperparameters(self):
                return self.config_space.get_all_conditional_hyperparameters()
        
            def get(self, name):
                return self.config_space.get_hyperparameter(name)
            
            def get_child_conditions_of(self, name):
                return self.config_space.get_child_conditions_of(name)

        return Wrapper(config_space)
else:
    def address_version_differences(config_space):
        return config_space


def convert_config_space(
    config_space: ConfigSpace.ConfigurationSpace,
) -> InputSpace:
    """
    Converts a input space specified in ConfigSpace to a raxpy input space
    specification

    Arguments
    ---------
    config_space : ConfigSpace.ConfigurationSpace
        The input space specified by ConfigSpace

    Returns
    -------
    InputSpace
        the input space specification as a raxpy data structure
    """
    config_space = address_version_differences(config_space)

    resolved_dimensions = {}

    for hp_name in config_space.get_hyperparameters():
        if isinstance(hp_name, str):
            hp = config_space.get(hp_name)
        else:
            hp = hp_name
            hp_name = hp.name
        
        dim = None
        if isinstance(hp, UniformFloatHyperparameter):
            tags = []
            if hp.log:
                tags.append(dim_tags.LOG)
            dim = Float(id=hp_name, lb=hp.lower, ub=hp.upper, tags=tags)

        elif isinstance(hp, UniformIntegerHyperparameter):
            tags = []
            if hp.log:
                tags.append(dim_tags.LOG)
            dim = Int(id=hp_name, lb=hp.lower, ub=hp.upper, tags=tags)
        elif isinstance(hp, OrdinalHyperparameter):
            tags = [dim_tags.ORDINAL]
            value_set = tuple(x for x in hp.sequence)
            if isinstance(value_set[0], int):
                dim = (
                    Int(
                        id=hp_name,
                        value_set=value_set,
                        tags=tags,
                    )
                )
            elif isinstance(value_set[0], float):
                dim = (
                    Float(
                        id=hp_name,
                        value_set=value_set,
                        tags=tags,
                    )
                )
            else:
                dim = (
                    Text(
                        id=hp_name,
                        value_set=value_set,
                        tags=tags,
                    )
                )
        elif isinstance(hp, CategoricalHyperparameter):
            tags = []
            value_set = tuple(x for x in hp.choices)
            if isinstance(value_set[0], int):
                dim = (
                    Int(
                        id=hp_name,
                        value_set=value_set,
                        tags=tags,
                    )
                )
            elif isinstance(value_set[0], bool):
                dim = (
                    Bool(
                        id=hp_name,
                        value_set=value_set,
                        tags=tags,
                    )
                )
            elif isinstance(value_set[0], float):
                dim = (
                    Float(
                        id=hp_name,
                        value_set=value_set,
                        tags=tags,
                    )
                )
            else:
                dim = (
                    Text(
                        id=hp_name,
                        value_set=value_set,
                        tags=tags,
                    )
                )
        elif isinstance(hp, Constant):
            tags = [dim_tags.CONSTANT]
            if isinstance(hp.value, int):
                dim = (
                    Int(
                        id=hp_name,
                        value_set=(hp.value,),
                        tags=tags,
                    )
                )
            elif isinstance(hp.value, bool):
                dim = (
                    Bool(
                        id=hp_name,
                        value_set=(hp.value,),
                        tags=tags,
                    )
                )
            elif isinstance(hp.value, float):
                dim = (
                    Float(
                        id=hp_name,
                        value_set=(hp.value,),
                        tags=tags,
                    )
                )
            elif isinstance(hp.value, str):
                dim = (
                    Text(
                        id=hp_name,
                        value_set=(hp.value,),
                        tags=tags,
                    )
                )
            else:
                raise NotImplementedError(
                    f"Not implemented, failed to convert configspace constant hyperparameter with value of type '{type(hp.value)}'"
                )
        else:
            raise NotImplementedError(
                f"Not implemented, failed to convert configspace paramter of type '{type(hp)}'"
            )
        resolved_dimensions[hp_name] = dim


    def check_children(hp_name, dim, visited_conditions=None):
        # check if any of the children of this dimension are in the resolved dimensions, if so add those
        # children to this dimension
        children_conditions = config_space.get_child_conditions_of(hp_name)
        if children_conditions is not None and len(children_conditions) > 0:
            for condition in children_conditions:
                if isinstance(condition, AndConjunction):
                    match_count = 0
                    unmatched_conditions = None
                    for i, component in enumerate(condition.components):
                        matched = False
                        parent_name = component.parent.name
                        child_name = component.child.name
                        
                        if visited_conditions is not None:
                            for v_condition in visited_conditions:
                                
                                if parent_name == v_condition[0]:
                                    matched = True
                                    break
                        if matched:
                            match_count += 1
                        else:
                            unmatched_conditions = component
                    if match_count == len(condition.components) - 1:
                        # since the other conditions are already handled, 
                        # we can handle this condition as if it is the only condition
                        condition = unmatched_conditions
                    else:
                        #print("Skipping and AndConjunction since other node should handle")
                        #Warning(f"Skipping and AndConjunction for child '{child_name}' of parent '{parent_name}' since other node should handle")
                        continue
                
                if not isinstance(condition, (EqualsCondition, InCondition)):
                    raise NotImplementedError(
                        f"Not implemented, failed to convert configspace condition of type '{type(condition)}'"
                    )
                    
                child_name = condition.child.name
                if child_name in resolved_dimensions:
                    child_dim = resolved_dimensions[child_name]
                    # only check children if the child dimension is not a variant dimension,
                    # since variant dimensions already have their children as options    
                    if not isinstance(child_dim, Variant):
                        c_visited_conditions = visited_conditions.copy() if visited_conditions is not None else list()
                        c_visited_conditions.append((hp_name, child_name))
                        child_dim = check_children(child_name, child_dim, visited_conditions=c_visited_conditions)
                        resolved_dimensions[child_name] = child_dim
                    original_values = dim.get_discrete_values()
                    # convert dim to a variant dimension if it is not already
                    if not isinstance(dim, Variant):
                        dim = Variant(
                            id=dim.id,
                            nullable=False,
                            options=list(
                                Composite(
                                    children=[],
                                    class_name=o_value,
                                    type_class=type(o_value),
                                    id=dim.id + "_" + str(o_value),
                                )
                                for o_value in original_values 
                            ),
                        )
                    
                    if isinstance(condition, EqualsCondition):
                        c_values = [condition.value]
                    elif isinstance(condition, InCondition):
                        c_values = condition.values
                    else:
                        raise NotImplementedError(
                            f"Not implemented, failed to convert configspace condition of type '{type(condition)}'"
                        )
                    if c_values is not None:
                        for c_value in c_values:
                            for option in dim.options:
                                if option.class_name == c_value:
                                    option.children.append(child_dim)
                                    break
        return dim

    root_dims = []
    for hp_name in config_space.unconditional_hyperparameters:
        if isinstance(hp_name, str):
            hp = config_space.get(hp_name)
        else:
            hp = hp_name
            hp_name = hp.name

        dim = resolved_dimensions[hp_name]

        dim = check_children(hp_name, dim)

        root_dims.append(dim)

    """
    if len(config_space.conditional_hyperparameters) > 0:
        stack_cs_dims = list(config_space.conditional_hyperparameters)

        while len(stack_cs_dims) > 0:
            cs_dim_name = stack_cs_dims.pop()
            cs_dim = config_space.get(cs_dim_name)
            
            if dim.child.name not in resolved_dimensions:
                cs_dim.append(dim)
            else:
                resolved_dimensions[dim.child.name] = dim

        raise NotImplementedError(
            "Not implemented, failed to convert conditional hyperparmaeters"
        )
    """
    return InputSpace(root_dims)


def convert_doe(doe:DesignOfExperiment, config_space:ConfigSpace) -> List[Configuration]:
    w_config_space = address_version_differences(config_space)

    points = []

    dim_id_map = doe.index_dim_id_map
    dim_map = doe.input_space.create_dim_map()

    for row in doe.decoded_input_sets:
        point_dict = {}
        for i in range(doe.dim_specification_count):
            dim_id = dim_id_map[i]
            value = row[i]
            if not np.isnan(value):
                dim = dim_map[dim_id]
                if dim is not None:
                    if isinstance(dim, Bool):
                        point_dict[dim_id] = value == 1.0
                    elif isinstance(dim, Int):
                        point_dict[dim_id] = int(value)
                    elif isinstance(dim, Float):
                        point_dict[dim_id] = float(value)
                    elif isinstance(dim, Text):
                        dv = dim.get_discrete_values()
                        if dv is not None:
                            point_dict[dim_id] = dv[int(value)]
                    elif isinstance(dim, Variant):
                        value = int(value)
                        dv = dim.get_discrete_values()
                        if dim.options is not None:
                            point_dict[dim_id] = dv[int(value)]
                    elif isinstance(dim, Composite):
                        # skip, only used for structure, not actually encoded in the doe
                        pass
                    else:
                        raise NotImplementedError(
                            f"Not implemented, failed to convert doe point for dimension of type '{type(dim)}'"
                        )
        
        for hp_name in w_config_space.get_hyperparameters():
            if isinstance(hp_name, str):
                hp = w_config_space.get(hp_name)
            else:
                hp = hp_name
                hp_name = hp.name
            
            if isinstance(hp, Constant):
                point_dict[hp_name] = hp.value

        #points.append(Configuration(config_space,values=point_dict))
        points.append(point_dict)
    return points
