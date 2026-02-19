import pytest
import numpy as np

from raxpy.spaces.dimensions import Float, Text
from raxpy.spaces import dim_tags

try:
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.hyperparameters import (
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
        CategoricalHyperparameter,
        OrdinalHyperparameter,
    )
    from raxpy.adapters.config_space import convert_config_space
except ImportError:
    pytest.skip(
        "Adapative Experiment (ax) module not available, so skipping ConfigSpace adapter tests",
        allow_module_level=True,
    )


def create_complex_configspace() -> ConfigurationSpace:
    """
    Creates and returns a ConfigSpace object with various hyperparameter types.

    This function showcases the creation of float, integer, categorical,
    and ordinal hyperparameters, and adds them to a single ConfigurationSpace.

    Returns:
        ConfigurationSpace: The configured space object.
    """
    # 1. Initialize a ConfigurationSpace object.
    # The 'seed' is used for reproducibility of random operations.
    cs = ConfigurationSpace(seed=1234)

    # 2. Define different types of hyperparameters.

    # A uniform float hyperparameter for a learning rate.
    # It can take any value between 0.0001 and 0.1, on a log scale.
    learning_rate = UniformFloatHyperparameter(
        name="learning_rate", lower=0.0001, upper=0.1, log=True
    )

    # A uniform integer hyperparameter for the number of estimators.
    # It can take any integer value between 50 and 200.
    n_estimators = UniformIntegerHyperparameter(
        name="n_estimators", lower=50, upper=200
    )

    # A categorical hyperparameter for the choice of activation function.
    # It can be one of the three specified string values.
    activation = CategoricalHyperparameter(
        name="activation",
        choices=["relu", "tanh", "sigmoid"],
        default_value="relu",
    )

    # An ordinal hyperparameter for the optimizer level.
    # The order of the sequence matters ('low' < 'medium' < 'high').
    optimizer_level = OrdinalHyperparameter(
        name="optimizer_level", sequence=["low", "medium", "high"]
    )

    # A categorical hyperparameter with a boolean-like choice.
    use_dropout = CategoricalHyperparameter(
        name="use_dropout", choices=[True, False]
    )

    # 3. Add the defined hyperparameters to the configuration space.
    cs.add(
        [learning_rate, n_estimators, activation, optimizer_level, use_dropout]
    )

    return cs


from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
    OrdinalHyperparameter,
)


def create_complex_configspace_with_conditions() -> ConfigurationSpace:
    """
    Creates and returns a ConfigSpace object with various hyperparameter types.

    This function showcases the creation of float, integer, categorical,
    ordinal, and conditional hyperparameters, and adds them to a single
    ConfigurationSpace.

    Returns:
        ConfigurationSpace: The configured space object.
    """
    cs = create_complex_configspace()

    # A float hyperparameter for the dropout rate, conditional on use_dropout.
    dropout_rate = UniformFloatHyperparameter(
        name="dropout_rate", lower=0.1, upper=0.5
    )

    # 3. Add the defined hyperparameters to the configuration space.
    cs.add(
        [
            dropout_rate,
        ]
    )

    # 4. Define a condition.
    # The dropout_rate is only active if use_dropout is True.
    dropout_condition = InCondition(
        child=dropout_rate, parent=cs["use_dropout"], values=[True]
    )

    # 5. Add the condition to the configuration space.
    cs.add(dropout_condition)

    return cs


def test_converting_config_space():
    """
    Tests the conversion of a config_space to raxpy spaces.

    Asserts
    -------
        that the proper number of dimensions are converted

    """
    cs = create_complex_configspace()

    rx_space = convert_config_space(cs)

    assert len(rx_space.children) == 5

    dim_map = rx_space.create_dim_map()
    dim_lr = dim_map["learning_rate"]

    assert isinstance(dim_lr, Float)
    assert dim_lr.tags is not None and "log" in dim_lr.tags
    assert dim_lr.lb == 0.0001
    assert dim_lr.ub == 0.1

    dim_a = dim_map["activation"]

    assert isinstance(dim_a, Text)
    assert dim_a.value_set is not None
    assert "relu" in dim_a.value_set

    dim_opt_level = dim_map["optimizer_level"]
    assert isinstance(dim_opt_level, Text)
    assert (
        dim_opt_level.tags is not None
        and dim_tags.ORDINAL in dim_opt_level.tags
    )
    assert dim_opt_level.value_set is not None
    assert "high" in dim_opt_level.value_set


def test_conditional_config_space():
    """
    Tests the conversion of a config_space with a condition to raxpy spaces
    throws an exception

    Asserts
    -------
        that a config space with conditions is recognized and throws an
        exception

    """
    cs = create_complex_configspace_with_conditions()

    threw_exception = False
    try:
        convert_config_space(cs)
    except NotImplementedError:
        threw_exception = True

    assert threw_exception
