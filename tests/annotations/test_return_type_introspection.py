""" 
    Unit tests considering the introspection of a function's
    return-type specification defining an output space.
"""

from typing import Annotated, Optional, Union

import raxpy.annotations.function_spec as fs
import raxpy.spaces as s
import raxpy


def test_no_return_type_func():
    """
    Testing the code intospection of a function
    without a return-type specified.

    Asserts
    -------
        Output space has 0 dimensions
    """

    def f():
        """
        functions supporting unit test
        """

    output_space = fs.extract_output_space(f)
    assert output_space is not None
    assert output_space.dimensions is not None
    assert len(output_space.dimensions) == 0


def assert_parameters(
    d,
    t,
    id,
    default_value,
    lb,
    ub,
    value_set,
    nullable=False,
    tags=None,
    label=None,
):
    """
    TODO Explain the Function
    """
    assert isinstance(d, t)
    assert d.id == id
    assert d.default_value == default_value
    assert d.lb == lb
    assert d.ub == ub
    assert d.value_set == value_set
    assert d.nullable == nullable
    assert d.tags == tags
    assert d.label == label


def test_unannotated_single_value_return_type_func():
    """
    Testing the code intospection of functions
    with simple, single value return-types specified.

    Asserts
    -------
        Output spaces have 1 dimension with the
        appropriate flags
    """

    def f() -> float:
        """
        Function used for unit testing
        """
        return 0.0

    output_space = fs.extract_output_space(f)
    assert output_space is not None
    assert output_space.dimensions is not None
    assert len(output_space.dimensions) == 1
    assert_parameters(
        output_space.dimensions[0],
        s.Float,
        fs.ID_ROOT_RETURN,
        None,
        None,
        None,
        None,
        nullable=False,
    )

    def f2() -> Optional[float]:
        """
        Function used for unit testing
        """
        return 0.0

    output_space2 = fs.extract_output_space(f2)
    assert output_space2 is not None
    assert output_space2.dimensions is not None
    assert len(output_space2.dimensions) == 1
    assert_parameters(
        output_space2.dimensions[0],
        s.Float,
        fs.ID_ROOT_RETURN,
        None,
        None,
        None,
        None,
        nullable=True,
    )

    def f3() -> Union[int, None]:
        """
        Function used for unit testing
        """
        return 0

    output_space3 = fs.extract_output_space(f3)
    assert output_space3 is not None
    assert output_space3.dimensions is not None
    assert len(output_space3.dimensions) == 1
    assert_parameters(
        output_space3.dimensions[0],
        s.Int,
        fs.ID_ROOT_RETURN,
        None,
        None,
        None,
        None,
        nullable=True,
    )


def test_annotated_single_value_return_type_func():
    """
    Testing the code intospection of functions
    with simple, annotated single value return-types
    specified.

    Asserts
    -------
        Output spaces have 1 dimension with the
        appropriate attributes
    """

    def f() -> Annotated[float, raxpy.Float(label="Zero")]:
        """
        Function used for unit testing
        """
        return 0.0

    output_space = fs.extract_output_space(f)
    assert output_space is not None
    assert output_space.dimensions is not None
    assert len(output_space.dimensions) == 1
    assert_parameters(
        output_space.dimensions[0],
        s.Float,
        fs.ID_ROOT_RETURN,
        None,
        None,
        None,
        None,
        nullable=False,
        label="Zero",
    )

    def f2() -> Annotated[
        Optional[float],
        raxpy.Float(label="ZeroOrNone", tags=[raxpy.tags.MAXIMIZE]),
    ]:
        """
        Function used for unit testing
        """
        return 0.0

    output_space2 = fs.extract_output_space(f2)
    assert output_space2 is not None
    assert output_space2.dimensions is not None
    assert len(output_space2.dimensions) == 1
    assert_parameters(
        output_space2.dimensions[0],
        s.Float,
        fs.ID_ROOT_RETURN,
        None,
        None,
        None,
        None,
        nullable=True,
        tags=[raxpy.tags.MAXIMIZE],
        label="ZeroOrNone",
    )

    def f3() -> (
        Annotated[Union[int, None], raxpy.Integer(label="Int", lb=0, ub=4)]
    ):
        return 0

    output_space3 = fs.extract_output_space(f3)
    assert output_space3 is not None
    assert output_space3.dimensions is not None
    assert len(output_space3.dimensions) == 1
    assert_parameters(
        output_space3.dimensions[0],
        s.Int,
        fs.ID_ROOT_RETURN,
        None,
        0,
        4,
        None,
        nullable=True,
        label="Int",
    )
