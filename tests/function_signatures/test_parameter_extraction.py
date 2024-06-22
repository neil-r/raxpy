from typing import Annotated, Optional

import raxpy.annotations.function_spec as fs
import raxpy.spaces.dimensions as d
import raxpy


def test_no_param_func():
  def f():
    pass

  input_space = fs.extract_input_space(f)
  assert input_space is not None
  assert input_space.dimensions is not None
  assert len(input_space.dimensions) == 0


def assert_parameters(d, t, id, default_value, lb, ub, value_set, optional=False, tags=None):
  assert type(d) == t
  assert d.id == id
  assert d.default_value == default_value
  assert d.lb == lb
  assert d.ub == ub
  assert d.value_set == value_set
  assert d.optional == optional
  assert d.tags == tags

def test_single_param_func():
  def f(
      x: Annotated[int, raxpy.Integer(
        lb=0,
        ub=5
      )] = 2,
  ):
    return x * 2

  input_space = fs.extract_input_space(f)
  assert input_space is not None
  assert input_space.dimensions is not None
  assert len(input_space.dimensions) == 1
  assert_parameters(input_space.dimensions[0], d.Int, "x", 2, 0, 5, None)


def test_mixed_spec_param_func():
  def f(
      x1: Annotated[int, raxpy.Integer(
        lb=0,
        ub=5
      )],
      x2: Annotated[float, raxpy.Float(
        lb=1.7,
        ub=3.3
      )],
      x3: Annotated[int, raxpy.Integer(
        ub=5
      )],
      x4: Annotated[Optional[int], raxpy.Integer(
        value_set={1,2,4}
      )] = None,
      x5: int = 3,
      x6: Optional[float] = None,
  ):
    return x1 * 2

  input_space = fs.extract_input_space(f)
  assert input_space is not None
  assert input_space.dimensions is not None
  assert len(input_space.dimensions) == 6
  assert_parameters(input_space.dimensions[0], d.Int, "x1", None, 0, 5, None)
  assert_parameters(input_space.dimensions[1], d.Float, "x2", None, 1.7, 3.3, None)
  assert_parameters(input_space.dimensions[2], d.Int, "x3", None, None, 5, None)
  assert_parameters(input_space.dimensions[3], d.Int, "x4", None, None, None, value_set={1,2,4}, optional=True)
  assert_parameters(input_space.dimensions[4], d.Int, "x5", 3, None, None, None, optional=False)
  assert_parameters(input_space.dimensions[5], d.Float, "x6", None, None, None, None, optional=True)


def test_object_spec_param_func():

  class CustomObject:
    pass

  def f(obj1: CustomObject):
    pass

  input_space = fs.extract_input_space(f)
  assert input_space is not None
  assert input_space.dimensions is not None
  assert len(input_space.dimensions) == 1
