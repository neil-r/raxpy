import raxpy.spaces.dimensions as d
import raxpy.spaces.root as s


def test_deriving_subspaces():
  space = s.Space(
    dimensions=[
      d.Float(
        id="x1",
        lb=3.0,
        ub=5.0,
        nullable=False
      ),
      d.Float(
        id="x2",
        lb=-3.0,
        ub=-5.0,
        nullable=True,
        portion_null=0.33
      ),
      d.Composite(
        id="x3",
        nullable=True,
        portion_null=0.33,
        children=[
          d.Int(
            id="x4",
            lb=6,
            ub=7,
            nullable=False
          ),
          d.Composite(
            id="x6",# Note that x6 is ignore since it is just for the specification's structure
            nullable=False,
            children=[d.Float(
              id="x5",
              value_set=[0.1,0.5,0.9],
              nullable=True,
              portion_null=0.33
            )],
          ),
        ]
      )
    ]
  )

  sub_spaces = space.derive_subspaces()

  assert sub_spaces is not None
  assert len(sub_spaces) == 6
  assert ["x1", "x2", "x3", "x4", "x5"] in sub_spaces
  assert ["x1", "x2", "x3", "x4"] in sub_spaces
  assert ["x1", "x3", "x4", "x5"] in sub_spaces
  assert ["x1", "x3", "x4"] in sub_spaces
  assert ["x1", "x2"] in sub_spaces
  assert ["x1"] in sub_spaces
