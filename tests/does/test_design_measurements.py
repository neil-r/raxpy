""" TODO Explain Module"""

import numpy as np
from scipy.stats.qmc import discrepancy

import raxpy.does.measure as a
import raxpy.spaces as s
import raxpy.does.doe as doe


def test_doe_assessments():
    """
    Tests the all-metric assessment function to ensure no major bugs stop the
    computation of all metrics

    Asserts
    -------
        the assessment computations do not raise an exception and returns an
        assessment object
    """
    space = s.InputSpace(
        dimensions=[
            s.Float(id="x1", lb=0.0, ub=1.0, nullable=True, portion_null=0.1),
            s.Float(id="x2", lb=0.0, ub=1.0, nullable=True, portion_null=0.1),
            s.Float(id="x3", lb=0.0, ub=1.0, nullable=True, portion_null=0.1),
        ]
    )

    design = doe.DesignOfExperiment(
        input_space=space,
        input_set_map={"x1": 0, "x2": 1, "x3": 2},
        input_sets=np.array(
            [
                [np.nan, 0.1, 0.1],
                [np.nan, 0.1, 0.2],
                [np.nan, np.nan, 0.1],
                [0.3, 0.1, 0.7],
                [0.4, 0.2, 0.8],
                [0.5, 0.3, 0.9],
                [0.6, 0.4, 0.6],
                [np.nan, np.nan, np.nan],
            ]
        ),
        encoding=doe.EncodingEnum.NONE,
    )

    assessment = a.measure_with_all_metrics(
        design, encoding=doe.EncodingEnum.NONE
    )

    assert assessment is not None


def test_metric_computations():
    """
    Tests a many of the DOE metric computations.

    Asserts
    -------
    a.compute_portion_of_total == 1.0 / 8.0
    assert a.compute_portion_of_total(sub_space_doe) == 4.0 / 8.0
    assert a.compute_discrepancy(sub_space_doe) == discrepancy()
    assert mipd > 0.0
    assert ard > 0.0
    assert mst_mean > 0.0
    assert mst_std > 0.0

    """
    space = s.InputSpace(
        dimensions=[
            s.Float(id="x1", lb=0.0, ub=1.0, nullable=True, portion_null=0.1),
            s.Float(id="x2", lb=0.0, ub=1.0, nullable=True, portion_null=0.1),
            s.Float(id="x3", lb=0.0, ub=1.0, nullable=True, portion_null=0.1),
        ]
    )

    whole_doe = doe.DesignOfExperiment(
        input_space=space,
        input_set_map={"x1": 0, "x2": 1, "x3": 2},
        input_sets=np.array(
            [
                [np.nan, 0.1, 0.1],
                [np.nan, 0.1, 0.2],
                [np.nan, np.nan, 0.1],
                [0.3, 0.1, 0.7],
                [0.4, 0.2, 0.8],
                [0.5, 0.3, 0.9],
                [0.6, 0.4, 0.6],
                [np.nan, np.nan, np.nan],
            ]
        ),
        encoding=doe.EncodingEnum.NONE,
    )

    assert (
        a.compute_portion_of_total(
            a.SubSpaceMetricComputeContext(
                whole_doe=whole_doe,
                sub_space_doe=whole_doe.extract_points_and_dimensions(
                    [i == 2 for i in range(8)],
                    dim_set=["x3"],
                    encoding=doe.EncodingEnum.NONE,
                ),
            )
        )
        == 1.0 / 8.0
    )

    # Extract a Full-sub-space design from the whole DOE (rows 3,4,5,6)
    sub_space_doe = a.SubSpaceMetricComputeContext(
        whole_doe=whole_doe,
        sub_space_doe=whole_doe.extract_points_and_dimensions(
            [i in [3, 4, 5, 6] for i in range(8)],
            dim_set=["x1", "x2", "x3"],
            encoding=doe.EncodingEnum.NONE,
        ),
    )

    assert a.compute_portion_of_total(sub_space_doe) == 4.0 / 8.0
    assert a.compute_discrepancy(sub_space_doe) == discrepancy(
        np.array(
            [
                [0.3, 0.1, 0.7],
                [0.4, 0.2, 0.8],
                [0.5, 0.3, 0.9],
                [0.6, 0.4, 0.6],
            ]
        ),
        method="MD",
    )

    # TODO: incorporate hand computed values
    # to ensure the following are correct
    mipd = a.compute_min_point_distance(sub_space_doe)
    assert mipd > 0.0

    ard = a.compute_average_reciprocal_distance_projection(sub_space_doe)
    assert ard > 0.0

    mst_mean, mst_std = a.compute_mst_stats(sub_space_doe)
    assert mst_mean > 0.0
    assert mst_std > 0.0


def test_whole_min_distance_computation():
    """
    Tests the min distance computation for a DOE with nan values.

    Asserts
    -------
        the computed min distance is equal to the known min distance
    """
    space = s.InputSpace(
        dimensions=[
            s.Float(id="x1", lb=0.0, ub=10.0, nullable=True, portion_null=0.1),
            s.Float(id="x2", lb=0.0, ub=10.0, nullable=True, portion_null=0.1),
            s.Float(id="x3", lb=0.0, ub=10.0, nullable=True, portion_null=0.1),
        ]
    )

    whole_doe = doe.DesignOfExperiment(
        input_space=space,
        input_set_map={"x1": 0, "x2": 1, "x3": 2},
        input_sets=np.array(
            [[1.0, 2.0, np.nan], [4.0, np.nan, np.nan], [7.0, 8.0, 9.0]]
        ),
        encoding=doe.EncodingEnum.NONE,
    )

    min_d = a.compute_min_interpoint_dist(whole_doe, [], doe.EncodingEnum.NONE)

    # the minimum distance between the distance is between points 1 and 2 (1-based indexing)
    assert min_d == ((4.0 - 1.0) ** 2 + (1)) ** 0.5


def test_compute_min_projected_distance():
    """
    Tests the min projected distance computation for a DOE with nan values.

    Asserts
    -------
        the computed min distance is equal to the known min distance
    """
    space = s.InputSpace(
        dimensions=[
            s.Float(id="x1", lb=0.0, ub=10.0, nullable=True, portion_null=0.1),
            s.Float(id="x2", lb=0.0, ub=10.0, nullable=True, portion_null=0.1),
            s.Float(id="x3", lb=0.0, ub=10.0, nullable=True, portion_null=0.1),
        ]
    )

    whole_doe = doe.DesignOfExperiment(
        input_space=space,
        input_set_map={"x1": 0, "x2": 1, "x3": 2},
        input_sets=np.array(
            [[1.0, 2.0, np.nan], [4.0, np.nan, np.nan], [7.0, 8.0, 9.0]]
        ),
        encoding=doe.EncodingEnum.NONE,
    )

    min_d = a.compute_min_projected_distance(
        whole_doe, [], doe.EncodingEnum.NONE
    )

    # the minimum distance between the distance is between points 1 and 2 (1-based indexing)
    assert min_d == 4.0 - 1.0


def test_compute_min_projected_distance_with_composites():
    """
    Tests the min projected distance computation for a DOE with nan values.

    Asserts
    -------
        the computed min distance is equal to the known min distance
    """
    space = s.InputSpace(
        dimensions=[
            s.Float(id="x1", lb=0.0, ub=10.0, nullable=True, portion_null=0.1),
            s.Float(id="x2", lb=0.0, ub=10.0, nullable=True, portion_null=0.1),
            s.Float(id="x3", lb=0.0, ub=10.0, nullable=True, portion_null=0.1),
            s.Composite(
                id="c1",
                children=[
                    s.Float(
                        id="x4",
                        lb=0.0,
                        ub=10.0,
                        nullable=True,
                        portion_null=0.1,
                    ),
                    s.Float(
                        id="x5",
                        lb=0.0,
                        ub=10.0,
                        nullable=True,
                        portion_null=0.1,
                    ),
                ],
                nullable=True,
                portion_null=0.1,
            ),
        ]
    )

    whole_doe = doe.DesignOfExperiment(
        input_space=space,
        input_set_map={"x1": 0, "x2": 1, "x3": 2, "c1": 3, "x4": 4, "x5": 5},
        input_sets=np.array(
            [
                [1.0, 2.0, np.nan, 1.0, 3.0, 2.0],
                [4.0, np.nan, np.nan, 1.0, np.nan, 1.0],
                [7.0, 8.0, 9.0, np.nan, np.nan, np.nan],
            ]
        ),
        encoding=doe.EncodingEnum.NONE,
    )

    min_d = a.compute_min_projected_distance(
        whole_doe, [], doe.EncodingEnum.NONE
    )

    # the minimum distance between the distance is between points 1 and 2 (1-based indexing)
    assert min_d == 2.0 - 1.0


def test_compute_star_discrepancy():
    space = s.InputSpace(
        dimensions=[
            s.Float(id="x1", lb=0.0, ub=10.0, nullable=True, portion_null=0.1),
            s.Float(id="x2", lb=0.0, ub=10.0, nullable=True, portion_null=0.1),
            s.Float(id="x3", lb=0.0, ub=10.0, nullable=True, portion_null=0.1),
            s.Composite(
                id="c1",
                children=[
                    s.Float(
                        id="x4",
                        lb=0.0,
                        ub=10.0,
                        nullable=True,
                        portion_null=0.1,
                    ),
                    s.Float(
                        id="x5",
                        lb=0.0,
                        ub=10.0,
                        nullable=True,
                        portion_null=0.1,
                    ),
                ],
                nullable=True,
                portion_null=0.1,
            ),
        ]
    )

    whole_doe = doe.DesignOfExperiment(
        input_space=space,
        input_set_map={"x1": 0, "x2": 1, "x3": 2, "c1": 3, "x4": 4, "x5": 5},
        input_sets=np.array(
            [
                [0.0, 0.2, np.nan, 1.0, 0.95, 0.2],
                [0.3, np.nan, np.nan, 1.0, np.nan, 0.3],
                [0.9, 1.0, 0.7, np.nan, np.nan, np.nan],
            ]
        ),
        encoding=doe.EncodingEnum.ZERO_ONE_NULL_ENCODING,
    )
    star_discrep = a.compute_star_discrepancy(whole_doe)

    assert star_discrep is not None
    assert star_discrep > 0.0


def test_compute_max_pro():
    space = s.InputSpace(
        dimensions=[
            s.Float(id="x1", lb=0.0, ub=10.0, nullable=True, portion_null=0.1),
            s.Float(id="x2", lb=0.0, ub=10.0, nullable=True, portion_null=0.1),
            s.Float(id="x3", lb=0.0, ub=10.0, nullable=True, portion_null=0.1),
            s.Composite(
                id="c1",
                children=[
                    s.Float(
                        id="x4",
                        lb=0.0,
                        ub=10.0,
                        nullable=True,
                        portion_null=0.1,
                    ),
                    s.Float(
                        id="x5",
                        lb=0.0,
                        ub=10.0,
                        nullable=True,
                        portion_null=0.1,
                    ),
                ],
                nullable=True,
                portion_null=0.1,
            ),
        ]
    )

    whole_doe = doe.DesignOfExperiment(
        input_space=space,
        input_set_map={"x1": 0, "x2": 1, "x3": 2, "c1": 3, "x4": 4, "x5": 5},
        input_sets=np.array(
            [
                [0.0, 0.2, np.nan, 1.0, 0.95, 0.2],
                [0.3, np.nan, np.nan, 1.0, np.nan, 0.3],
                [0.9, 1.0, 0.7, np.nan, np.nan, np.nan],
            ]
        ),
        encoding=doe.EncodingEnum.ZERO_ONE_NULL_ENCODING,
    )
    max_pro_value = a.compute_max_pro(whole_doe)

    assert max_pro_value is not None
    assert max_pro_value > 0.0


def test_compute_average_dim_distance():
    """
    Tests the computation of the average miniumn distance for dimensions.

    Asserts
    -------
        the proper average is computed
    """
    space = s.InputSpace(
        dimensions=[
            s.Float(id="x1", lb=0.0, ub=10.0, nullable=True, portion_null=0.1),
            s.Float(id="x2", lb=0.0, ub=10.0, nullable=True, portion_null=0.1),
            s.Float(id="x3", lb=0.0, ub=10.0, nullable=True, portion_null=0.1),
            s.Composite(
                id="c1",
                children=[
                    s.Float(
                        id="x4",
                        lb=0.0,
                        ub=10.0,
                        nullable=True,
                        portion_null=0.1,
                    ),
                    s.Float(
                        id="x5",
                        lb=0.0,
                        ub=10.0,
                        nullable=True,
                        portion_null=0.1,
                    ),
                ],
                nullable=True,
                portion_null=0.1,
            ),
        ]
    )
    design = doe.DesignOfExperiment(
        input_space=space,
        input_set_map={"x1": 0, "x2": 1, "x3": 2, "c1": 3, "x4": 4, "x5": 5},
        input_sets=np.array(
            [
                [0.0, 0.2, np.nan, 1.0, 0.95, 0.2],
                [0.3, np.nan, np.nan, 1.0, np.nan, 0.3],
                [0.9, 1.0, 0.7, np.nan, np.nan, np.nan],
            ]
        ),
        encoding=doe.EncodingEnum.ZERO_ONE_NULL_ENCODING,
    )

    actual_average = a.compute_average_dim_dist(design)

    expected_average = (0.3 + 0.8 + 0.0 + 0.1) / 4

    assert actual_average == expected_average
