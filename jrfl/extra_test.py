import itertools

import numpy as np

from .check import assert_allclose
from .extra import broadcast_index, PRNGSequence


def test_broadcast_index():
    values = np.array(
        [
            [1.1, 1.2, 1.3],
            [1.4, 1.5, 1.6],
            [2.1, 2.2, 2.3],
            [2.4, 2.5, 2.6],
            [3.1, 3.2, 3.3],
            [3.4, 3.5, 3.6],
            [4.1, 4.2, 4.3],
            [4.4, 4.5, 4.6],
        ]
    )
    action_indices = np.array(
        [[[[0, 2, 1, 0, 2, 1, 0, 2], [0, 2, 1, 0, 2, 1, 0, 2]]]], dtype=np.int32
    )
    result = broadcast_index(values, action_indices)
    expected_result = np.array(
        [
            [
                [
                    [1.1, 1.6, 2.2, 2.4, 3.3, 3.5, 4.1, 4.6],
                    [1.1, 1.6, 2.2, 2.4, 3.3, 3.5, 4.1, 4.6],
                ]
            ]
        ]
    )
    np.testing.assert_allclose(result, expected_result)

    values = np.array(
        [
            [
                [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]],
                [[2.1, 2.2, 2.3], [2.4, 2.5, 2.6]],
                [[3.1, 3.2, 3.3], [3.4, 3.5, 3.6]],
                [[4.1, 4.2, 4.3], [4.4, 4.5, 4.6]],
            ],
            [
                [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]],
                [[2.1, 2.2, 2.3], [2.4, 2.5, 2.6]],
                [[3.1, 3.2, 3.3], [3.4, 3.5, 3.6]],
                [[4.1, 4.2, 4.3], [4.4, 4.5, 4.6]],
            ],
        ]
    )
    action_indices = np.array(
        [[[0, 2], [1, 0], [2, 1], [0, 2]], [[0, 2], [1, 0], [2, 1], [0, 2]]],
        dtype=np.int32,
    )
    result = broadcast_index(values, action_indices)
    expected_result = np.array(
        [
            [[1.1, 1.6], [2.2, 2.4], [3.3, 3.5], [4.1, 4.6]],
            [[1.1, 1.6], [2.2, 2.4], [3.3, 3.5], [4.1, 4.6]],
        ]
    )
    assert_allclose(result, expected_result)


def test_prng_sequence():
    v1 = list(itertools.islice(PRNGSequence(0), 10))
    v2 = list(itertools.islice(PRNGSequence(0), 10))
    v3 = list(itertools.islice(PRNGSequence(1), 10))
    assert np.array_equal(v1, v2)
    assert not np.array_equal(v1, v3)
