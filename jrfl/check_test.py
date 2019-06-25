import numpy as np
import pytest

from .check import assert_array


def test_assert_array():
    arr = np.array([1, 2, 3], dtype=np.int64)
    assert_array(arr, shape=(3,), dtypes=(np.int64,))

    with pytest.raises(AssertionError):
        assert_array(arr, shape=(2,))

    with pytest.raises(AssertionError):
        assert_array(arr, dtypes=(np.int32,))

    arr2 = np.zeros(shape=(1, 2, 3, 4, 5))

    assert_array(arr2, shape=(1, None, None, 4, 5))
    assert_array(arr2, shape=(1, ..., 4, 5))
    assert_array(arr2, shape=(1, ...))
    assert_array(arr2, shape=(1, 2, 3, 4, 5, ...))
    assert_array(arr2, shape=(..., 1, 2, 3, 4, 5))
    assert_array(arr2, shape=(..., 5))

    with pytest.raises(Exception):
        assert_array(arr2, shape=(1, ..., ..., 5))

    with pytest.raises(AssertionError):
        assert_array(arr2, shape=(2, ...))

    with pytest.raises(AssertionError):
        assert_array(arr2, shape=(..., 6))
