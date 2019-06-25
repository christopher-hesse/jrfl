import numpy as np


def assert_array(arr, shape=None, dtypes=None):
    """
    Assert various properties of an ndarray
    """
    if shape is not None:
        if shape.count(...) == 0:
            assert len(arr.shape) == len(
                shape
            ), f"invalid shape actual={arr.shape} desired={shape}"
            for a, d in zip(arr.shape, shape):
                if d is not None:
                    assert a == d, f"invalid shape actual={arr.shape} desired={shape}"
        elif shape.count(...) == 1:
            offset = 0
            for i, d in enumerate(shape):
                if d is ...:
                    dist_from_end = len(shape) - 1 - i
                    offset = len(arr.shape) - dist_from_end
                    continue
                assert d is None or arr.shape[offset] == d
                offset += 1
        else:
            raise Exception("too many ellipses")
    if dtypes is not None:
        assert arr.dtype in dtypes, arr.dtype


def assert_allclose(a, b, rtol=1e-5, atol=1e-8):
    """
    Assert that two arrays have the same shape and similiar values
    """
    assert a.shape == b.shape, f"shape mismatch a={a.shape} b={b.shape}"
    assert np.allclose(a, b, rtol=rtol, atol=atol)
