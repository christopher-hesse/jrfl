import numpy as np

from .tf import sparse_softmax_cross_entropy_with_logits
from .check import assert_allclose


def test_sparse_softmax_cross_entropy_with_logits():
    labels = np.array([[0, 1], [1, 2]], dtype=np.int32)
    logits = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32
    )

    # from tf.nn.sparse_softmax_cross_entropy_with_logits()
    ref = np.array([[2.407606, 1.4076059], [1.4076059, 0.40760595]], dtype=np.float32)
    out = sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    assert out.shape == labels.shape
    assert_allclose(ref, out)
