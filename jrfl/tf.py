import jax.numpy as jp
from jax.experimental.stax import logsoftmax

from .check import assert_array


def sparse_softmax_cross_entropy_with_logits(*, labels, logits):
    """
    https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits
    """
    assert labels.shape == logits.shape[:-1]
    assert_array(labels, dtypes=(jp.int32,))
    assert_array(logits, dtypes=(jp.float32,))
    log_probs = logsoftmax(logits, axis=-1)
    chosen_log_probs = jp.squeeze(
        jp.take_along_axis(log_probs, jp.expand_dims(labels, axis=-1), axis=-1), axis=-1
    )
    return -chosen_log_probs
