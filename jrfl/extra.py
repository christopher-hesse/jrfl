import jax.numpy as jp

from .check import assert_array


def broadcast_index(values, indices):
    """
    This is a more general form of batched_index

    The last dimension of `indices` contains indices for the last dimension of `values`,
    and the shape of `indices` must have the same trailing dimensions as `values`
    (excluding the final index `num_values` dimension).

    `values` will be broadcast to the correct shape to match up with `indices`, then for
    each set of `num_values` in the final dimension of the broadcast array, the one indicated
    by the corresponding value in `indices` will be chosen.

    Args:
        values: tensor of shape [..., num_values]
        indices: tensor of shape [...] + values.shape[:-1]

    Returns:
        tensor of shape `indices.shape` but with values from `values` instead of the indices
    """
    assert_array(indices, shape=(...,) + values.shape[:-1])
    indexed_values = jp.take_along_axis(
        values.reshape((1,) + values.shape),
        indices.reshape((-1,) + values.shape[:-1] + (1,)),
        axis=-1,
    )
    flat_result = jp.squeeze(indexed_values, axis=-1)
    return flat_result.reshape(indices.shape)
