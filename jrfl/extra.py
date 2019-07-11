import functools

import jax.numpy as jp
import jax
import jax.random

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


class PRNGSequence:
    """
    An iterator that returns a new PRNG key each time next() is called
    """

    def __init__(self, seed):
        self._key = jax.random.PRNGKey(seed)
        self._count = 0

    def __iter__(self):
        return self

    @functools.partial(jax.jit, static_argnums=(0,))
    def _next_key(self, count):
        return jax.random.fold_in(self._key, count)

    def __next__(self):
        key = self._next_key(self._count)
        self._count += 1
        return key
