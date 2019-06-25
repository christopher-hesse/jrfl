# jrfl

This is a port of [TRFL](https://github.com/deepmind/trfl) to [JAX](https://github.com/google/jax).

The included functions are:

* `discrete_policy_gradient`
* `discrete_policy_gradient_loss`
* `policy_gradient`
* `policy_gradient_loss
* `scan_discounted_sum`
* `batched_index`

See the [TRFL Documentation](https://github.com/deepmind/trfl/blob/master/docs/trfl.md) for the documentation for each function.

There are a few classes implementing [Tensorflow Probability](https://www.tensorflow.org/probability) interfaces since some of the TRFL functions expect them.  These are:

* `Distribution`
* `Independent`
* `Normal`
* `Categorical`

A few new functions included in this package are:

* `assert_array` - check array shape and dtype
* `broadcast_index` - a more general `batched_index`