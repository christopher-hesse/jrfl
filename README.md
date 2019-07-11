# jrfl

This is a port of [TRFL](https://github.com/deepmind/trfl) to [JAX](https://github.com/google/jax).

The included functions are:

* [`discrete_policy_gradient`](https://github.com/deepmind/trfl/blob/master/docs/trfl.md#discrete_policy_gradientpolicy_logits-actions-action_values-namediscrete_policy_gradient)
* [`discrete_policy_gradient_loss`](https://github.com/deepmind/trfl/blob/master/docs/trfl.md#discrete_policy_gradient_losspolicy_logits-actions-action_values-namediscrete_policy_gradient_loss)
* [`policy_gradient`](https://github.com/deepmind/trfl/blob/master/docs/trfl.md#policy_gradientpolicies-actions-action_values-policy_varsnone-namepolicy_gradient)
* [`policy_gradient_loss`](https://github.com/deepmind/trfl/blob/master/docs/trfl.md#policy_gradient_losspolicies-actions-action_values-policy_varsnone-namepolicy_gradient_loss)
* [`scan_discounted_sum`](https://github.com/deepmind/trfl/blob/master/docs/trfl.md#scan_discounted_sumsequence-decay-initial_value-reversefalse-sequence_lengthsnone-back_proptrue-namescan_discounted_sum)
* [`batched_index`](https://github.com/deepmind/trfl/blob/master/docs/trfl.md#batched_indexvalues-indices)

There are a few classes implementing [Tensorflow Probability](https://www.tensorflow.org/probability) interfaces since some of the TRFL functions expect them.  These are:

* [`Distribution`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution)
* [`Independent`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Independent)
* [`Normal`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Normal)
* [`Categorical`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Categorical)

A few new functions included in this package are:

* `assert_array` - check array shape and dtype
* `broadcast_index` - a more general `batched_index`
* `PRNGSequence` - an infinite iterator of `PRNGKey`s