import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
import jax.random

from .tfp import Categorical, Normal, Independent
from .check import assert_allclose


@pytest.mark.parametrize("name", ["categorical-probs", "categorical-logits", "normal"])
def test_compare(name):
    rand = np.random.RandomState(0)

    def rand_probs():
        return rand.uniform(size=(2, 3, 4)).astype(np.float32)

    if name.startswith("categorical"):

        def normalize(p):
            return p / np.expand_dims(np.sum(p, axis=-1), axis=-1)

        if name == "categorical-probs":
            kwargs1 = {"probs": normalize(rand_probs())}
            kwargs2 = {"probs": normalize(rand_probs())}
        elif name == "categorical-logits":
            kwargs1 = {"logits": rand_probs()}
            kwargs2 = {"logits": rand_probs()}
        else:
            raise Exception("invalid name")

        dist = Categorical(**kwargs1)
        dist2 = Categorical(**kwargs2)
        tf_dist = tfp.distributions.Categorical(**kwargs1)
        tf_dist2 = tfp.distributions.Categorical(**kwargs2)
    elif name == "normal":
        l1 = rand_probs()
        l2 = rand_probs()
        s1 = rand_probs()
        s2 = rand_probs()
        dist = Normal(loc=l1, scale=s1)
        dist2 = Normal(loc=l2, scale=s2)
        tf_dist = tfp.distributions.Normal(loc=l1, scale=s1)
        tf_dist2 = tfp.distributions.Normal(loc=l2, scale=s2)
    else:
        raise Exception("invalid distribution")

    assert dist.batch_shape == tf_dist.batch_shape
    assert dist.event_shape == tf_dist.event_shape

    tf.random.set_random_seed(0)
    seed = jax.random.PRNGKey(0)
    with tf.Session() as sess:
        assert_allclose(dist.mode(), sess.run(tf_dist.mode()), rtol=0.01)
        assert_allclose(dist.entropy(), sess.run(tf_dist.entropy()), rtol=0.01)

        sample_shape = (1, 5, 9)
        for values in [
            sess.run(tf_dist.sample(sample_shape)),
            dist.sample(sample_shape=sample_shape, seed=seed),
        ]:
            if name == "normal":
                assert_allclose(sess.run(tf_dist.stddev()), dist.stddev(), rtol=0.01)
                assert_allclose(sess.run(tf_dist.mean()), dist.mean(), rtol=0.01)
                assert_allclose(
                    sess.run(tf_dist.variance()), dist.variance(), rtol=0.01
                )

            assert_allclose(sess.run(tf_dist.cdf(values)), dist.cdf(values), atol=1e-5)

            assert_allclose(
                sess.run(tf_dist.survival_function(values)),
                dist.survival_function(values),
                atol=1e-5,
            )

            assert_allclose(
                sess.run(tf_dist.prob(values)), dist.prob(values), atol=1e-5
            )

            assert_allclose(
                sess.run(tf_dist.kl_divergence(tf_dist)), dist.kl_divergence(dist)
            )
            assert_allclose(
                dist.kl_divergence(dist), np.zeros_like(dist.kl_divergence(dist))
            )

            assert_allclose(
                sess.run(tf_dist.kl_divergence(tf_dist2)),
                dist.kl_divergence(dist2),
                atol=1e-5,
            )

            assert_allclose(
                sess.run(tf_dist.cross_entropy(tf_dist2)), dist.cross_entropy(dist2)
            )

        if name == "normal":
            quantiles = (np.arange(np.prod(dist.batch_shape) * 10) / 1000).reshape(
                (-1,) + dist.batch_shape
            )
            assert_allclose(
                sess.run(tf_dist.quantile(quantiles)),
                dist.quantile(quantiles),
                atol=1e-5,
            )


def test_independent():
    rand = np.random.RandomState(0)
    seed = jax.random.PRNGKey(0)

    def rand_probs():
        return rand.uniform(size=(2, 3, 4)).astype(np.float32)

    loc = rand_probs()
    scale = rand_probs()

    tf_ind = tfp.distributions.Independent(
        distribution=tfp.distributions.Normal(loc=loc, scale=scale),
        reinterpreted_batch_ndims=1,
    )
    ind = Independent(
        distribution=Normal(loc=loc, scale=scale), reinterpreted_batch_ndims=1
    )

    assert tf_ind.batch_shape == ind.batch_shape
    assert tf_ind.event_shape == ind.event_shape

    with tf.Session() as sess:
        tf_sample = sess.run(tf_ind.sample(sample_shape=(5, 4, 3)))
        sample = ind.sample(sample_shape=(5, 4, 3), seed=seed)
        assert tf_sample.shape == sample.shape

        assert_allclose(sess.run(tf_ind.prob(sample)), ind.prob(sample), atol=1e-5)

        assert_allclose(
            sess.run(tf_ind.log_prob(sample)), ind.log_prob(sample), atol=1e-5
        )

        assert_allclose(sess.run(tf_ind.cdf(sample)), ind.cdf(sample), atol=1e-5)

        assert_allclose(
            sess.run(tf_ind.log_cdf(sample)), ind.log_cdf(sample), atol=1e-5
        )
