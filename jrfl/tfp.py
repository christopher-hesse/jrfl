"""
Based on Tensorflow Probability https://www.tensorflow.org/probability
"""

import abc

import jax.random
import jax.lax
import jax.numpy as jp
import jax.scipy.special
from jax.experimental.stax import logsoftmax
from jax.scipy.stats import norm
from .check import assert_array
from .trfl import broadcast_index


class Distribution(abc.ABC):
    """
    https://www.tensorflow.org/api_docs/python/tf/distributions/Distribution
    """

    @abc.abstractmethod
    def cdf(self, value):
        pass

    @abc.abstractmethod
    def covariance(self):
        pass

    @abc.abstractmethod
    def entropy(self):
        pass

    @abc.abstractmethod
    def kl_divergence(self, other):
        pass

    @abc.abstractmethod
    def mean(self):
        pass

    @abc.abstractmethod
    def mode(self):
        pass

    @abc.abstractmethod
    def prob(self, value):
        pass

    @abc.abstractmethod
    def quantile(self, value):
        pass

    @abc.abstractmethod
    def sample(self, sample_shape, seed):
        pass

    @abc.abstractmethod
    def stddev(self):
        pass

    @abc.abstractmethod
    def variance(self):
        pass

    def cross_entropy(self, other):
        return self.entropy() + self.kl_divergence(other)

    def log_cdf(self, value):
        return jp.log(self.cdf(value))

    def log_prob(self, value):
        return jp.log(self.prob(value))

    def survival_function(self, value):
        return 1 - self.cdf(value)


class Independent(Distribution):
    """
    https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Independent
    https://colab.research.google.com/github/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Understanding_TensorFlow_Distributions_Shapes.ipynb#scrollTo=1iR23BMBrG6z
    """

    def __init__(self, distribution, reinterpreted_batch_ndims=0):
        self.dist = distribution
        self.batch_shape = self.dist.batch_shape[:-reinterpreted_batch_ndims]
        self.event_shape = (
            self.dist.batch_shape[-reinterpreted_batch_ndims:] + self.dist.event_shape
        )

    def _reduce_event(self, method, reduce_fn, value):
        assert_array(value, shape=(...,) + self.batch_shape + self.event_shape)
        result = method(value)
        # reduce along event axis
        reduced_result = result.reshape(value.shape[: -len(self.event_shape)] + (-1,))
        return reduce_fn(reduced_result, axis=-1)

    def cdf(self, value):
        return self._reduce_event(self.dist.cdf, jp.prod, value)

    def log_cdf(self, value):
        return self._reduce_event(self.dist.log_cdf, jp.sum, value)

    def covariance(self):
        return self.dist.covariance()

    def entropy(self):
        return self.dist.entropy()

    def kl_divergence(self, other):
        return self.dist.kl_divergence(other)

    def mean(self):
        return self.dist.mean()

    def mode(self):
        return self.dist.mode()

    def prob(self, value):
        return self._reduce_event(self.dist.prob, jp.prod, value)

    def log_prob(self, value):
        return self._reduce_event(self.dist.log_prob, jp.sum, value)

    def quantile(self, value):
        return self.dist.quantile(value)

    def sample(self, sample_shape, seed):
        return self.dist.sample(sample_shape, seed)

    def stddev(self):
        return self.dist.stddev()

    def variance(self):
        return self.dist.variance()


class Categorical(Distribution):
    """
    https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Categorical
    """

    def __init__(self, probs=None, logits=None):
        if logits is None:
            assert probs is not None
            self.log_probs = jp.log(probs)
            self._logits = self.log_probs
        else:
            assert probs is None
            self.log_probs = logsoftmax(logits, axis=-1)
            self._logits = logits
        self._probs = jp.exp(self.log_probs)
        self.batch_shape = self.log_probs.shape[:-1]
        self.event_shape = ()
        self._cdf_probs = jp.cumsum(self._probs, axis=-1) - self._probs

    def cdf(self, value):
        return broadcast_index(self._cdf_probs, value)

    def covariance(self):
        raise NotImplementedError

    def entropy(self):
        return -jp.sum(self._probs * self.log_probs, axis=-1)

    def kl_divergence(self, other):
        assert isinstance(other, Categorical)
        return -jp.sum(self._probs * (other.log_probs - self.log_probs), axis=-1)

    def mean(self):
        raise NotImplementedError

    def mode(self):
        return jp.argmax(self._probs, axis=-1)

    def prob(self, value):
        return broadcast_index(self._probs, value)

    def quantile(self, value):
        raise NotImplementedError

    def sample(self, sample_shape, seed):
        return jp.argmax(
            self._logits
            + jax.random.gumbel(seed, shape=sample_shape + self._logits.shape),
            axis=-1,
        )

    def stddev(self):
        raise NotImplementedError

    def variance(self):
        raise NotImplementedError


class Normal(Distribution):
    """
    https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Normal
    """

    def __init__(self, loc, scale):
        self.batch_shape = scale.shape
        self.event_shape = ()
        assert_array(loc, shape=scale.shape, dtypes=[scale.dtype])
        self._loc = loc
        self._scale = scale

    def cdf(self, value):
        return norm.cdf(value, loc=self._loc, scale=self._scale)

    def log_cdf(self, value):
        return norm.logcdf(value, loc=self._loc, scale=self._scale)

    def covariance(self):
        raise NotImplementedError

    def entropy(self):
        return 0.5 * jp.log(2 * jp.pi * jp.e * self.variance())

    def kl_divergence(self, other):
        assert isinstance(other, Normal)
        return (self.mean() - other.mean()) ** 2 / (2 * other.variance()) + 0.5 * (
            self.variance() / other.variance()
            - 1
            - jp.log(self.variance() / other.variance())
        )

    def mean(self):
        return self._loc

    def mode(self):
        return self._loc

    def prob(self, value):
        assert_array(value, shape=(...,) + self.batch_shape)
        return norm.pdf(value, loc=self._loc, scale=self._scale)

    def log_prob(self, value):
        assert_array(value, shape=(...,) + self.batch_shape)
        return norm.logpdf(value, loc=self._loc, scale=self._scale)

    def quantile(self, value):
        return (
            jp.sqrt(2) * jax.scipy.special.erfinv(2 * value - 1) * self._scale
            + self._loc
        )

    def sample(self, sample_shape, seed):
        return (
            jax.random.normal(seed, shape=sample_shape + self._loc.shape) * self._scale
            + self._loc
        )

    def stddev(self):
        return self._scale

    def variance(self):
        return self._scale ** 2
