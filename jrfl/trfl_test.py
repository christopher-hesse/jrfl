"""
Based on test cases from https://github.com/deepmind/trfl
"""

import numpy as np
import jax
from jax import grad
import jax.tree_util
import jax.numpy as jp
import pytest

from .check import assert_array, assert_allclose

from .trfl import (
    discrete_policy_gradient,
    discrete_policy_gradient_loss,
    scan_discounted_sum,
    policy_gradient,
    policy_gradient_loss,
    batched_index,
)


def test_discrete_policy_gradient():
    policy_logits = np.array([[0, 1], [0, 1], [1, 1], [0, 100]], dtype=np.float32)
    action_values = np.array([0, 1, 2, 1], dtype=np.float32)
    actions = np.array([0, 0, 1, 1], dtype=np.int32)
    loss = discrete_policy_gradient(
        policy_logits=policy_logits, actions=actions, action_values=action_values
    )
    assert loss.shape == (4,)

    # Calculate the targets with:
    #     loss = action_value*(-logits[action] + log(sum_a(exp(logits[a]))))
    #  The final case (with large logits), runs out of precision and gets
    #  truncated to 0, but isn't `nan`.
    assert_allclose(loss, np.array([0, 1.313262, 1.386294, 0]))

    def loss_fun(policy_logits, actions, action_values):
        loss = discrete_policy_gradient(
            policy_logits=policy_logits, actions=actions, action_values=action_values
        )
        return jp.sum(loss)

    grad_fn = grad(loss_fun)
    grad_policy_logits = grad_fn(policy_logits, actions, action_values)
    assert_allclose(
        grad_policy_logits,
        np.array([[0, 0], [-0.731, 0.731], [1, -1], [0, 0]]),
        atol=1e-4,
    )

    # it's an error to get the gradient with respect to integer inputs
    # grad_fn1 = grad(loss_fun, argnums=1)
    # grad_actions = grad_fn1(policy_logits, actions, action_values)
    # assert np.array_equal(grad_actions, np.zeros_like(grad_actions))

    grad_fn2 = grad(loss_fun, argnums=2)
    grad_action_values = grad_fn2(policy_logits, actions, action_values)
    assert np.array_equal(grad_action_values, np.zeros_like(grad_action_values))


@pytest.mark.parametrize("is_multi_actions", [True, False])
def test_discrete_policy_gradient_loss(is_multi_actions):
    policy_logits = np.array([[[0, 1], [0, 1]], [[1, 1], [0, 100]]], dtype=np.float32)
    actions = np.array([[0, 0], [1, 1]], dtype=np.int32)

    if is_multi_actions:
        num_action_components = 3
        policy_logits_nest = [policy_logits for _ in range(num_action_components)]
        actions_nest = [actions for _ in range(num_action_components)]
    else:
        num_action_components = 1
        policy_logits_nest = [policy_logits]
        actions_nest = [actions]

    action_values = np.array([[0, 1], [2, 1]], dtype=np.float32)

    loss = discrete_policy_gradient_loss(
        policy_logits=policy_logits_nest,
        actions=actions_nest,
        action_values=action_values,
    )
    assert loss.shape == (2,)

    # computed by summing expected losses from DiscretePolicyGradientTest
    # over the two sequences of length two which I've split the batch
    # into:
    assert_allclose(loss, num_action_components * np.array([1.386294, 1.313262]))

    def loss_fun(policy_logits, actions, action_values):
        loss = discrete_policy_gradient_loss(
            policy_logits=policy_logits, actions=actions, action_values=action_values
        )
        return jp.sum(loss)

    grad_fn = grad(loss_fun)
    g = grad_fn(policy_logits_nest, actions_nest, action_values)
    for sg in g:
        assert_allclose(
            sg, np.array([[[0, 0], [-0.731, 0.731]], [[1, -1], [0, 0]]]), atol=1e-4
        )

    # it's an error to get the gradient with respect to integer inputs
    # for argnums in [1, 2]:
    for argnums in [2]:
        grad_fn = grad(loss_fun, argnums=argnums)
        g = grad_fn(policy_logits_nest, actions_nest, action_values)
        for sg in g:
            assert np.array_equal(sg, np.zeros_like(sg))


class MockDistribution:
    def __init__(self, batch_shape, parameter):
        self.batch_shape = batch_shape
        self.event_shape = ()
        self._parameter = parameter
        self._entropy = np.arange(np.prod(batch_shape)).reshape(batch_shape)
        self._entropy *= parameter * parameter

    def log_prob(self, actions):
        return self._parameter * actions

    def entropy(self):
        return self._entropy


def setup_pgops_mock(sequence_length=3, batch_size=2, num_policies=3):
    t, b = sequence_length, batch_size
    policies = [MockDistribution((t, b), i + 1) for i in range(num_policies)]
    actions = [
        np.arange(t * b, dtype=np.int32).reshape((t, b)) for i in range(num_policies)
    ]
    if num_policies == 1:
        policies, actions = policies[0], actions[0]

    def entropy_scale_op(policies):
        return len(jax.tree_util.tree_flatten(policies))

    return policies, actions, entropy_scale_op


def test_policy_gradient():
    policies, actions, _ = setup_pgops_mock(
        sequence_length=3, batch_size=2, num_policies=1
    )
    action_values = np.array([[-0.5, 0.5], [-1.0, 0.5], [1.5, -0.5]])
    loss = policy_gradient(
        policies=policies, actions=actions, action_values=action_values
    )
    expected_loss = np.array([[0.0, -0.5], [2.0, -1.5], [-6.0, 2.5]])
    assert_allclose(loss, expected_loss)


def test_policy_gradient_loss():
    policies, actions, _ = setup_pgops_mock(
        sequence_length=3, batch_size=2, num_policies=3
    )
    action_values = np.array([[-0.5, 0.5], [-1.0, 0.5], [1.5, -0.5]])
    loss = policy_gradient_loss(
        policies=policies, actions=actions, action_values=action_values
    )
    expected_loss = np.array([-24.0, 3.0])
    assert_allclose(loss, expected_loss)


def test_scan_discounted_sum_shapes():
    sequence_in = np.zeros(shape=(1647, 2001), dtype=np.float32)
    decays_in = np.zeros(shape=(1647, 2001), dtype=np.float32)
    bootstrap = np.zeros(shape=(2001,), dtype=np.float32)
    result = scan_discounted_sum(
        sequence=sequence_in, decay=decays_in, initial_value=bootstrap
    )
    assert_array(result, shape=sequence_in.shape)

    sequence_in = np.zeros(shape=(4, 8, 15, 16, 23, 42), dtype=np.float32)
    decays_in = np.zeros(shape=(4, 8, 15, 16, 23, 42), dtype=np.float32)
    bootstrap = np.zeros(shape=(8, 15, 16, 23, 42), dtype=np.float32)
    result = scan_discounted_sum(
        sequence=sequence_in, decay=decays_in, initial_value=bootstrap
    )
    assert_array(result, shape=sequence_in.shape)


def test_scan_discounted_sum_decays():
    sequence = [[3, 1, 5, 2, 1], [-1.7, 1.2, 2.3, 0, 1]]
    decays = [[0.5, 0.9, 1.0, 0.1, 0.5], [0.9, 0.5, 0.0, 2, 0.8]]
    # We use transpose because it is easier to define the input data in
    # BxT (batch x time) form, while scan_discounted_sum assumes TxB form.
    sequence_in = np.transpose(np.array(sequence, dtype=np.float32))
    decays_in = np.transpose(np.array(decays, dtype=np.float32))
    bootstrap = np.array([0, 1.5], dtype=np.float32)
    result = scan_discounted_sum(
        sequence=sequence_in, decay=decays_in, initial_value=bootstrap
    )

    expected_result = np.array(
        [
            [
                3,
                3 * 0.9 + 1,
                (3 * 0.9 + 1) * 1.0 + 5,
                ((3 * 0.9 + 1) * 1.0 + 5) * 0.1 + 2,
                (((3 * 0.9 + 1) * 1.0 + 5) * 0.1 + 2) * 0.5 + 1,
            ],
            [
                -1.7 + 1.5 * 0.9,
                (-1.7 + 1.5 * 0.9) * 0.5 + 1.2,
                ((-1.7 + 1.5 * 0.9) * 0.5 + 1.2) * 0.0 + 2.3,
                (((-1.7 + 1.5 * 0.9) * 0.5 + 1.2) * 0.0 + 2.3) * 2 + 0,
                ((((-1.7 + 1.5 * 0.9) * 0.5 + 1.2) * 0.0 + 2.3) * 2 + 0) * 0.8 + 1,
            ],
        ],
        dtype=np.float32,
    )
    assert_allclose(np.transpose(expected_result), result)


def test_batched_index():
    values = np.array(
        [
            [1.1, 1.2, 1.3],
            [1.4, 1.5, 1.6],
            [2.1, 2.2, 2.3],
            [2.4, 2.5, 2.6],
            [3.1, 3.2, 3.3],
            [3.4, 3.5, 3.6],
            [4.1, 4.2, 4.3],
            [4.4, 4.5, 4.6],
        ]
    )
    action_indices = np.array(
        [[0, 2, 1, 0, 2, 1, 0, 2], [0, 2, 1, 0, 2, 1, 0, 2]], dtype=np.int32
    )
    action_indices = np.array([0, 2, 1, 0, 2, 1, 0, 2], dtype=np.int32)
    result = batched_index(values, action_indices)
    expected_result = np.array([1.1, 1.6, 2.2, 2.4, 3.3, 3.5, 4.1, 4.6])
    np.testing.assert_allclose(result, expected_result)

    values = np.array(
        [
            [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]],
            [[2.1, 2.2, 2.3], [2.4, 2.5, 2.6]],
            [[3.1, 3.2, 3.3], [3.4, 3.5, 3.6]],
            [[4.1, 4.2, 4.3], [4.4, 4.5, 4.6]],
        ]
    )
    action_indices = np.array([[0, 2], [1, 0], [2, 1], [0, 2]], dtype=np.int32)
    result = batched_index(values, action_indices)
    expected_result = np.array([[1.1, 1.6], [2.2, 2.4], [3.3, 3.5], [4.1, 4.6]])
    assert_allclose(result, expected_result)
