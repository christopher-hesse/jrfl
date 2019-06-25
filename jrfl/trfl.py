"""
Function signatures are from https://github.com/deepmind/trfl
"""

import jax.numpy as jp
import jax

from .check import assert_array
from .extra import broadcast_index
from .tf import sparse_softmax_cross_entropy_with_logits


def discrete_policy_gradient(*, policy_logits, actions, action_values):
    """
    https://github.com/deepmind/trfl/blob/master/docs/trfl.md#discrete_policy_gradientpolicy_logits-actions-action_values-namediscrete_policy_gradient
    """
    actions = jax.lax.stop_gradient(actions)
    action_values = jax.lax.stop_gradient(action_values)
    ce = sparse_softmax_cross_entropy_with_logits(labels=actions, logits=policy_logits)
    return ce * action_values


def discrete_policy_gradient_loss(*, policy_logits, actions, action_values):
    """
    https://github.com/deepmind/trfl/blob/master/docs/trfl.md#discrete_policy_gradient_losspolicy_logits-actions-action_values-namediscrete_policy_gradient_loss
    """
    flat_policy_logits, _ = jax.tree_util.tree_flatten(policy_logits)
    flat_actions, _ = jax.tree_util.tree_flatten(actions)
    assert len(flat_policy_logits) == len(flat_actions)
    num_action_components = len(flat_actions)
    num_timesteps, num_batch = action_values.shape
    seqs = []
    for scalar_policy_logits, scalar_actions in zip(flat_policy_logits, flat_actions):
        assert_array(
            scalar_policy_logits,
            shape=(num_timesteps, num_batch, None),
            dtypes=(jp.float32,),
        )
        assert_array(
            scalar_actions, shape=(num_timesteps, num_batch), dtypes=(jp.int32,)
        )
        seq = discrete_policy_gradient(
            policy_logits=scalar_policy_logits,
            actions=scalar_actions,
            action_values=action_values,
        )
        assert_array(seq, shape=(num_timesteps, num_batch))
        seqs.append(seq)
    stacked_seqs = jp.stack(seqs)
    assert_array(stacked_seqs, shape=(num_action_components, num_timesteps, num_batch))
    return jp.sum(jp.sum(stacked_seqs, axis=0), axis=0)


def policy_gradient(*, policies, actions, action_values):
    """
    https://github.com/deepmind/trfl/blob/master/docs/trfl.md#policy_gradientpolicies-actions-action_values-policy_varsnone-namepolicy_gradient
    """
    actions = jax.lax.stop_gradient(actions)
    action_values = jax.lax.stop_gradient(action_values)
    assert_array(
        actions,
        shape=policies.batch_shape + policies.event_shape,
        dtypes=(jp.int32, jp.float32),
    )
    assert_array(action_values, shape=policies.batch_shape)
    act_log_probs = policies.log_prob(actions)
    assert_array(act_log_probs, shape=action_values.shape)
    return -act_log_probs * action_values


def policy_gradient_loss(*, policies, actions, action_values):
    """
    https://github.com/deepmind/trfl/blob/master/docs/trfl.md#policy_gradient_losspolicies-actions-action_values-policy_varsnone-namepolicy_gradient_loss
    """
    flat_policies, _ = jax.tree_util.tree_flatten(policies)
    flat_actions, _ = jax.tree_util.tree_flatten(actions)
    assert len(flat_policies) == len(flat_actions)
    num_action_components = len(flat_actions)
    num_timesteps, num_batch = action_values.shape
    seqs = []
    for scalar_policies, scalar_actions in zip(flat_policies, flat_actions):
        assert scalar_policies.batch_shape == (num_timesteps, num_batch)
        assert_array(
            scalar_actions,
            shape=(num_timesteps, num_batch) + scalar_policies.event_shape,
            dtypes=(jp.int32, jp.float32),
        )
        seq = policy_gradient(
            policies=scalar_policies,
            actions=scalar_actions,
            action_values=action_values,
        )
        assert_array(seq, shape=(num_timesteps, num_batch))
        seqs.append(seq)
    stacked_seqs = jp.stack(seqs)
    assert_array(stacked_seqs, shape=(num_action_components, num_timesteps, num_batch))
    return jp.sum(stacked_seqs, axis=(0, 1))


def scan_discounted_sum(
    *, sequence, decay, initial_value, reverse=False, back_prop=True
):
    """
    https://github.com/deepmind/trfl/blob/master/docs/trfl.md#scan_discounted_sumsequence-decay-initial_value-reversefalse-sequence_lengthsnone-back_proptrue-namescan_discounted_sum
    """
    assert_array(sequence, shape=(None, None, ...))
    assert_array(decay, shape=sequence.shape)
    assert_array(initial_value, shape=sequence.shape[1:])

    def f(carry, x):
        out = x[0] + carry * x[1]
        return out, out

    if reverse:
        sequence = jp.flip(sequence, axis=0)
        decay = jp.flip(decay, axis=0)
    combined = jp.stack([sequence, decay], axis=1)
    _carry, discounted = jax.lax.scan(f, initial_value, combined)
    if reverse:
        discounted = jp.flip(discounted, axis=0)
    if not back_prop:
        discounted = jax.lax.stop_gradient(discounted)
    return discounted


def batched_index(values, indices):
    """
    https://github.com/deepmind/trfl/blob/master/docs/trfl.md#batched_indexvalues-indices
    """
    assert_array(indices, shape=values.shape[:-1], dtypes=(jp.int32,))
    assert len(indices.shape) in (1, 2)
    return broadcast_index(values, indices)
