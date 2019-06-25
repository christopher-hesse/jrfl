from .check import assert_array
from .extra import broadcast_index
from .tfp import Distribution, Independent, Normal, Categorical
from .trfl import (
    discrete_policy_gradient,
    discrete_policy_gradient_loss,
    policy_gradient,
    policy_gradient_loss,
    scan_discounted_sum,
    batched_index,
)


__all__ = [
    "Categorical",
    "Distribution",
    "Independent",
    "Normal",
    "assert_array",
    "batched_index",
    "broadcast_index",
    "discrete_policy_gradient",
    "discrete_policy_gradient_loss",
    "policy_gradient",
    "policy_gradient_loss",
    "scan_discounted_sum",
]
