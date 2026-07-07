from __future__ import annotations

import math

from .tree import MIN_BRANCH_LENGTH, PhyloTree


def log_prior(
    tree: PhyloTree, mu: float, branch_rate: float = 10.0, mu_rate: float = 1.0
) -> float:
    """Log prior for branch lengths and mutation rate.

    - Topology prior is uniform and omitted as an additive constant.
    - Branch lengths ~ Exponential(branch_rate), i.i.d.
    - mu ~ Exponential(mu_rate).
    """
    if mu <= 0.0:
        return float("-inf")
    if branch_rate <= 0.0 or mu_rate <= 0.0:
        raise ValueError("Prior rates must be positive.")

    logp = math.log(mu_rate) - mu_rate * mu
    log_branch_norm = math.log(branch_rate)
    for child, length in tree.branch_length.items():
        if child == tree.root or length <= MIN_BRANCH_LENGTH:
            return float("-inf")
        logp += log_branch_norm - branch_rate * length
    return logp

