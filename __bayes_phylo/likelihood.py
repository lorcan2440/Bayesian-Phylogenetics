from __future__ import annotations

import math

from .jc69 import transition_terms
from ..tree import PhyloTree


def build_leaf_sequences(tree: PhyloTree, data_by_taxon: dict[str, tuple[int, ...]]) -> dict[int, tuple[int, ...]]:
    leaf_sequences: dict[int, tuple[int, ...]] = {}
    for node, name in tree.leaf_name.items():
        try:
            leaf_sequences[node] = data_by_taxon[name]
        except KeyError as exc:
            raise ValueError(f"Taxon '{name}' is missing from sequence data.") from exc
    return leaf_sequences


def log_likelihood(
    tree: PhyloTree,
    leaf_sequences: dict[int, tuple[int, ...]],
    n_sites: int,
    mu: float,
) -> float:
    """Compute log P(D | T, t_T, mu) under JC69 via Felsenstein pruning."""
    if mu <= 0.0:
        return float("-inf")

    postorder = tree.postorder()
    edge_terms = {
        child: transition_terms(length, mu)
        for child, length in tree.branch_length.items()
    }

    log_like = 0.0
    for site in range(n_sites):
        partials: dict[int, list[float]] = {}
        site_log_scale = 0.0

        for node in postorder:
            if node in leaf_sequences:
                observed_state = leaf_sequences[node][site]
                vec = [0.0, 0.0, 0.0, 0.0]
                vec[observed_state] = 1.0
                partials[node] = vec
                continue

            left, right = tree.children[node]
            left_vec = partials[left]
            right_vec = partials[right]

            left_sum = left_vec[0] + left_vec[1] + left_vec[2] + left_vec[3]
            right_sum = right_vec[0] + right_vec[1] + right_vec[2] + right_vec[3]

            left_same, left_diff = edge_terms[left]
            right_same, right_diff = edge_terms[right]

            node_vec = [0.0, 0.0, 0.0, 0.0]
            for state in range(4):
                contrib_left = left_same * left_vec[state] + left_diff * (
                    left_sum - left_vec[state]
                )
                contrib_right = right_same * right_vec[state] + right_diff * (
                    right_sum - right_vec[state]
                )
                node_vec[state] = contrib_left * contrib_right

            max_val = max(node_vec)
            if max_val <= 0.0:
                return float("-inf")

            inv = 1.0 / max_val
            for i in range(4):
                node_vec[i] *= inv
            site_log_scale += math.log(max_val)
            partials[node] = node_vec

        root_vec = partials[tree.root]
        root_like = 0.25 * (root_vec[0] + root_vec[1] + root_vec[2] + root_vec[3])
        if root_like <= 0.0:
            return float("-inf")

        log_like += math.log(root_like) + site_log_scale

    return log_like

