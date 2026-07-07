from __future__ import annotations

import math

NUCLEOTIDES = ("A", "C", "G", "T")
N_STATES = 4


def transition_terms(branch_length: float, mu: float) -> tuple[float, float]:
    """Return (p_same, p_diff) for JC69 along a branch."""
    exponent = math.exp(-(4.0 / 3.0) * mu * branch_length)
    p_same = 0.25 + 0.75 * exponent
    p_diff = 0.25 - 0.25 * exponent
    return p_same, p_diff

