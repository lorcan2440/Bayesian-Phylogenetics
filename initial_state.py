import numpy as np
from tree import PhyloTree, random_binary_tree
from priors import R_PARAMS_EXP_LAMBDA, PI_PARAMS_DIRICHLET_ALPHA, BRANCH_LENGTHS_EXP_LAMBDA, ALPHA_GAMMA_SHAPE


def get_initial_tree_and_branch_lengths(taxa: list[str]) -> tuple[PhyloTree, dict[int, float]]:

    tree = random_binary_tree(taxa=taxa, init_branch_mean=1 / BRANCH_LENGTHS_EXP_LAMBDA)
    branch_lengths = tree.branch_length.copy()

    return tree, branch_lengths


def get_initial_GTR_rate_params() -> np.ndarray:

    # GTR rate parameters ~ Exp(1)
    r_params = np.random.exponential(scale=1 / R_PARAMS_EXP_LAMBDA, size=5)

    return r_params


def get_initial_GTR_pi_params() -> np.ndarray:

    # GTR equilibrium frequencies ~ Dirichlet
    pi_params = np.random.dirichlet(alpha=PI_PARAMS_DIRICHLET_ALPHA)

    return pi_params


def get_initial_alpha() -> float:

    # alpha parameter ~ Gamma(1)
    alpha = np.random.gamma(shape=ALPHA_GAMMA_SHAPE, scale=1 / ALPHA_GAMMA_SHAPE)

    return alpha


def get_initial_state(taxa: list[str]) -> tuple[PhyloTree, dict[int, float], np.ndarray, np.ndarray, float]:

    tree, branch_lengths = get_initial_tree_and_branch_lengths(taxa)
    r_params = get_initial_GTR_rate_params()
    pi_params = get_initial_GTR_pi_params()
    alpha = get_initial_alpha()

    return tree, branch_lengths, r_params, pi_params, alpha
