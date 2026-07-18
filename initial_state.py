import numpy as np
import random
from tree import PhyloTree, random_binary_tree
from priors import R_PARAMS_EXP_LAMBDA, PI_PARAMS_DIRICHLET_ALPHA, BRANCH_LENGTHS_EXP_LAMBDA, ALPHA_GAMMA_SHAPE


def get_initial_tree_and_branch_lengths(taxa: list[str], rng_seed: int = 0) -> tuple[PhyloTree, dict[int, float]]:

    rng = random.Random(rng_seed)
    tree = random_binary_tree(taxa=taxa, rng=rng, init_branch_mean=1 / BRANCH_LENGTHS_EXP_LAMBDA)
    branch_lengths = tree.branch_length.copy()

    return tree, branch_lengths


def get_initial_GTR_rate_params(rng_seed: int = 0) -> np.ndarray:

    # GTR rate parameters ~ Exp(1)
    rng = np.random.default_rng(rng_seed)
    r_params = rng.exponential(scale=1 / R_PARAMS_EXP_LAMBDA, size=6)

    return r_params


def get_initial_GTR_pi_params(rng_seed: int = 0) -> np.ndarray:

    # GTR equilibrium frequencies ~ Dirichlet
    rng = np.random.default_rng(rng_seed)
    pi_params = rng.dirichlet(alpha=PI_PARAMS_DIRICHLET_ALPHA)

    return pi_params


def get_initial_alpha(rng_seed: int = 0) -> float:

    # alpha parameter ~ Gamma(1)
    rng = np.random.default_rng(rng_seed)
    alpha = rng.gamma(shape=ALPHA_GAMMA_SHAPE, scale=1 / ALPHA_GAMMA_SHAPE)

    return alpha


def get_initial_state(taxa: list[str], rng_seed: int = 0) -> tuple[PhyloTree, dict[int, float], np.ndarray, np.ndarray, float]:

    # derive deterministic sub-seeds so one seed controls the full initial state reproducibly
    rng = np.random.default_rng(rng_seed)
    tree_seed, r_seed, pi_seed, alpha_seed = [int(s) for s in rng.integers(0, np.iinfo(np.uint32).max, size=4)]

    tree, branch_lengths = get_initial_tree_and_branch_lengths(taxa, rng_seed=tree_seed)
    r_params = get_initial_GTR_rate_params(rng_seed=r_seed)
    pi_params = get_initial_GTR_pi_params(rng_seed=pi_seed)
    alpha = get_initial_alpha(rng_seed=alpha_seed)

    return tree, branch_lengths, r_params, pi_params, alpha
