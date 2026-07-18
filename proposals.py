import numpy as np
from scipy.special import softmax

from tree import PhyloTree


BRANCH_LENGTH_PERTURB_SIGMA = 0.1
ALPHA_PERTURB_SIGMA = 0.1
PI_PARAM_PERTURB_SIGMA = 0.1
RATE_PARAM_PERTURB_SIGMA = 0.1


def propose_new_tree(tree: PhyloTree, rng_seed: int = 0) -> tuple[PhyloTree, float]:
    """
    Propose a new tree topology using a nearest neighbor interchange (NNI) move, and return
    1) the new tree and 2) the ratio of the proposal probabilities q(T' | T) / q(T | T'), which is 1 for NNI moves.

    The new tree is sampled from T' ~ p(T' | T), where T is the current tree and T' is a proposed tree topology.

    Args:
        tree: The current phylogenetic tree.
        rng_seed: Seed for the local random number generator used by this function.

    Returns:
        A new phylogenetic tree with a proposed topology and the ratio of the proposal probabilities.
    """
    new_tree = tree.copy()
    rng = np.random.default_rng(rng_seed)
    success = new_tree.propose_nni(rng)
    if not success:
        raise RuntimeError("Failed to propose a new tree using NNI.")
    return new_tree, 1.0


def propose_new_branch_lengths(branch_lengths: np.ndarray, eta: float = None, rng_seed: int = 0) -> tuple[np.ndarray, float]:
    '''
    Propose a new set of branch length parameters, by perturbing only one of the branches, and return
    1) the new branch lengths and 2) the ratio of the proposal probabilities q(b' | b) / q(b | b').

    At random, one of the branch lengths is changed to b' = b * exp(eta), where eta ~ N(0, sigma^2) and sigma 
    is the constant `BRANCH_LENGTH_PERTURB_SIGMA`. This represents a log-normal proposal distribution conditioned on
    the current branch length.
    
    ### Arguments
    #### Required
    - `branch_lengths` (np.ndarray): Array of current branch lengths.
    #### Optional
    - `eta` (float): Optional perturbation value. If not provided, it will be sampled from N(0, sigma^2).
    - `rng_seed` (int, default = 0): Seed for the local random number generator used by this function.

    ### Returns
    - `np.ndarray`: Array of proposed branch lengths.
    - `float`: Ratio of the proposal probabilities q(b' | b) / q(b | b').
    '''    

    # choose a random branch index to perturb
    rng = np.random.default_rng(rng_seed)
    branch_index = rng.integers(len(branch_lengths))
    orig_branch_length = branch_lengths[branch_index]

    # exponentially perturb the chosen branch length
    if eta is None:
        eta = rng.normal(0, BRANCH_LENGTH_PERTURB_SIGMA)
    branch_lengths[branch_index] *= np.exp(eta)

    # calculate the ratio of the proposal probabilities q(b' | b) / q(b | b')
    # the ratio of the log-normal PDFs simplifies to b / b'
    ratio = orig_branch_length / branch_lengths[branch_index]

    return branch_lengths, ratio


def propose_new_GTR_rates(r_params: np.ndarray, eta: float = None, rng_seed: int = 0) -> tuple[np.ndarray, float]:
    '''
    Propose a new set of GTR rate parameters, by perturbing only one of the rates, and return
    1) the new rate parameters and 2) the ratio of the proposal probabilities q(r' | r) / q(r | r').

    At random, one of the rates is changed to r' = r * exp(eta), where eta ~ N(0, sigma^2) and sigma 
    is the constant `RATE_PARAM_PERTURB_SIGMA`.
    
    ### Arguments
    #### Required
    - `r_params` (np.ndarray): Array of current GTR rate parameters.
    #### Optional
    - `eta` (float): Optional perturbation value. If not provided, it will be sampled from N(0, sigma^2).
    - `rng_seed` (int, default = 0): Seed for the local random number generator used by this function.

    ### Returns
    - `np.ndarray`: Array of proposed GTR rate parameters.
    - `float`: Ratio of the proposal probabilities q(r' | r) / q(r | r').
    '''    

    # choose a random rate index to perturb
    rng = np.random.default_rng(rng_seed)
    rate_index = rng.integers(len(r_params))
    orig_rate = r_params[rate_index]

    # exponentially perturb the chosen rate parameter
    if eta is None:
        eta = rng.normal(0, RATE_PARAM_PERTURB_SIGMA)
    r_params[rate_index] *= np.exp(eta)

    # calculate the ratio of the proposal probabilities q(r' | r) / q(r | r')
    ratio = orig_rate / r_params[rate_index]

    return r_params, ratio


def propose_new_GTR_freqs(pi_params: np.ndarray, eta: np.ndarray = None, rng_seed: int = None) -> tuple[np.ndarray, float]:
    '''
    Propose a new set of equilibrium base frequency parameters, by adding and subtracting the same small
    value delta from two bases at random, returning 1) the new base frequencies and 2) the ratio of 
    the proposal probabilities q(pi' | pi) / q(pi | pi').
    
    ### Returns
    - `tuple[np.ndarray, float]`: Tuple containing the new base frequencies and the ratio of the 
    proposal probabilities.
    '''

    if rng_seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(rng_seed)

    # centred log ratio (CLR) transformation to map the simplex to R^3 (logit space)
    z = np.log(pi_params[:-1] / pi_params[-1])
    
    # perturb in logit space
    if eta is None:
        eta = rng.multivariate_normal([0, 0, 0], PI_PARAM_PERTURB_SIGMA ** 2 * np.eye(3))
    z_new = z + eta
    
    # inverse transformation to map back to the simplex
    pi_new = softmax(np.append(z_new, 0))

    # in logit space, this proposal is symmetric, but we need to use the Jacobian of the transformation to get 
    # the proposal ratio in pi-space, which is ratio of the products of the pi parameters
    ratio = np.prod(pi_params) / np.prod(pi_new)

    return pi_new, ratio


def propose_new_alpha(alpha: float, eta: float = None, rng_seed: int = 0) -> tuple[float, float]:
    '''
    Propose a new alpha parameter, by perturbing it, returning 1) the new alpha and 2) the ratio of the 
    proposal probabilities q(alpha' | alpha) / q(alpha | alpha').

    The alpha parameter is changed to alpha' = alpha * exp(eta), where eta ~ N(0, sigma^2) and sigma 
    is the constant `ALPHA_PERTURB_SIGMA`.
    
    ### Arguments
    #### Required
    - `alpha` (float): Current alpha parameter.
    #### Optional
    - `eta` (float): Optional perturbation value. If not provided, it will be sampled from N(0, sigma^2).
    - `rng_seed` (int, default = 0): Seed for the local random number generator used by this function.
    
    ### Returns
    - `float`: Proposed alpha parameter.
    - `float`: Ratio of the proposal probabilities q(alpha' | alpha) / q(alpha | alpha').
    '''    

    # exponentially perturb the alpha parameter
    if eta is None:
        rng = np.random.default_rng(rng_seed)
        eta = rng.normal(0, ALPHA_PERTURB_SIGMA)
    exp_eta = np.exp(eta)
    new_alpha = alpha * exp_eta
    ratio = 1 / exp_eta

    return new_alpha, ratio
