import numpy as np

from tree import PhyloTree


BRANCH_LENGTH_PERTURB_SIGMA = 0.1
ALPHA_PERTURB_SIGMA = 0.1
PI_PARAM_PERTURB_EPSILON = 0.02
RATE_PARAM_PERTURB_SIGMA = 0.1


def propose_new_tree(tree: PhyloTree) -> tuple[PhyloTree, float]:
    """
    Propose a new tree topology using a nearest neighbor interchange (NNI) move, and return
    1) the new tree and 2) the ratio of the proposal probabilities q(T' | T) / q(T | T'), which is 1 for NNI moves.

    The new tree is sampled from T' ~ p(T' | T), where T is the current tree and T' is a proposed tree topology.

    Args:
        tree: The current phylogenetic tree.
        rng: A random number generator.

    Returns:
        A new phylogenetic tree with a proposed topology and the ratio of the proposal probabilities.
    """
    new_tree = tree.copy()
    success = new_tree.propose_nni(np.random)
    if not success:
        raise RuntimeError("Failed to propose a new tree using NNI.")
    return new_tree, 1.0


def propose_new_branch_lengths(branch_lengths: np.ndarray) -> tuple[np.ndarray, float]:
    '''
    Propose a new set of branch length parameters, by perturbing only one of the branches, and return
    1) the new branch lengths and 2) the ratio of the proposal probabilities q(b' | b) / q(b | b').

    At random, one of the branch lengths is changed to b' = b * exp(eta), where eta ~ N(0, sigma^2) and sigma 
    is the constant `BRANCH_LENGTH_PERTURB_SIGMA`. This represents a log-normal proposal distribution conditioned on
    the current branch length.
    
    ### Arguments
    - `branch_lengths` (np.ndarray): Array of current branch lengths.
    
    ### Returns
    - `np.ndarray`: Array of proposed branch lengths.
    - `float`: Ratio of the proposal probabilities q(b' | b) / q(b | b').
    '''    

    # choose a random branch index to perturb
    branch_index = np.random.randint(len(branch_lengths))
    orig_branch_length = branch_lengths[branch_index]

    # exponentially perturb the chosen branch length
    eta = np.random.normal(0, BRANCH_LENGTH_PERTURB_SIGMA)
    branch_lengths[branch_index] *= np.exp(eta)

    # calculate the ratio of the proposal probabilities q(b' | b) / q(b | b')
    # the ratio of the log-normal PDFs simplifies to b / b'
    ratio = orig_branch_length / branch_lengths[branch_index]

    return branch_lengths, ratio


def propose_new_GTR_rates(r_params: np.ndarray) -> tuple[np.ndarray, float]:
    '''
    Propose a new set of GTR rate parameters, by perturbing only one of the rates, and return
    1) the new rate parameters and 2) the ratio of the proposal probabilities q(r' | r) / q(r | r').

    At random, one of the rates is changed to r' = r * exp(eta), where eta ~ N(0, sigma^2) and sigma 
    is the constant `RATE_PARAM_PERTURB_SIGMA`.
    
    ### Arguments
    - `r_params` (np.ndarray): Array of current GTR rate parameters.
    
    ### Returns
    - `np.ndarray`: Array of proposed GTR rate parameters.
    - `float`: Ratio of the proposal probabilities q(r' | r) / q(r | r').
    '''    

    # choose a random rate index to perturb
    rate_index = np.random.randint(len(r_params))
    orig_rate = r_params[rate_index]

    # exponentially perturb the chosen rate parameter
    eta = np.random.normal(0, RATE_PARAM_PERTURB_SIGMA)
    r_params[rate_index] *= np.exp(eta)

    # calculate the ratio of the proposal probabilities q(r' | r) / q(r | r')
    ratio = orig_rate / r_params[rate_index]

    return r_params, ratio


def propose_new_GTR_freqs(pi_params: np.ndarray) -> tuple[np.ndarray, float]:

    # choose two nucleotides at random (of the indices 0, 1, 2, 3)
    i, j = np.random.choice(4, size=2, replace=False)

    new_pi_params = pi_params.copy()

    # perturb the first nucleotide's frequency by a small amount
    delta = np.random.uniform(-PI_PARAM_PERTURB_EPSILON, PI_PARAM_PERTURB_EPSILON)
    new_pi_params[i] += delta
    new_pi_params[j] -= delta

    if new_pi_params[i] > 1.0 or new_pi_params[j] < 0.0:
        return propose_new_GTR_freqs(pi_params)  # try again if the perturbation goes out of bounds
    
    # re-normalise to remove any numerical drift
    new_pi_params /= np.sum(new_pi_params)
    
    return new_pi_params, 1.0  # the proposal is symmetric, so the ratio is 1


def propose_new_alpha(alpha: float) -> tuple[float, float]:
    '''
    Propose a new alpha parameter, by perturbing it, returning 1) the new alpha and 2) the ratio of the 
    proposal probabilities q(alpha' | alpha) / q(alpha | alpha').

    The alpha parameter is changed to alpha' = alpha * exp(eta), where eta ~ N(0, sigma^2) and sigma 
    is the constant `ALPHA_PERTURB_SIGMA`.
    
    ### Arguments
    - `alpha` (float): Current alpha parameter.
    
    ### Returns
    - `float`: Proposed alpha parameter.
    - `float`: Ratio of the proposal probabilities q(alpha' | alpha) / q(alpha | alpha').
    '''    

    # exponentially perturb the alpha parameter
    eta = np.random.normal(0, ALPHA_PERTURB_SIGMA)
    exp_eta = np.exp(eta)
    new_alpha = alpha * exp_eta
    ratio = 1 / exp_eta

    return new_alpha, ratio
