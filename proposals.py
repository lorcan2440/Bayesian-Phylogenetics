import numpy as np
from scipy.special import softmax

from tree import PhyloTree


BRANCH_LENGTH_PERTURB_SIGMA = 0.1
ALPHA_PERTURB_SIGMA = 0.1
PI_PARAM_PERTURB_SIGMA = 0.1
RATE_PARAM_PERTURB_SIGMA = 0.1

DEFAULT_STATE_CHANGE_PROBS = {
    "tree": 0.2,
    "branch_lengths": 0.2,
    "r_params": 0.2,
    "pi_params": 0.2,
    "alpha": 0.2
}


def propose_new_tree(tree: PhyloTree, rng_seed: int = 0, log_ratio: bool = True) -> tuple[PhyloTree, float]:
    """
    Propose a new tree topology using a nearest neighbor interchange (NNI) move, and return
    1) the new tree and 2) the ratio of the proposal probabilities q(T' | T) / q(T | T'), which is 1 for NNI moves.

    The new tree is sampled from T' ~ p(T' | T), where T is the current tree and T' is a proposed tree topology.

    ### Arguments
    #### Required
    - `tree` (PhyloTree): the current tree topology.
    #### Optional
    - `eta` (float): Optional perturbation value. If not provided, it will be sampled from N(0, sigma^2).
    - `rng_seed` (int, default = 0): Seed for the local random number generator used by this function.
    - `log_ratio` (bool, default = True): If True, return the log of the proposal ratio. If False, return the proposal ratio.

    ### Returns
    - `PhyloTree`: A new phylogenetic tree with a proposed topology and the ratio of the proposal probabilities.
    - `float`: Ratio of the proposal probabilities q(T' | T) / q(T | T').

    Returns:
        A new phylogenetic tree with a proposed topology and the ratio of the proposal probabilities.
    """
    new_tree = tree.copy()
    rng = np.random.default_rng(rng_seed)
    success = new_tree.propose_nni(rng)
    if not success:
        raise RuntimeError("Failed to propose a new tree using NNI.")
    
    if log_ratio:
        return new_tree, 0.0
    else:
        return new_tree, 1.0


def propose_new_branch_lengths(branch_lengths: np.ndarray, eta: float = None, rng_seed: int = 0, log_ratio: bool = True) -> tuple[np.ndarray, float]:
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
    - `log_ratio` (bool, default = True): If True, return the log of the proposal ratio. If False, return the proposal ratio.

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
    if log_ratio:
        log_orig = np.log(orig_branch_length)
        log_new = np.log(branch_lengths[branch_index])
        log_ratio = log_orig - log_new
        return branch_lengths, log_ratio
    else:
        ratio = orig_branch_length / branch_lengths[branch_index]
        return branch_lengths, ratio


def propose_new_GTR_rates(r_params: np.ndarray, eta: float = None, rng_seed: int = 0, log_ratio: bool = True) -> tuple[np.ndarray, float]:
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
    - `log_ratio` (bool, default = True): If True, return the log of the proposal ratio. If False, return the proposal ratio.

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
    if log_ratio:
        log_orig = np.log(orig_rate)
        log_new = np.log(r_params[rate_index])
        log_ratio = log_orig - log_new
        return r_params, log_ratio
    else:
        ratio = orig_rate / r_params[rate_index]
        return r_params, ratio


def propose_new_GTR_freqs(pi_params: np.ndarray, eta: np.ndarray = None, rng_seed: int = None, log_ratio: bool = True) -> tuple[np.ndarray, float]:
    '''
    Propose a new set of equilibrium base frequency parameters, by adding and subtracting the same small
    value delta from two bases at random, returning 1) the new base frequencies and 2) the ratio of 
    the proposal probabilities q(pi' | pi) / q(pi | pi').

    ### Arguments
    #### Required
    - `pi_params` (np.ndarray): Array of current equilibrium base frequencies.
    #### Optional
    - `eta` (np.ndarray): Optional perturbation value. If not provided, it will be sampled from N(0, sigma^2).
    - `rng_seed` (int, default = 0): Seed for the local random number generator used by this function.
    - `log_ratio` (bool, default = True): If True, return the log of the proposal ratio. If False, return the proposal ratio.
    
    ### Returns
    - `tuple[np.ndarray, float]`: Tuple containing the new base frequencies and the ratio of the 
    proposal probabilities. If `log_ratio` is True, the ratio is returned in log space.
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
    if log_ratio:
        log_ratio = np.sum(np.log(pi_params)) - np.sum(np.log(pi_new))
        return pi_new, log_ratio
    else:
        ratio = np.prod(pi_params) / np.prod(pi_new)
        return pi_new, ratio


def propose_new_alpha(alpha: float, eta: float = None, rng_seed: int = 0, log_ratio: bool = True) -> tuple[float, float]:
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
    - `log_ratio` (bool, default = True): If True, return the log of the proposal ratio. If False, return the proposal ratio.
    
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

    if log_ratio:
        log_ratio = -eta
        return new_alpha, log_ratio
    else:
        ratio = 1 / exp_eta
        return new_alpha, ratio


def propose_new_state(tree: PhyloTree, branch_lengths: dict[int, float], r_params: np.ndarray, pi_params: np.ndarray, alpha: float,
        state_change_probs: dict[str, float] = DEFAULT_STATE_CHANGE_PROBS, rng_seed: int = 0, log_ratio: bool = True) \
        -> tuple[tuple, float, str]:
    '''
    Propose a new state of the Markov chain, by proposing new values for the tree topology, branch lengths,
    GTR rate parameters, GTR equilibrium frequencies, or alpha parameter. Only one of these state variables is changed 
    at a time, and the new state is returned along with the ratio of the proposal probabilities.
    
    Return 1) the new state and 2) the ratio of the proposal probabilities q(x' | x) / q(x | x').
    If `log_ratio` is True, the ratio returned is ln q(x' | x) - ln q(x | x').
    
    ### Arguments
    #### Required
    - `tree` (PhyloTree): Current tree topology.
    - `branch_lengths` (dict[int, float]): Current branch lengths.
    - `r_params` (np.ndarray): Current GTR rate parameters.
    - `pi_params` (np.ndarray): Current GTR equilibrium frequencies.
    - `alpha` (float): Current alpha parameter.
    #### Optional
    - `rng_seed` (int, default = 0): Seed for the local random number generator used by this function.
    - `log_ratio` (bool, default = True): If True, return the log of the proposal ratio. If False, return the proposal ratio.
    - `state_change_probs` (dict[str, float], default = DEFAULT_STATE_CHANGE_PROBS): Dictionary of probabilities for 
    proposing a new value for each state variable. The keys are the names of the state variables and the values are 
    the probabilities of proposing a new value for that variable. The probabilities must sum to 1.
    
    ### Returns
    - `tuple`: the new state and the ratio of the proposal probabilities.
    - `float`: Ratio of the proposal probabilities q(x' | x) / q(x | x'). 
    If `log_ratio` is True, this value is ln q(x' | x) - ln q(x | x').
    - `str`: The name of the state variable that was changed: "tree", "branch_lengths", "r_params", "pi_params", or "alpha".
    '''
    STATE_VARIABLES = ["tree", "branch_lengths", "r_params", "pi_params", "alpha"]
    state_choice = np.random.choice(STATE_VARIABLES, p=[state_change_probs[var] for var in STATE_VARIABLES])

    if state_choice == "tree":
        new_tree, proposal_ratio = propose_new_tree(tree, rng_seed=rng_seed, log_ratio=log_ratio)
        proposed_state = (new_tree, branch_lengths.copy(), r_params, pi_params, alpha)
    elif state_choice == "branch_lengths":
        branch_ids = sorted(branch_lengths.keys())
        branch_lengths_arr = np.array([branch_lengths[branch_id] for branch_id in branch_ids], dtype=float)
        new_branch_lengths_arr, proposal_ratio = propose_new_branch_lengths(branch_lengths_arr, rng_seed=rng_seed, log_ratio=log_ratio)
        new_branch_lengths = {branch_id: float(length) for branch_id, length in zip(branch_ids, new_branch_lengths_arr)}
        proposed_state = (tree, new_branch_lengths, r_params, pi_params, alpha)
    elif state_choice == "r_params":
        new_r_params, proposal_ratio = propose_new_GTR_rates(r_params.copy(), rng_seed=rng_seed, log_ratio=log_ratio)
        proposed_state = (tree, branch_lengths.copy(), new_r_params, pi_params, alpha)
    elif state_choice == "pi_params":
        new_pi_params, proposal_ratio = propose_new_GTR_freqs(pi_params.copy(), rng_seed=rng_seed, log_ratio=log_ratio)
        proposed_state = (tree, branch_lengths.copy(), r_params, new_pi_params, alpha)
    elif state_choice == "alpha":
        new_alpha, proposal_ratio = propose_new_alpha(alpha, rng_seed=rng_seed, log_ratio=log_ratio)
        proposed_state = (tree, branch_lengths.copy(), r_params, pi_params, new_alpha)

    return proposed_state, proposal_ratio, state_choice
