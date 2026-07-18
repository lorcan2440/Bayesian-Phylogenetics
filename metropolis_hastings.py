import numpy as np

from initial_state import get_initial_state
from tree import PhyloTree
from priors import calc_prior_density
from proposals import propose_new_state
from likelihood import calc_log_likelihood
from utils import get_logger, branch_lengths_to_array


logger = get_logger()


def metropolis_hastings_iteration(sequences: dict[str, str], tree: PhyloTree, branch_lengths: dict[int, float], 
        r_params: np.ndarray, pi_params: np.ndarray, alpha: float, T: float) -> tuple:
    '''
    Return the next state of the Markov chain, given the current state.
    
    ### Arguments
    #### Required
    - `sequences` (dict[str, str]): a dictionary of sequences, where the keys are taxa names and the values are the corresponding sequences.
    - `tree` (PhyloTree): the current tree topology.
    - `branch_lengths` (dict[int, float]): the current branch lengths associated with the branches of the current tree.
    - `r_params` (np.ndarray): the current GTR rate parameters associated with the current tree.
    - `pi_params` (np.ndarray): the current GTR equilibrium frequencies associated with the current tree.
    - `alpha` (float): the current shape parameter for the gamma distribution associated with the current tree.
    #### Optional
    - `T` (float, default = 1.0): the temperature parameter.
    If T > 1 ('hot chain'), the acceptance probability is increased; if T < 1 ('cold chain'), the acceptance probability 
    is decreased. This should be set to 1.0 for standard MCMC, but should be changed if doing simulated annealing (SA) or MC^3.
    
    ### Returns
    - `tuple`: a tuple containing the next state of the Markov chain, in the same format as the input arguments: 
    `(tree, branch_lengths, r_params, pi_params, alpha)`.
    '''

    # calculate inverse temperature
    beta = 1 / T
    # get the proposed new state x' and the ratio of the proposal probabilities q(x' | x) / q(x | x')
    proposed_state, proposal_ratio, state_choice = propose_new_state(tree, branch_lengths, r_params, pi_params, alpha, log_ratio=False)
    proposed_tree, proposed_branch_lengths, proposed_r_params, proposed_pi_params, proposed_alpha = proposed_state

    # calculate p(D | x') and p(D | x) using our model
    current_likelihood = calc_log_likelihood(sequences, tree, branch_lengths, r_params, pi_params, alpha, calc_raw_likelihood=True)
    proposed_likelihood = calc_log_likelihood(sequences, proposed_tree, proposed_branch_lengths, proposed_r_params, proposed_pi_params, proposed_alpha, calc_raw_likelihood=True)
        
    # calculate p(x') and p(x) using the prior distributions
    current_branch_lengths_array = branch_lengths_to_array(branch_lengths)
    current_prior = calc_prior_density(len(sequences), current_branch_lengths_array, r_params, pi_params, alpha, log_prior=False)
    proposed_branch_lengths_array = branch_lengths_to_array(proposed_branch_lengths)
    proposed_prior = calc_prior_density(len(sequences), proposed_branch_lengths_array, proposed_r_params, proposed_pi_params, proposed_alpha, log_prior=False)
        
    # A(x' | x) = min{1, [p(D | x') * p(x') * q(x | x') / (p(D | x) * p(x) * q(x' | x))]^(1/T)}
    acceptance_ratio = (proposed_likelihood * proposed_prior) / (current_likelihood * current_prior)
    acceptance_ratio /= proposal_ratio
    acceptance_ratio **= beta
    acceptance_probability = min(1, acceptance_ratio)  # probability of accepting the proposed state

    logger.info(f"Proposed new {state_choice}: {proposed_state}, acceptance probability = {acceptance_probability:.4f}, β = {beta:.4f}")

    # with probability A(x' | x), accept the proposed state x' as the next state of the Markov chain; 
    # otherwise, reject the proposal and stay in the current state x
    if np.random.rand() < acceptance_probability:  # accept
        new_tree, new_branch_lengths, new_r_params, new_pi_params, new_alpha = proposed_tree, proposed_branch_lengths, proposed_r_params, proposed_pi_params, proposed_alpha
        logger.info(f"Accepted latest proposal.")
    else:  # reject
        new_tree, new_branch_lengths, new_r_params, new_pi_params, new_alpha = tree, branch_lengths, r_params, pi_params, alpha
        logger.info(f"Rejected latest proposal.")

    return new_tree, new_branch_lengths, new_r_params, new_pi_params, new_alpha


def metropolis_hastings_iteration_in_log_space(sequences: dict[str, str], tree: PhyloTree, branch_lengths: dict[int, float], r_params: np.ndarray, pi_params: np.ndarray, alpha: float, 
        T: float = 1.0) -> tuple:
    '''
    Return the next state of the Markov chain, given the current state.
    
    ### Arguments
    #### Required
    - `sequences` (dict[str, str]): a dictionary of sequences, where the keys are taxa names and the values are the corresponding sequences.
    - `tree` (PhyloTree): the current tree topology.
    - `branch_lengths` (dict[int, float]): the current branch lengths associated with the branches of the current tree.
    - `r_params` (np.ndarray): the current GTR rate parameters associated with the current tree.
    - `pi_params` (np.ndarray): the current GTR equilibrium frequencies associated with the current tree.
    - `alpha` (float): the current shape parameter for the gamma distribution associated with the current tree.
    #### Optional
    - `T` (float, default = 1.0): the temperature parameter.
    If T > 1 ('hot chain'), the acceptance probability is increased; if T < 1 ('cold chain'), the acceptance probability 
    is decreased. This should be set to 1.0 for standard MCMC, but should be changed if doing simulated annealing (SA) or MC^3.
    
    ### Returns
    - `tuple`: a tuple containing the next state of the Markov chain, in the same format as the input arguments: 
    `(tree, branch_lengths, r_params, pi_params, alpha)`.
    '''

    # calculate inverse temperature
    beta = 1 / T
    # get the proposed new state x' and the ratio of the proposal probabilities ln q(x' | x) - ln q(x | x')
    proposed_state, log_proposal_ratio, state_choice = propose_new_state(tree, branch_lengths, r_params, pi_params, alpha, log_ratio=True)
    proposed_tree, proposed_branch_lengths, proposed_r_params, proposed_pi_params, proposed_alpha = proposed_state

    # A(x' | x) = min{1, exp ΔE/T}
    # ΔE = ln p(D | x') - ln p(D | x) + ln p(x') - ln p(x) - (ln q(x' | x) - ln q(x | x'))
    current_log_likelihood = calc_log_likelihood(sequences, tree, branch_lengths, r_params, pi_params, alpha)
    proposed_log_likelihood = calc_log_likelihood(sequences, proposed_tree, proposed_branch_lengths, proposed_r_params, proposed_pi_params, proposed_alpha)
    current_branch_lengths_array = branch_lengths_to_array(branch_lengths)
    current_log_prior = calc_prior_density(len(sequences), current_branch_lengths_array, r_params, pi_params, alpha)
    proposed_branch_lengths_array = branch_lengths_to_array(proposed_branch_lengths)
    proposed_log_prior = calc_prior_density(len(sequences), proposed_branch_lengths_array, proposed_r_params, proposed_pi_params, proposed_alpha)
    delta_E = proposed_log_likelihood - current_log_likelihood + proposed_log_prior - current_log_prior - log_proposal_ratio
    
    logger.info(f"Proposed new {state_choice}: {proposed_state}, ΔE = {delta_E:.4f}, β = {beta:.4f}")

    # acceptance probability is min(1, exp(ΔE / T)), but keep in log-space to avoid numerical issues
    # generate uniform random number u ~ U(0, 1), calculate ln u and compare to ΔE / T
    u = np.random.rand()
    log_u = np.log(u)
    if log_u < beta * delta_E:  # accept
        new_tree, new_branch_lengths, new_r_params, new_pi_params, new_alpha = proposed_tree, proposed_branch_lengths, proposed_r_params, proposed_pi_params, proposed_alpha
        logger.info("Accepted latest proposal.")
    else:  # reject
        new_tree, new_branch_lengths, new_r_params, new_pi_params, new_alpha = tree, branch_lengths, r_params, pi_params, alpha
        logger.info("Rejected latest proposal.")
    
    return new_tree, new_branch_lengths, new_r_params, new_pi_params, new_alpha