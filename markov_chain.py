import numpy as np

from initial_state import get_initial_state
from metropolis_hastings import metropolis_hastings_iteration_in_log_space
from utils import get_logger


logger = get_logger()


def run_markov_chain(sequences: dict[str, str], n_iter: int, n_burn_in: int, n_prune: int = 10, T: float = 1.0) \
        -> tuple[list, list, list, list, list]:
    '''
    Run the Metropolis-Hastings algorithm for a given number of iterations and return the Markov chain.
    
    ### Arguments
    #### Required
    - `sequences` (dict[str, str]): a dictionary of sequences, where the keys are taxa names and the values are the corresponding sequences.
    - `n_iter` (int): the number of iterations to run the Markov chain.
    - `n_burn_in` (int): the number of initial iterations to discard as burn-in.
    #### Optional
    - `n_prune` (int, default = 10): once iteration has finished, we decimate by retaining only every `n_prune`'th sample, 
    to reduce autocorrelation in the Markov chain.
    - `T` (float, default = 1.0): the temperature parameter.
    If T > 1 ('hot chain'), the acceptance probability is increased; if T < 1 ('cold chain'), the acceptance probability 
    is decreased. This should be set to 1.0 for standard MCMC, but should be changed if doing simulated annealing (SA) or MC^3.
    
    ### Returns
    - `tuple`: a tuple containing five lists, each representing the Markov chain for one of the parameters: 
    `(trees_arr, branch_lengths_arr, r_params_arr, pi_params_arr, alpha_arr)`.
    '''

    logger.info(f"Running Markov chain for {n_iter} iterations with T = {T:.4f}...")

    taxa = list(sequences.keys())

    # init empty buffers to store all states
    trees_arr = []
    branch_lengths_arr = []
    r_params_arr = []
    pi_params_arr = []
    alpha_arr = []

    # set initial state
    tree, branch_lengths, r_params, pi_params, alpha = get_initial_state(taxa)

    trees_arr.append(tree)
    branch_lengths_arr.append(branch_lengths)
    r_params_arr.append(r_params)
    pi_params_arr.append(pi_params)
    alpha_arr.append(alpha)

    logger.info(f"Initial state: tree = {tree.to_newick()}, branch_lengths = {branch_lengths}, r_params = {r_params}, pi_params = {pi_params}, alpha = {alpha}")

    # iterate through the Markov chain
    for i in range(1, n_iter):
        tree, branch_lengths, r_params, pi_params, alpha = metropolis_hastings_iteration_in_log_space(
            sequences, tree, branch_lengths, r_params, pi_params, alpha, T=T
        )
        trees_arr.append(tree)
        branch_lengths_arr.append(branch_lengths)
        r_params_arr.append(r_params)
        pi_params_arr.append(pi_params)
        alpha_arr.append(alpha)
        
        logger.info(f"Iteration {i}: tree = {tree.to_newick()}, branch_lengths = {branch_lengths}, r_params = {r_params}, pi_params = {pi_params}, alpha = {alpha}")

    # discard burn-in samples
    trees_arr = trees_arr[n_burn_in:]
    branch_lengths_arr = branch_lengths_arr[n_burn_in:]
    r_params_arr = r_params_arr[n_burn_in:]
    pi_params_arr = pi_params_arr[n_burn_in:]
    alpha_arr = alpha_arr[n_burn_in:]

    # decimate the Markov chain to reduce autocorrelation
    trees_arr = trees_arr[::n_prune]
    branch_lengths_arr = branch_lengths_arr[::n_prune]
    r_params_arr = r_params_arr[::n_prune]
    pi_params_arr = pi_params_arr[::n_prune]
    alpha_arr = alpha_arr[::n_prune]

    return trees_arr, branch_lengths_arr, r_params_arr, pi_params_arr, alpha_arr


if __name__ == "__main__":

    sequences = {
        "Human": "AA",
        "Chimpanzee": "AA",
        "Gorilla": "CA",
        "Orangutan": "TG"
    }

    n_iter = 1_000_000
    n_burn_in = 100_000
    n_prune = 10
    T = 1.0

    trees_arr, branch_lengths_arr, r_params_arr, pi_params_arr, alpha_arr = run_markov_chain(
        sequences=sequences, n_iter=n_iter, n_burn_in=n_burn_in, n_prune=n_prune, T=T
    )