import numpy as np
from scipy.stats import dirichlet, expon, beta, gamma, uniform
from scipy.special import factorial2

from tree import PhyloTree


R_PARAMS_EXP_LAMBDA = 1.0  # Exponential prior for r_params
PI_PARAMS_DIRICHLET_ALPHA = np.ones(4)  # Dirichlet prior for pi_params
BRANCH_LENGTHS_EXP_LAMBDA = 10.0  # Exponential prior for branch lengths
ALPHA_GAMMA_SHAPE = 2.0  # Gamma prior for alpha


def calc_prior_density_tree(num_taxa: int, _tree: PhyloTree = None, log_prior: bool = True) -> float:
    '''
    Calculates the prior probability of a given tree topology p(T).
    
    ### Arguments
    - `num_taxa` (int): number of taxa in the phylogenetic tree
    - `_tree` (PhyloTree, optional): phylogenetic tree object containing a given topology (not used)
    
    ### Returns
    - `float`: value of the prior probability p(T)
    '''

    if _tree is not None:
        num_taxa = len(_tree.leaf_name)
    
    # tree ~ uniform over all possible rooted binary tree topologies
    num_trees = factorial2(2 * num_taxa - 3)
    prior_tree = 1 / num_trees

    if log_prior:
        return np.log(prior_tree)
    else:
        return prior_tree


def calc_prior_density_branch_lengths(branch_lengths: np.ndarray, log_prior: bool = True) -> float:
    '''
    Calculates the prior probability of a given set of branch lengths p(b).
    
    ### Arguments
    - `branch_lengths` (np.ndarray): array of branch lengths for the tree
    
    ### Returns
    - `float`: value of the prior probability p(b)
    '''    

    # branch_lengths ~ Exp(10)
    prior_branch_lengths = np.prod(expon.pdf(branch_lengths, scale=1 / BRANCH_LENGTHS_EXP_LAMBDA))

    if log_prior:
        return np.log(prior_branch_lengths)
    else:
        return prior_branch_lengths


def calc_prior_density_r_params(r_params: np.ndarray, log_prior: bool = True) -> float:
    '''
    Calculates the prior probability of a given set of rate parameters p(r).
    
    ### Arguments
    - `r_params` (np.ndarray): array of rate parameters for the substitution model
    
    ### Returns
    - `float`: value of the prior probability p(r)
    '''    

    # r_params ~ Exp(1)
    prior_r = np.prod(expon.pdf(r_params, scale=1 / R_PARAMS_EXP_LAMBDA))

    if log_prior:
        return np.log(prior_r)
    else:
        return prior_r


def calc_prior_density_pi_params(pi_params: np.ndarray, log_prior: bool = True) -> float:
    '''
    Calculates the prior probability of a given set of equilibrium frequencies p(π).
    
    ### Arguments
    - `pi_params` (np.ndarray): array of equilibrium frequencies for the substitution model
    
    ### Returns
    - `float`: value of the prior probability p(π)
    '''    

    # pi_params ~ Dirichlet(1, 1, 1, 1)
    prior_pi = dirichlet.pdf(pi_params, alpha=PI_PARAMS_DIRICHLET_ALPHA)

    if log_prior:
        return np.log(prior_pi)
    else:
        return prior_pi


def calc_prior_density_alpha(alpha: float, log_prior: bool = True) -> float:
    '''
    Calculates the prior probability of a given shape parameter for the gamma distribution p(α).
    
    ### Arguments
    - `alpha` (float): shape parameter for the gamma distribution
    
    ### Returns
    - `float`: value of the prior probability p(α)
    '''    

    # alpha ~ Gamma(2, 2)
    prior_alpha = gamma.pdf(alpha, a=ALPHA_GAMMA_SHAPE, scale=1 / ALPHA_GAMMA_SHAPE)

    if log_prior:
        return np.log(prior_alpha)
    else:
        return prior_alpha


def calc_prior_density(num_taxa: int, branch_lengths: np.ndarray, r_params: np.ndarray, pi_params: np.ndarray, 
                    alpha: float, log_prior: bool = True) -> float:
    '''
    Calculates the value of the prior probability p(T, b, θ) at a given point in parameter space.
    
    ### Arguments
    - `num_taxa` (int): number of taxa in the phylogenetic tree
    - `branch_lengths` (np.ndarray): array of branch lengths for the tree
    - `r_params` (np.ndarray): array of rate parameters for the substitution model
    - `pi_params` (np.ndarray): array of equilibrium frequencies for the substitution model
    - `alpha` (float): shape parameter for the gamma distribution
    
    ### Returns
    - `float`: value of the prior probability p(T, b, θ)
    '''

    prior_tree = calc_prior_density_tree(num_taxa, log_prior=log_prior)
    prior_branch_lengths = calc_prior_density_branch_lengths(branch_lengths, log_prior=log_prior)
    prior_r = calc_prior_density_r_params(r_params, log_prior=log_prior)
    prior_pi = calc_prior_density_pi_params(pi_params, log_prior=log_prior)
    prior_alpha = calc_prior_density_alpha(alpha, log_prior=log_prior)

    if log_prior:
        log_prior_result = prior_tree + prior_branch_lengths + prior_r + prior_pi + prior_alpha
        return log_prior_result
    else:
        prior_result = prior_tree * prior_branch_lengths * prior_r * prior_pi * prior_alpha
        return prior_result


if __name__ == "__main__":

    sequences = {
        "Human": "AA",
        "Chimpanzee": "AA",
        "Gorilla": "CA",
        "Orangutan": "TG"
    }
    branch_length = {
        0: 0.05,
        1: 0.05,
        2: 0.08,
        3: 0.12,
        4: 0.10,
        5: 0.10,
    }

    tree = PhyloTree(
        root=6,
        children={
            6: (4, 5),
            4: (0, 1),
            5: (2, 3),
        },
        parent={0: 4, 
                1: 4, 
                2: 5, 
                3: 5, 
                4: 6, 
                5: 6},
        leaf_name={0: "Human", 
                   1: "Chimpanzee", 
                   2: "Gorilla", 
                   3: "Orangutan"},
        next_id=7,
    )

    r_AC, r_AG, r_AT, r_CG, r_CT, r_GT = 0.2, 0.1, 0.4, 0.5, 0.8, 1.0
    pi_A, pi_C, pi_G, pi_T = 0.3, 0.2, 0.2, 0.3

    alpha = 0.5

    r_params = (r_AC, r_AG, r_AT, r_CG, r_CT, r_GT)
    pi_params = (pi_A, pi_C, pi_G, pi_T)

    tree.branch_length = branch_length
    num_taxa = len(tree.leaf_name)

    branch_lengths = np.array([branch_length[i] for i in range(len(branch_length))])
    log_prior = calc_prior_density(num_taxa, branch_lengths, r_params, pi_params, alpha, log_prior=True)
    print("Log-prior density:", log_prior)
