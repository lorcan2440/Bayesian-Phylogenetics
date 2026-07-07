# external modules
import numpy as np
from scipy.stats import gamma as gamma_dist

# local modules
from tree import PhyloTree
from tree_visualisation import render_newick_svg
from utils import get_logger


logger = get_logger()

CHARS = ("A", "C", "G", "T")


def GTR_Q_matrix(r_params: tuple[float], pi_params: tuple[float]) -> np.ndarray:
    '''
    Compute the continuous-time Markov chain (CTMC) transition rate matrix Q for the General Time Reversible (GTR) 
    model of nucleotide substitution, which obeys:

    dP(t)/dt = P(t) @ Q

    ### Arguments
    - `r_params` (tuple[float]): GTR rate parameters (r_AC, r_AG, r_AT, r_CG, r_CT, r_GT)
    - `pi_params` (tuple[float]): GTR equilibrium frequencies (pi_A, pi_C, pi_G, pi_T)

    ### Returns
    - `np.ndarray`: GTR transition rate matrix Q
    '''

    r_AC, r_AG, r_AT, r_CG, r_CT, r_GT = r_params
    pi_A, pi_C, pi_G, pi_T = pi_params

    Q = np.array([
        [-1 * (r_AC * pi_C + r_AG * pi_G + r_AT * pi_T), r_AC * pi_C, r_AG * pi_G, r_AT * pi_T],
        [r_AC * pi_A, -1 * (r_AC * pi_A + r_CG * pi_G + r_CT * pi_T), r_CG * pi_G, r_CT * pi_T],
        [r_AG * pi_A, r_CG * pi_C, -1 * (r_AG * pi_A + r_CG * pi_C + r_GT * pi_T), r_GT * pi_T],
        [r_AT * pi_A, r_CT * pi_C, r_GT * pi_G, -1 * (r_AT * pi_A + r_CT * pi_C + r_GT * pi_G)]
    ])

    return Q


def diagonalise_GTR_Q_matrix(Q: np.ndarray) -> tuple[np.ndarray]:
    '''
    Diagonalise the continuous-time Markov chain (CTMC) transition rate matrix Q for the General Time Reversible (GTR) 
    model of nucleotide substitution.
    
    ### Arguments
    - `Q` (np.ndarray): GTR transition rate matrix
    
    ### Returns
    - `tuple[np.ndarray]`: a tuple containing the eigenvector matrix S, the 1D array of eigenvalues lambda_vals, 
    and the inverse matrix of S.
    '''

    lambda_vals, S = np.linalg.eig(Q)

    # ensure eigenvalues and eigenvectors are real-valued (they should be for a valid GTR matrix)
    lambda_vals = np.real(lambda_vals)
    S = np.real(S)

    S_inv = np.linalg.inv(S)  # pre-compute inverse matrix of S

    return S, lambda_vals, S_inv


def calc_transition_probability_matrix(S: np.ndarray, lambda_vals: np.ndarray, S_inv: np.ndarray, branch_length: float, gamma: float) -> np.ndarray:
    '''
    Calculate the discrete-time Markov chain (DTMC) transition matrix P given 
    the CTMC transition rate matrix Q (in its diagonalised form).

    Returns P(t), where t is the branch length.
    
    ### Arguments
    - `S` (np.ndarray): eigenvector matrix of Q
    - `lambda_vals` (np.ndarray): 1D array of eigenvalues of Q
    - `S_inv` (np.ndarray): inverse of the eigenvector matrix S
    - `branch_length` (float): branch length for which to calculate the transition matrix
    - `gamma` (float): rate heterogeneity parameter

    ### Returns
    - `np.ndarray`: transition probability matrix P
    '''

    P = S @ np.diag(np.exp(gamma * branch_length * lambda_vals)) @ S_inv
    return P


def calc_likelihood_of_ancestral_char(tree: PhyloTree, node: int, char_index: int, gamma: float, site_index: int) -> float:
    '''Calculate the likelihood L_node(char) of observing a character at a given node, given the likelihoods of its 
    children and the transition probabilities along the branches.

    Returns L_node^(k)(char).

    ### Arguments
    - `tree` (PhyloTree): the phylogenetic tree
    - `node` (int): the node ID for which to calculate the likelihood
    - `char_index` (int): the index of the character in CHARS for which to calculate the likelihood
    - `gamma` (float): rate heterogeneity parameter

    ### Returns
    - `float`: likelihood of observing the character at the given node
    '''

    logger.info(f'\t\t\tCalculating likelihood of ancestral node {node} having character "{CHARS[char_index]}".')

    left_child, right_child = tree.children[node]

    left_branch_length = tree.branch_length[left_child]
    right_branch_length = tree.branch_length[right_child]

    P_left = calc_transition_probability_matrix(tree.S, tree.lambda_vals, tree.S_inv, left_branch_length, gamma)
    P_right = calc_transition_probability_matrix(tree.S, tree.lambda_vals, tree.S_inv, right_branch_length, gamma)

    left_sum = 0
    for left_char_index, _left_char in enumerate(CHARS):
        left_transition_prob = P_left[char_index, left_char_index]
        left_likelihood = tree.likelihoods[site_index][left_child][left_char_index]
        left_sum += left_transition_prob * left_likelihood

    right_sum = 0
    for right_char_index, _right_char in enumerate(CHARS):
        right_transition_prob = P_right[char_index, right_char_index]
        right_likelihood = tree.likelihoods[site_index][right_child][right_char_index]
        right_sum += right_transition_prob * right_likelihood

    prod = left_sum * right_sum
    return prod


def felsenstein_pruning(sequences: dict[str, str], tree: PhyloTree, site_index: int, gamma_vals: np.ndarray) -> float:
    '''Use Felsenstein's pruning algorithm to calculate the likelihood of observing characters at all nodes 
    in the tree, at a given site.

    Returns p(D_k | T, b, theta), where D_k are the observed characters at site k, T is the tree topology, 
    b are the branch lengths, and theta are the GTR parameters.

    ### Arguments
    - `sequences` (dict[str, str]): dictionary mapping taxon names to their sequences
    - `tree` (PhyloTree): the phylogenetic tree
    - `site_index` (int): the index of the site for which to calculate the likelihoods
    - `gamma_vals` (np.ndarray): array of gamma values for rate heterogeneity

    ### Returns
    - `float`: the likelihood of observing the characters at the given site
    '''

    logger.info(f"Calculating likelihood for site {site_index}.")

    n_gamma_bins = len(gamma_vals)

    root_likelihood = 0
    for gamma in gamma_vals:  # repeat for each sampled gamma value (heterogeneous rate parameter)

        logger.info(f"\tCalculating likelihood with gamma = {gamma:.4f}.")

        # initialise likelihood vectors at this site, for all nodes
        tree.likelihoods[site_index] = {}
        for node in range(tree.next_id):
            if node in tree.leaf_name:
                observed_char = sequences[tree.leaf_name[node]][site_index]  # nucleotide in leaf node from given data
                # likelihood is 1 if the character matches, 0 otherwise
                tree.likelihoods[site_index][node] = [1.0 if observed_char == char else 0.0 for char in CHARS]
            else:
                tree.likelihoods[site_index][node] = [None] * len(CHARS)  # non-leaf nodes are left as None

        # assign likelihoods to all nodes of the tree
        for node in tree.postorder_traversal():
            if node not in tree.children:
                continue  # we already know the likelihoods for leaf nodes
            else:
                logger.info(f"\t\tCalculating likelihood vector for ancestral node {node}.")
                for char_index, _char in enumerate(CHARS):  # set the likelihood vector for this node
                    tree.likelihoods[site_index][node][char_index] = calc_likelihood_of_ancestral_char(tree, node, char_index, gamma, site_index)

        # compute site likelihood at the root, p(D_k | T, b, theta, gamma_k)
        root_likelihood_gamma = 0
        for char_index, _char in enumerate(CHARS):
            root_likelihood_gamma += tree.pi_params[char_index] * tree.likelihoods[site_index][tree.root][char_index]

        # marginalise over gamma values
        root_likelihood += root_likelihood_gamma * (1 / n_gamma_bins)

    # returns p(D_k | T, b, theta)
    return root_likelihood


def calc_likelihood(sequences: dict[str, str], tree: PhyloTree, branch_length: dict[int, float], r_params: tuple[float], pi_params: tuple[float], alpha: float, n_gamma_bins: int) -> float:
    '''Calculate the likelihood of observing the given sequences at the leaves of the tree, under the GTR model with 
    rate heterogeneity.

    Returns p(D | T, b, theta), where D are the observed sequences at the leaves, T is the tree topology, 
    b are the branch lengths, and theta are the GTR parameters.

    This output is suitable for use in a Metropolis-Hastings algorithm to sample from the posterior 
    distribution of trees, branch lengths, and GTR parameters, in Bayesian MCMC.

    ### Arguments
    - `tree` (PhyloTree): the phylogenetic tree
    - `sequences` (dict[str, str]): a dictionary mapping leaf names to their corresponding sequences
    - `r_params` (tuple[float]): GTR rate parameters (r_AC, r_AG, r_AT, r_CG, r_CT, r_GT)
    - `pi_params` (tuple[float]): GTR equilibrium frequencies (pi_A, pi_C, pi_G, pi_T)
    - `alpha` (float): shape parameter for the gamma distribution modeling rate heterogeneity
    - `n_gamma_bins` (int): number of discrete gamma bins to use for rate heterogeneity

    ### Returns
    - `float`: the likelihood of observing the sequences at the leaves of the tree
    '''

    logger.info("Started calculation of likelihood.")

    # assert all sequences have the same length
    if len(set(len(seq) for seq in sequences.values())) != 1:
        raise ValueError("All sequences must have the same length.")
    
    tree.branch_length = branch_length

    n_chars = len(list(sequences.values())[0])  # number of sites (characters) in the sequences

    # calculate the GTR transition rate matrix Q and its diagonalisation
    Q = GTR_Q_matrix(r_params, pi_params)
    tree.S, tree.lambda_vals, tree.S_inv = diagonalise_GTR_Q_matrix(Q)

    tree.pi_params = pi_params  # store GTR frequency parameters

    # discrete gamma approximation for rate heterogeneity
    gamma_cdf_vals = np.arange(1, 2 * n_gamma_bins, 2) / (2 * n_gamma_bins)
    gamma_vals = gamma_dist.ppf(gamma_cdf_vals, a=alpha, scale=1 / alpha)

    # init empty likelihoods table for all sites
    tree.likelihoods = {site_index: None for site_index in range(n_chars)}

    # assume independence over sites: total likelihood is the product of likelihoods at each site
    likelihood = 1.0
    for site_index in range(n_chars):
        likelihood *= felsenstein_pruning(sequences, tree, site_index, gamma_vals)

    return likelihood


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

    n_gamma_bins = 4

    r_params = (r_AC, r_AG, r_AT, r_CG, r_CT, r_GT)
    pi_params = (pi_A, pi_C, pi_G, pi_T)

    likelihood = calc_likelihood(sequences, tree, branch_length, r_params, pi_params, alpha, n_gamma_bins)

    print(f"Likelihood of observing the sequences at the leaves of the tree: p(D | T, b, theta, alpha) = {likelihood}")

    tree_newick = tree.to_newick(include_lengths=True)

    svg_str = render_newick_svg(tree_newick, width=800)

    with open("phylo_tree.svg", "w") as f:
        f.write(svg_str)