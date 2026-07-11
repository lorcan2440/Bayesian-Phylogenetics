# unit tests for likelihood.py

# built-in modules
import re
import shutil
import subprocess
from pathlib import Path

# external modules
import numpy as np
from scipy.linalg import expm

# local modules
if __name__ == '__main__':
    import __init__

from likelihood import GTR_Q_matrix, calc_log_likelihood, diagonalise_GTR_Q_matrix, calc_transition_probability_matrix
from utils import get_logger


logger = get_logger()


def test_matrix_exponential(gtr_params: tuple[np.ndarray, np.ndarray]):

    '''
    Test that our matrix exponential calculation using diagonalisation exploiting the structure of the CTMC matrix
    matches scipy's more general expm function. Three different test cases are provided by the gtr_params fixture.
    '''

    # get parameters for this test case and assemble the Q matrix
    r_params, pi_params = gtr_params
    Q = GTR_Q_matrix(r_params, pi_params)

    # calculate a time step
    branch_length = 0.875
    gamma = 1.25
    dt = branch_length * gamma

    # calculate exp(Q * dt) using our functions
    Q_eig = diagonalise_GTR_Q_matrix(Q, pi_params)
    P = calc_transition_probability_matrix(Q_eig, branch_length, gamma)

    # calculate exp(Q * dt) using scipy's expm function
    P_scipy = expm(Q * dt)

    # check they are close
    if not np.allclose(P, P_scipy, atol=1e-9):
        logger.error("Matrix exponential calculation does not match scipy's expm function.")
        raise AssertionError("Matrix exponential calculation does not match scipy's expm function.")


def test_likelihood(create_test_tree):

    # get the extant taxa sequences, the phylogenetic tree, the branch lengths and the gamma shape parameter
    # from the fixture (a single fixed test case)
    sequences, tree, branch_length, alpha = create_test_tree
    tree.branch_length = branch_length
    
    # define directories for PAML
    repo_root = Path(__file__).resolve().parents[1]
    SEQUENCES_FILE_PATH = repo_root / "tests" / "paml_config" / "test_case.phy"
    TREE_FILE_PATH = repo_root / "tests" / "paml_config" / "test_tree_newick.txt"
    PAML_CONFIG_FILE_PATH = repo_root / "tests" / "paml_config" / "baseml.ctl"
    PAML_OUTPUT_FILE_PATH = repo_root / "tests" / "paml_output" / "results.txt"
    PAML_BASEML_CTL_SRC = repo_root / "paml" / "src" / "baseml.ctl"
    PAML_BASEML_SRC = repo_root / "paml" / "bin" / "baseml.exe"
    PAML_OUTPUT_DIR = repo_root / "tests" / "paml_output"

    # copy the template baseml.ctl file provided by PAML into our test directory for editing
    shutil.copy(PAML_BASEML_CTL_SRC, PAML_CONFIG_FILE_PATH)

    # write our sequences and tree to input files for PAML
    num_taxa = len(sequences)
    num_sites = len(list(sequences.values())[0])
    with open(SEQUENCES_FILE_PATH, "w") as f:
        f.write(f"{num_taxa} {num_sites}\n")
        for taxon, sequence in sequences.items():
            f.write(f"{taxon}  {sequence}\n")

    with open(TREE_FILE_PATH, "w") as f:
        f.write(tree.to_newick(include_lengths=True) + "\n")

    # edit the baseml.ctl file to point to the test files and set input parameters for the GTR+Γ model
    with open(PAML_CONFIG_FILE_PATH, "r+") as f:
        ctl_lines = f.readlines()
        replacement_vals = {
            "seqfile": SEQUENCES_FILE_PATH,
            "treefile": TREE_FILE_PATH,
            "outfile": PAML_OUTPUT_FILE_PATH,
            "alpha": alpha,     # for gamma distribution
            "fix_blength": 2,   # fix branch lengths to our inputs
            "model": 7,         # for GTR
            "ndata": 1,
            "verbose": 1
        }

        for i, line in enumerate(ctl_lines):
            for key, value in replacement_vals.items():
                regex = rf"^\s*({key})\s*=\s*.*?(\s*\*.*)?$"
                replace = re.sub(regex, lambda m: f"{m.group(1)} = {value}{m.group(2) or ''}", line)
                if replace != line:
                    ctl_lines[i] = replace
                    break

        f.seek(0)
        f.writelines(ctl_lines)
        f.truncate()

    # run PAML's baseml program
    paml_output = subprocess.run(
        [str(PAML_BASEML_SRC), str(PAML_CONFIG_FILE_PATH)],
        cwd=str(PAML_OUTPUT_DIR),
        check=False,
        capture_output=True,
        text=True,
    )
    if paml_output.returncode != 0:
        has_results = PAML_OUTPUT_FILE_PATH.exists() and PAML_OUTPUT_FILE_PATH.stat().st_size > 0
        if not has_results:
            raise RuntimeError(
                f"Baseml failed with exit code {paml_output.returncode} and produced no results file"
            )
        # ignore the error code if results were still written
        # TODO: can we stop this error code happening?
        logger.warning(
            f"Baseml exited with code {paml_output.returncode}, "
            f"but results were written to {PAML_OUTPUT_FILE_PATH}"
        )

    # parse the parameters and log-likelihood from the PAML output file
    with open(PAML_OUTPUT_FILE_PATH, "r") as f:
        paml_lines = f.readlines()
        q_line_output = ""
        q_line_start = np.inf
        for i, line in enumerate(paml_lines):
            if line.startswith("lnL"):
                paml_log_likelihood = float(line.split()[4])
            if line.startswith("Rate parameters"):
                paml_r_params = tuple(float(x) for x in line.split()[2:])
            if line.startswith("Base frequencies"):
                paml_pi_params = tuple(float(x) for x in line.split()[2:])
            if line.startswith("Rate matrix Q"):
                q_line_start = i + 1  # Q matrix is printed in the next 4 lines
            if q_line_start <= i <= q_line_start + 3:
                q_line_output += f"{line}\n"
        # assemble Q numpy array then permute to match our order of T,C,A,G -> A,C,G,T
        Q_paml = np.array([[float(x) for x in line.split(' ') if x] for line in q_line_output.split('\n') if line])
        perm = [2, 1, 3, 0]
        Q_paml = Q_paml[np.ix_(perm, perm)]

    # use PAML's optimised rate parameters and base frequencies as input to Python's likelihood calculation
    # produce input parameters for Python - need to re-order PAML's T,C,A,G -> Python's A,C,G,T
    r_CT, r_AT, r_GT, r_AC, r_CG = paml_r_params
    r_AG = 1.0  # by definition
    pi_T, pi_C, pi_A, pi_G = paml_pi_params
    r_params = (r_AC, r_AG, r_AT, r_CG, r_CT, r_GT)
    pi_params = (pi_A, pi_C, pi_G, pi_T)

    logger.info(f"Rate parameters from PAML: {r_params}")
    logger.info(f"Base frequencies from PAML: {pi_params}")

    # check Q matrix is the same
    Q = GTR_Q_matrix(r_params, pi_params)
    if not np.allclose(Q, Q_paml, atol=1e-6):
        logger.error("Q matrix does not match PAML's output.")
        raise AssertionError("CTMC Q matrix does not match PAML's output.")

    # run our likelihood calculation and check it matches PAML's log-likelihood
    log_likelihood = calc_log_likelihood(sequences, tree, branch_length, r_params, pi_params, alpha,
        n_gamma_bins=4, calc_raw_likelihood=False)
    if not np.isclose(log_likelihood, paml_log_likelihood, atol=1e-6):
        logger.error("Log-likelihood does not match PAML's output.")
        raise AssertionError("Log-likelihood does not match PAML's output.")


if __name__ == "__main__":

    from tree import PhyloTree

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

    alpha = 0.5

    params = (sequences, tree, branch_length, alpha)

    test_likelihood(params)