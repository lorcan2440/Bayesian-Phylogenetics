# unit tests for likelihood.py

import numpy as np
from scipy.linalg import expm

# local imports
if __name__ == '__main__':
    import __init__

from likelihood import GTR_Q_matrix, diagonalise_GTR_Q_matrix, calc_transition_probability_matrix

# test 1: check that the matrix exponential calculation is correct using scipy expm function
# test 2: check that the likelihood calculation for a single site matches PAML's output
# test 3: check that the likelihood calculation for multiple sites matches PAML's output


def test_matrix_exponential():

    r_params = np.array([0.2, 0.1, 0.4, 0.5, 0.8, 1.0])
    pi_params = np.array([0.3, 0.2, 0.2, 0.3])

    Q = GTR_Q_matrix(r_params, pi_params)

    branch_length = 0.875
    gamma = 1.25
    dt = branch_length * gamma

    # calculate exp(Q * dt) using our functions
    Q_eig = diagonalise_GTR_Q_matrix(Q, pi_params)
    P = calc_transition_probability_matrix(Q_eig, branch_length, gamma)

    # calculate exp(Q * dt) using scipy's expm function
    P_scipy = expm(Q * dt)

    # check that the two matrices are close
    assert np.allclose(P, P_scipy, atol=1e-6), "Matrix exponential calculation does not match scipy's expm function"