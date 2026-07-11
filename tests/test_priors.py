import numpy as np
from scipy.integrate import nquad

# local modules
if __name__ == '__main__':
    import __init__

import priors
from priors import calc_prior_density, calc_prior_density_alpha, calc_prior_density_pi_params, \
    calc_prior_density_r_params, calc_prior_density_branch_lengths, calc_prior_density_tree
from utils import get_logger


logger = get_logger()


def test_prior_density_pi_params():

    # NOTE: the second test case takes about a minute to run

    for alpha_param in [np.ones(4), np.array([0.5, 1.0, 1.5, 2.0])]:

        # set constant in priors module
        priors.PI_PARAMS_DIRICHLET_ALPHA = alpha_param

        # test that the prior density integrates to 1 over the simplex of valid pi_params (i.e., the 3-simplex in 4D space)
        integrated_prior, error = nquad(
            lambda pi_C, pi_G, pi_T: calc_prior_density_pi_params(np.array([1.0 - pi_C - pi_G - pi_T, pi_C, pi_G, pi_T]), log_prior=False),
            [lambda pi_G, pi_T, *_: [0.0, 1.0 - pi_G - pi_T], lambda pi_T, *_: [0.0, 1.0 - pi_T], [0.0, 1.0]],
            opts={'epsabs': 1e-4, 'epsrel': 1e-4}
        )

        print(error, integrated_prior)

        assert error < 1e-4, f"Integration error is too high: {error}"
        assert np.isclose(integrated_prior, 1.0, atol=1e-4), (
            f"Integrated prior should be 1, got {integrated_prior} (error estimate: {error})"
        )


if __name__ == '__main__':
    test_prior_density_pi_params()