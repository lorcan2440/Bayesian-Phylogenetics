# external modules
import numpy as np
from scipy.stats import ttest_1samp, shapiro

# local modules
if __name__ == '__main__':
    import __init__

from proposals import propose_new_GTR_freqs


def test_propose_new_GTR_freqs():

    RNG_SEED = 42

    # test 1: check the function can be repeatedly called without error
    pi_params_list = np.array([[0.3, 0.25, 0.1, 0.35]])
    for _ in range(100):
        pi_params, ratio = propose_new_GTR_freqs(pi_params_list[-1], rng_seed=RNG_SEED)
        pi_params_list = np.vstack((pi_params_list, pi_params))
        assert np.isclose(np.sum(pi_params), 1.0)
        assert np.all(0.0 <= pi_params) and np.all(pi_params <= 1.0)
    
    # test 3: check the function works near the upper boundary
    pi_params = np.array([0.99, 0.005, 0.0025, 0.0025])
    pi_params, ratio = propose_new_GTR_freqs(pi_params, eta=np.array([0.03, -0.03, 0.0]), rng_seed=RNG_SEED)
    assert np.isclose(np.sum(pi_params), 1.0)
    assert np.all(0.0 <= pi_params) and np.all(pi_params <= 1.0)

    # test 4: check the function works near the lower boundary
    pi_params = np.array([0.005, 0.99, 0.0025, 0.0025])
    pi_params, ratio = propose_new_GTR_freqs(pi_params, eta=np.array([-0.03, 0.03, 0.0]), rng_seed=RNG_SEED)
    assert np.isclose(np.sum(pi_params), 1.0)
    assert np.all(0.0 <= pi_params) and np.all(pi_params <= 1.0)

    # test 5: check the function works for a perturbation that would take a base out of bounds
    pi_params = np.array([0.3, 0.4, 0.1, 0.2])
    pi_params, ratio = propose_new_GTR_freqs(pi_params, eta=np.array([1.0, -1.0, 0.5]), rng_seed=RNG_SEED)
    assert np.isclose(np.sum(pi_params), 1.0)
    assert np.all(0.0 <= pi_params) and np.all(pi_params <= 1.0)


test_propose_new_GTR_freqs()
