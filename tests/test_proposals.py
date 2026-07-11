# external modules
import numpy as np

# local modules
if __name__ == '__main__':
    import __init__

from proposals import propose_new_GTR_freqs


def test_propose_new_GTR_freqs():

    pi_params = np.array([0.25, 0.25, 0.25, 0.25])  # uniform base frequencies
    ratio = None

    for i in range(1000):
        is_all_in_range = np.all((pi_params >= 0.0) & (pi_params <= 1.0))
        adds_to_one = np.isclose(np.sum(pi_params), 1.0)
        print(f"Iteration {i}:\tpi_params = {pi_params}\tAll positive: {is_all_in_range}\tAdds to one: {adds_to_one}\tRatio: {ratio}")
        pi_params, ratio = propose_new_GTR_freqs(pi_params)


test_propose_new_GTR_freqs()