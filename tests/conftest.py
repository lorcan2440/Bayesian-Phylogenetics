# pragma: no cover
if __name__ == "__main__":
    import __init__  # noqa

# external imports
import pytest
import numpy as np
from likelihood import PhyloTree


@pytest.fixture(
    params=[
        (
            np.array([0.2, 0.1, 0.4, 0.5, 0.8, 1.0]),
            np.array([0.3, 0.2, 0.2, 0.3]),
        ),
        (
            np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1]),
            np.array([0.25, 0.25, 0.25, 0.25]),
        ),
        (
            np.array([0.05, 0.7, 0.2, 1.1, 0.3, 0.9]),
            np.array([0.1, 0.4, 0.3, 0.2]),
        ),
    ],
    ids=["baseline", "symmetric_pi", "biased_pi"],
)
def gtr_params(request: pytest.FixtureRequest) -> tuple[np.ndarray, np.ndarray]:
    return request.param


@pytest.fixture
def create_test_tree():

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

    return sequences, tree, branch_length, alpha