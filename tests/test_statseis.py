import pytest
import numpy as np

@pytest.fixture
def sample_numbers():
    return [0.1,0.2,0.3]

# from statseis.statseis import gamma_law_MLE

# print(gamma_law_MLE([0.1,0.2,0.3]))

# def test_gamma_law_MLE(sample_numbers):
#     assert gamma_law_MLE(sample_numbers) == 9.998000000000001


