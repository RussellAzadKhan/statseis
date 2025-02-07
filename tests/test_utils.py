import pytest
from statseis.utils import min_max_median_mean

@pytest.fixture
def sample_numbers():
    return (0, 5, 10)

def test_min_max_median_mean(sample_numbers):
    assert min_max_median_mean(sample_numbers) == (0, 10, 5, 5)