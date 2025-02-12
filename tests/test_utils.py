import pytest
from statseis.utils import min_max_median_mean
import numpy as np
import pandas as pd

@pytest.fixture
def sample_numbers():
    return (0, 5, 10)

def test_min_max_median_mean(sample_numbers):
    assert min_max_median_mean(sample_numbers) == (0, 10, 5, 5)

from statseis.utils import no_nans_or_infs

# def test_no_nans_or_infs_dataframe():
#     data = {
#         'A': [1, 2, np.nan, 4, np.inf],
#         'B': [5, np.inf, 7, 8, np.nan]
#     }
#     df = pd.DataFrame(data)
    
#     result = no_nans_or_infs(df, metric='A').reset_index(drop=True)
#     expected_result = pd.DataFrame({'A': [1, 2, 4], 'B': [5, np.inf, 8]})
#     pd.testing.assert_frame_equal(result, expected_result)