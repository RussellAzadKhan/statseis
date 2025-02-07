# import pytest
# import numpy as np
# from statseis.mc import freq_mag_dist

# def test_freq_mag_dist():
#     # Create a sample list of magnitudes for testing
#     magnitudes = [3.0, 3.1, 3.5, 4.0, 4.3, 4.5, 5.0, 5.2, 5.5, 6.0]
#     mbin = 0.1

#     # Call the function
#     bins, events_per_bin, cum_events = freq_mag_dist(magnitudes, mbin)

#     # Test if the returned bins are correct and evenly spaced based on the given mbin
#     expected_bins = np.arange(3.0, 6.1, 0.1)
#     assert np.allclose(bins, expected_bins), f"Bins mismatch: expected {expected_bins}, got {bins}"

#     # Test if the number of events per bin is calculated correctly
#     expected_events_per_bin = np.array([2, 1, 1, 1, 2, 2, 1, 1, 1, 1])
#     assert np.all(events_per_bin == expected_events_per_bin), f"Events per bin mismatch: expected {expected_events_per_bin}, got {events_per_bin}"

#     # Test if the cumulative distribution is calculated correctly
#     expected_cum_events = np.cumsum(expected_events_per_bin)
#     assert np.all(cum_events == expected_cum_events), f"Cumulative distribution mismatch: expected {expected_cum_events}, got {cum_events}"

#     # Edge Case: Test with a single magnitude value
#     magnitudes_single = [4.5]
#     bins_single, events_single, cum_single = freq_mag_dist(magnitudes_single, mbin)
#     assert len(bins_single) == 1, f"Expected 1 bin, got {len(bins_single)}"
#     assert events_single[0] == 1, f"Expected 1 event in the bin, got {events_single[0]}"

#     # Edge Case: Test with no magnitudes (empty array)
#     magnitudes_empty = []
#     bins_empty, events_empty, cum_empty = freq_mag_dist(magnitudes_empty, mbin)
#     assert len(bins_empty) == 0, f"Expected no bins for empty input, got {len(bins_empty)}"
#     assert len(events_empty) == 0, f"Expected no events for empty input, got {len(events_empty)}"
#     assert len(cum_empty) == 0, f"Expected no cumulative events for empty input, got {len(cum_empty)}"
