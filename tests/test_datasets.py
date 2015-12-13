from src.datasets import (get_bin,
                          validate_bin_ranges)
from nose.tools import (assert_equal,
                        assert_raises)

def test_get_bin():
    """
    Test the `get_bin` function in the `datasets` extension.
    """

    vals = [0.1, 0.0001, 1.5, 1.99, 2.5]
    expected_bins = [1, -1, 1, 1, 2]
    bin_ranges = [(0.1, 2.0), (2.1, 3.0)]

    for i, val in enumerate(vals):
        assert_equal(get_bin(bin_ranges, val), expected_bins[i])


def test_validate_bin_ranges_invalid():
    """
    Test the `validate_bin_ranges` function given invalid `bin_ranges`
    values.
    """

    bin_ranges_invalid = [
        [(0.0, 19.9)], # One bin only
        [(0.1, 2.9), (3.0, 3.0)], # 3.0 == 3.0
        [(0.0, 1.1), (2.1, 2.9)], # 2.1 - 1.1 > 0.1
        [(5.1, 2.1), (2.0, 4.9)], # 5.1 > 2.1
        [(0, 1.0), (1.1, 2.0)], # 0 is int, not float
        [(0.1, 0.9), (1.0, 1.45)] # 1.45 has more than one decimal
                                  # place precision
        ]
    for bin_ranges in bin_ranges_invalid:
        assert_raises(ValueError, validate_bin_ranges, bin_ranges)


def test_validate_bin_ranges_valid():
    """
    Test the `validate_bin_ranges` function given valid `bin_ranges`
    values.
    """

    bin_ranges_valid = [(0.0, 90.1), (90.2, 104.5), (104.6, 150.9)]
    validate_bin_ranges(bin_ranges_valid)
