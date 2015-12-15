from nose2.compat import unittest
from nose.tools import (assert_equal,
                        assert_raises)

from src.datasets import (get_bin,
                          get_bin_ranges,
                          validate_bin_ranges)

def test_get_bin():
    """
    Test the `get_bin` function.
    """

    vals = [0.1, 0.0001, 1.5, 1.99, 2.5]
    expected_bins = [1, -1, 1, 1, 2]
    bin_ranges = [(0.1, 2.0), (2.1, 3.0)]

    for i, val in enumerate(vals):
        assert_equal(get_bin(bin_ranges, val), expected_bins[i])


class GetBinRangesTestCase(unittest.TestCase):
    """
    Test the `get_bin_ranges` function.
    """

    @staticmethod
    def test_get_bin_ranges_invalid():
        """
        Test the `get_bin_ranges` function given invalid `bin_ranges`
        arguments.
        """

        args = [
            [0.5, 0.5], # `_min` and `_max` are equal
            [0.5000000001, 0.5], # `_min` and `_max` are "almost" equal
            [0.5, 0.50000001], # `_min` and `_max` are "almost" equal
            [0.5, 0.2], # `_min` is greater than `_max`
            [0.0, 100.0, 5, -1.0], # `factor` is not non-zero/positive
            [0.0, 100.0, 5, 0.0] # `factor` is not non-zero
            ]
        for _args in args:
            assert_raises(ValueError, get_bin_ranges, *_args)

    @staticmethod
    def test_get_bin_ranges_valid():
        """
        Test the `get_bin_ranges` function with some valid inputs.
        """

        args = [
            [0.0, 5.0, 5, 1.0],
            [0.0, 5.0, 5, 1.5],
            [0.0, 900.0, 3, 5.0],
            [1.0, 2.0, 2, 2.0],
        ]
        expected_outputs = [
            [(0.0, 1.0), (1.1, 2.0), (2.1, 3.0), (3.1, 4.0), (4.1, 5.0)],
            [(0.0, 0.4), (0.5, 1.0), (1.1, 1.9), (2.0, 3.2), (3.3, 5.0)],
            [(0.0, 29.0), (29.1, 174.2), (174.3, 900.0)],
            [(1.0, 1.3), (1.4, 2.0)]
            ]
        for _args, _expected_outputs in zip(args, expected_outputs):
            assert_equal(get_bin_ranges(*_args), _expected_outputs)


class ValidateBinRangesTestCase(unittest.TestCase):
    """
    Test the `validate_bin_ranges` method.
    """

    @staticmethod
    def test_validate_bin_ranges_invalid():
        """
        Test the `validate_bin_ranges` function given invalid
        `bin_ranges` values.
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

    @staticmethod
    def test_validate_bin_ranges_valid():
        """
        Test the `validate_bin_ranges` function given valid
        `bin_ranges` values.
        """

        bin_ranges_valid = [(0.0, 90.1), (90.2, 104.5), (104.6, 150.9)]
        assert_equal(validate_bin_ranges(bin_ranges_valid), None)
