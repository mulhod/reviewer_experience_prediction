"""
Test various functions in the `datasets` module.
"""
import numpy as np
from nose2.compat import unittest
from nose.tools import (assert_equal,
                        assert_raises)

from src.datasets import (get_bin,
                          get_bin_ranges,
                          validate_bin_ranges,
                          compute_label_value)

def test_get_bin():
    """
    Test the `get_bin` function.
    """

    vals = [0.1, 0.0001, 1.5, 1.99, 2.5]
    expected_bins = [1, None, 1, 1, 2]
    bin_ranges = [(0.1, 2.0), (2.1, 3.0)]

    for i, (val, bin_) in enumerate(zip(vals, expected_bins)):
        assert_equal(get_bin(bin_ranges, val), bin_)


class GetBinRangesTestCase(unittest.TestCase):
    """
    Test the `get_bin_ranges` function.
    """

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
            [0.0, 100.0, 5, 0.0], # `factor` is not non-zero
            [0.0, 74018341812517056.0, 2, 5.0], # `_max` is too big
            [0.0, 100.235634] # `_max` is more precise than to one decimal
                              # place
            ]
        for _args in args:
            assert_raises(ValueError, get_bin_ranges, *_args)

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

    def test_validate_bin_ranges_valid():
        """
        Test the `validate_bin_ranges` function given valid
        `bin_ranges` values.
        """

        bin_ranges_valid = [(0.0, 90.1), (90.2, 104.5), (104.6, 150.9)]
        assert_equal(validate_bin_ranges(bin_ranges_valid), None)


class ComputeLabelValueTestCase(unittest.TestCase):
    """
    Test the `compute_label_value` function.
    """

    def test_compute_label_value_negative(self):
        """
        Test the `compute_label_value` function with a negative input
        value.
        """

        value = -1.0
        label = "total_game_hours"
        with self.assertRaises(ValueError):
            compute_label_value(value, label)

    def test_compute_label_value_kwargs_conflict(self):
        """
        Test the `compute_label_value` function with conflicting keyword
        arguments.
        """

        value = 1.0
        label = "total_game_hours"
        with self.assertRaises(ValueError):
            compute_label_value(value, label, lognormal=True, power_transform=2.0)

    def test_compute_label_value_not_found_in_bin_ranges(self):
        """
        Test the `compute_label_value` function with a value that cannot
        be located in the given bin ranges.
        """

        value = 34.0
        label = "total_game_hours"
        bin_ranges = [(0.0, 5.0), (5.1, 20.2), (20.3, 33.9)]
        with self.assertRaises(ValueError):
            compute_label_value(value, label, bin_ranges=bin_ranges)

    def test_compute_label_value_lognormal(self):
        """
        Test the `compute_label_value` function with the `lognormal`
        keyword argument set to True.
        """

        value = 45.9
        label = "total_game_hours"
        expected_value = np.log(value)
        assert_equal(compute_label_value(value, label, lognormal=True),
                     expected_value)

    def test_compute_label_value_power_transform(self):
        """
        Test the `compute_label_value` function with the
        `power_transform` keyword argument set to 2.0.
        """

        value = 45.9
        label = "total_game_hours"
        power = 2.0
        expected_value = value ** 2.0
        assert_equal(compute_label_value(value, label, power_transform=power),
                     expected_value)

    def test_compute_label_value_bin_ranges(self):
        """
        Test the `compute_label_value` function with the `bin_ranges`
        keyword argument set.
        """

        value = 34.0
        label = "total_game_hours"
        bin_ranges = [(0.0, 5.0), (5.1, 20.2), (20.3, 34.0)]
        expected_value = 3
        assert_equal(compute_label_value(value, label, bin_ranges=bin_ranges),
                     expected_value)

    def test_compute_label_value_percentage(self):
        """
        Test the `compute_label_value` function with one of the
        percentage labels.
        """

        value = 0.5
        label = "found_helpful_percentage"
        expected_value = 100.0 * value
        assert_equal(compute_label_value(value, label), expected_value)
