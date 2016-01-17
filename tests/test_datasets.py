"""
Test various functions in the `datasets` module.
"""
from os.path import join
from shutil import rmtree
from tempfile import mkdtemp

from nose2.compat import unittest
from nose.tools import (assert_equal,
                        assert_raises)

from src.datasets import (get_bin,
                          get_bin_ranges,
                          get_game_files,
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
            [0.0, 100.0, 5, 0.0], # `factor` is not non-zero
            [0.0, 74018341812517056.0, 2, 5.0], # `_max` is too big
            [0.0, 100.235634] # `_max` is more precise than to one decimal
                              # place
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


class GetGameFilesTestCase(unittest.TestCase):
    """
    Test the `get_game_files` method.
    """

    @staticmethod
    def test_get_game_files_empty_games_string():
        """
        Test the `get_game_files` function given an empty games string.
        """

        # Test with an empty string
        assert_raises(ValueError, get_game_files, '')

    @staticmethod
    def test_get_game_files_sample_file_only():
        """
        Test the `get_game_files` function given a games string that
        only includes the "sample" file.
        """

        # Test with an empty string
        assert_raises(ValueError, get_game_files, 'sample.jsonlines')

    @staticmethod
    def test_get_game_files_invalid_directory():
        """
        Test the `get_game_files` function given an invalid directory.
        """

        # Make temporary_directory
        path = mkdtemp()

        # Test with an invalid directory
        assert_raises(FileNotFoundError, get_game_files,
                      'Dota_2,Counter_Strike',
                      join(path, 'nonexistent_directory'))

        # Remove the temporary directory
        rmtree(path)

    @staticmethod
    def test_get_game_files_empty_directory():
        """
        Test the `get_game_files` function given an empty directory.
        """

        # Make temporary_directory
        path = mkdtemp()

        assert_raises(ValueError, get_game_files, 'all', path)

        # Remove the temporary directory
        rmtree(path)

    @staticmethod
    def test_get_game_files_unrecognized_games():
        """
        Test the `get_game_files` function given unrecognized games.
        """

        for games_str in ['Dota', 'Dota,Arma', 'Dota_2,Arma', 'Dota,Arma_3']:
            assert_raises(FileNotFoundError, get_game_files, games_str)

    @staticmethod
    def test_get_game_files_valid_with_extension():
        """
        Test the `get_game_files` function given valid values with the
        ".jsonlines" extension.
        """

        assert_equal(get_game_files('Arma_3.jsonlines,Dota_2.jsonlines'),
                     ['Arma_3.jsonlines', 'Dota_2.jsonlines'])

    @staticmethod
    def test_get_game_files_valid_without_extension():
        """
        Test the `get_game_files` function given valid values without
        the ".jsonlines" extension.
        """

        assert_equal(get_game_files('Arma_3,Dota_2'),
                     ['Arma_3.jsonlines', 'Dota_2.jsonlines'])

    @staticmethod
    def test_get_game_files_valid_with_sample_file():
        """
        Test the `get_game_files` function given valid values including
        the sample file.
        """

        for sample_file in ['sample', 'sample.jsonlines']:
            assert_equal(get_game_files('Arma_3,Dota_2,{}'.format(sample_file)),
                         ['Arma_3.jsonlines', 'Dota_2.jsonlines'])
