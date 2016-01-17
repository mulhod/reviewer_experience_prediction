"""
Test various functions/classes in the `experiments` module.

A MongoDB database must be up and running and accessible at port 37017
on `localhost`. 
"""
from nose2.compat import unittest
from nose.tools import assert_raises

from src.mongodb import connect_to_db
from src.experiments import ExperimentalData

class ExperimentalDataTestCase(unittest.TestCase):
    """
    Test the `ExperimentalData` class.
    """

    @staticmethod
    def test_ExperimentalData_invalid():
        """
        Test the `ExperimentalData` class given invalid arguments.
        """

        db = connect_to_db('localhost', 37017)
        kwargs = [
            # `games` contains unrecognized entries
            dict(db=db,
                 games=set(['Dota']),
                 test_games=set(['Dota_2']),
                 n_partition=100,
                 prediction_label='total_game_hours',
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 max_partitions=4,
                 n_grid_search_partition=200),
            # `test_games` contains unrecognized entries
            dict(db=db,
                 games=set(['Dota_2']),
                 test_games=set(['Dota']),
                 n_partition=100,
                 prediction_label='total_game_hours',
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 max_partitions=4,
                 n_grid_search_partition=200),
            # `games` is empty
            dict(db=db,
                 games=set(),
                 test_games=set(['Dota_2']),
                 n_partition=100,
                 prediction_label='total_game_hours',
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 max_partitions=4,
                 n_grid_search_partition=200),
            # `batch_size` < 1
            dict(db=db,
                 games=set(['Dota_2']),
                 test_games=set(['Dota_2']),
                 n_partition=100,
                 prediction_label='total_game_hours',
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 max_partitions=4,
                 n_grid_search_partition=200,
                 batch_size=0),
            # `max_partitions` < 0
            dict(db=db,
                 games=set(['Dota_2']),
                 test_games=set(['Dota_2']),
                 n_partition=100,
                 prediction_label='total_game_hours',
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 max_partitions=-1,
                 n_grid_search_partition=200),
            # `n_partition` must be specified if `max_partitions` is not
            dict(db=db,
                 games=set(['Dota_2']),
                 test_games=set(['Dota_2']),
                 prediction_label='total_game_hours',
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 max_partitions=4,
                 n_grid_search_partition=200),
            # `games` == `test_games` and `max_test_samples` > -1
            dict(db=db,
                 games=set(['Dota_2']),
                 test_games=set(['Dota_2']),
                 n_partition=100,
                 prediction_label='total_game_hours',
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 max_partitions=4,
                 n_grid_search_partition=200,
                 max_test_samples=500),
            # `games` != `test_games` and `max_test_samples` < 0
            dict(db=db,
                 games=set(['Dota_2', 'Arma_3']),
                 test_games=set(['Counter_Strike']),
                 n_partition=100,
                 prediction_label='total_game_hours',
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 max_partitions=4,
                 n_grid_search_partition=200,
                 max_test_samples=-1)
            ]
        for _kwargs in kwargs:
            assert_raises(ValueError, ExperimentalData, *_kwargs)
'''
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
'''