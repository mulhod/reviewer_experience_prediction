"""
Test various functions/classes in the `experiments` module.

A MongoDB database must be up and running and accessible at port 37017
on `localhost`. 
"""
from itertools import chain

import numpy as np
from nose2.compat import unittest
from nose.tools import (assert_equal,
                        assert_raises)

from src.mongodb import connect_to_db
from src.experiments import ExperimentalData

class ExperimentalDataTestCase(unittest.TestCase):
    """
    Test the `ExperimentalData` class.
    """

    @classmethod
    def setUpClass(self):
        """
        Set up the tests, mainly just creating a connection to the
        running MongoDB database.
        """

        self.db = connect_to_db('localhost', 37017)
        self.prediction_label = 'total_game_hours'

    def test_ExperimentalData_invalid(self):
        """
        Test the `ExperimentalData` class given invalid arguments.
        """

        kwargs = [
            # `games` contains unrecognized entries
            dict(games=set(['Dota']),
                 max_partitions=4),
            # `test_games` contains unrecognized entries
            dict(games=set(['Dota_2']),
                 test_games=set(['Dota']),
                 max_partitions=4),
            # `games` is empty
            dict(games=set(),
                 max_partitions=4),
            # `batch_size` < 1
            dict(games=set(['Dota_2']),
                 n_partition=100,
                 batch_size=0),
            # `max_partitions` < 0
            dict(games=set(['Dota_2']),
                 max_partitions=-1),
            # `n_partition` must be specified if `max_partitions` is not
            dict(games=set(['Dota_2'])),
            # `games` == `test_games`, but `max_test_samples` > -1
            dict(games=set(['Dota_2', 'Arma_3']),
                 test_games=set(['Dota_2', 'Arma_3']),
                 n_partition=100,
                 max_test_samples=50),
            # `games` != `test_games` and `max_test_samples` < 0
            dict(games=set(['Dota_2', 'Arma_3']),
                 test_games=set(['Counter_Strike']),
                 n_partition=100,
                 max_test_samples=-1),
            # `test_bin_ranges` is specified but not `bin_ranges`
            dict(games=set(['Dota_2', 'Arma_3']),
                 test_games=set(['Counter_Strike']),
                 n_partition=100,
                 test_bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 max_test_samples=50),
            # `test_bin_ranges` is not the same length as `bin_ranges`
            dict(games=set(['Dota_2', 'Arma_3']),
                 test_games=set(['Counter_Strike']),
                 n_partition=100,
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 test_bin_ranges=[(0.0, 199.1), (199.2, 876.4), (876.5, 3035.0),
                                  (3035.1, 8397.4)],
                 max_test_samples=50),
            # `n_grid_search_partition` < 1
            dict(games=set(['Dota_2', 'Arma_3']),
                 n_partition=100,
                 n_grid_search_partition=0)
            ]
        for _kwargs in kwargs:
            assert_raises(ValueError,
                          ExperimentalData,
                          self.db,
                          prediction_label=self.prediction_label,
                          **_kwargs)

    def test_ExperimentalData_valid(self):
        """
        Test the `ExperimentalData` class given valid arguments.
        """

        # Different combinations of arguments
        games_set = set(['Dota_2'])
        kwargs = [
            dict(n_partition=20,
                 n_grid_search_partition=30,
                 max_partitions=3,
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)]),
            # Not specifying `max_partitions`
            dict(n_partition=20,
                 n_grid_search_partition=30,
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)]),
            # Not specifying `n_partition`
            dict(n_grid_search_partition=30,
                 max_partitions=3,
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)]),
            # Specifying a set of test games (equal to games) and test
            # bin ranges
            dict(n_partition=20,
                 n_grid_search_partition=30,
                 max_partitions=3,
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 test_games=games_set,
                 test_bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)]),
            # Specifying a set of test games (different from games),
            # test bin ranges, and `max_test_samples`
            dict(n_partition=20,
                 n_grid_search_partition=30,
                 max_partitions=3,
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 test_games=set(['Dota_2', 'Arma_3']),
                 test_bin_ranges=[(0.0, 299.0), (299.1, 3306.8), (3306.9, 17443.4)],
                 max_test_samples=30),
            # Specifying a set of test games (completely different from
            # games) and test bin ranges
            dict(n_partition=20,
                 n_grid_search_partition=30,
                 max_partitions=3,
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 test_games=set(['Arma_3', 'Counter_Strike_Global_Offensive']),
                 test_bin_ranges=[(0.0, 294.4), (294.5, 2762.5), (2762.6, 15129.3)],
                 max_test_samples=30)
        ]

        for _kwargs in kwargs:
            
            # Make the dataset with the given combination of arguments
            exp_data = ExperimentalData(self.db,
                                        self.prediction_label,
                                        games_set,
                                        **_kwargs)

            # There should always be 3 grid search folds (hard-coded)
            assert_equal(len(exp_data.grid_search_set), 3)

            # Each grid search fold should contain no more than the
            # ceiling of `n_grid_search_partition`/3
            for fold in exp_data.grid_search_set:
                assert len(fold) <= np.ceil(_kwargs['n_grid_search_partition']/3)

            # The `test_set` attribute should be None if no test set was
            # generated; otherwise, it should contain of an array of IDs
            # that is less than or equal to the `max_test_samples`
            # parameter value
            test_set = exp_data.test_set
            test_games = _kwargs.get('test_games', None)
            max_test_samples = _kwargs.get('max_test_samples', -1)
            if max_test_samples < 0:
                assert not len(test_set)
            else:
                assert len(exp_data.test_set)
                if max_test_samples < 0:
                    # If negative, then no samples should be added
                    # Note: Not really sure if it can ever make it in
                    # here.
                    assert not len(exp_data.test_set)
                elif max_test_samples > 0:
                    # If greater than 0, the number of samples should be
                    # less than or equal to the value of the
                    # `max_test_samples` parameter
                    assert len(exp_data.test_set) <= max_test_samples
                else:
                    # If 0, then it means that there is no cap on the
                    # number of samples to include, so its length should
                    # resolve to true at the very least
                    assert len(exp_data.test_set)

            # The `num_datasets` attribute and length of the
            # `datasets_dict` attribute should both be less than or
            # equal to the value of the `max_partitions` parameter
            # (unless that value is 0, which signifies no limit)
            max_partitions = _kwargs.get('max_partitions', 0)
            if max_partitions:
                assert exp_data.num_datasets <= max_partitions
                assert len(exp_data.datasets_dict) <= max_partitions
            else:
                assert exp_data.num_datasets
                assert exp_data.datasets_dict

            # Each fold in the `datasets_dict` attribute should be less
            # than or equal to the value of the `n_partition` parameter
            # in length
            n_partition = _kwargs.get('n_partition', None)
            if n_partition:
                for fold in exp_data.datasets_dict:
                    assert len(exp_data.datasets_dict[fold]) <= n_partition
            else:
                for fold in exp_data.datasets_dict:
                    assert len(exp_data.datasets_dict[fold])

            # There absolutely should not be any IDs in common between
            # the varuous subsets of data
            grid_search_ids_set = set(chain(*exp_data.grid_search_set))
            datasets_ids_set = set(chain(*exp_data.datasets_dict.values()))
            assert_equal(grid_search_ids_set.difference(datasets_ids_set),
                         grid_search_ids_set)
            if len(exp_data.test_set):
                test_set_ids_set = set(exp_data.test_set)
                for ids_set in [grid_search_ids_set, datasets_ids_set]:
                    assert_equal(ids_set.difference(test_set_ids_set),
                                 ids_set)
