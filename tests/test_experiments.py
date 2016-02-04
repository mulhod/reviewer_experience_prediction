"""
Test various functions/classes in the `experiments` module.

A MongoDB database must be up and running and accessible at port 37017
on `localhost`. 
"""
from itertools import chain
from collections import Counter

import pudb
import numpy as np
from schema import SchemaError
from nose2.compat import unittest
from nose.tools import (assert_equal,
                        assert_raises)

from src import (LEARNER_DICT,
                 DEFAULT_PARAM_GRIDS,
                 parse_non_nlp_features_string)
from src.mongodb import connect_to_db
from src.experimental import (ExperimentalData,
                              CVExperimentConfig)

class ExperimentalDataTestCase(unittest.TestCase):
    """
    Test the `ExperimentalData` class.
    """

    db = connect_to_db('localhost', 37017)
    prediction_label = 'total_game_hours'

    def test_ExperimentalData_invalid(self):
        """
        Test the `ExperimentalData` class given invalid arguments.
        """

        kwargs = [
            # `games` contains unrecognized entries
            dict(games=set(['Dota']),
                 folds=0,
                 grid_search_folds=0),
            # `test_games` contains unrecognized entries
            dict(games=set(['Dota_2']),
                 test_games=set(['Dota']),
                 folds=0,
                 grid_search_folds=0),
            # `games` is empty
            dict(games=set(),
                 folds=0,
                 grid_search_folds=0),
            # `batch_size` < 1
            dict(games=set(['Dota_2']),
                 folds=0,
                 grid_search_folds=0,
                 batch_size=0),
            # `test_bin_ranges` is specified but not `bin_ranges`
            dict(games=set(['Dota_2', 'Arma_3']),
                 test_games=set(['Counter_Strike']),
                 folds=0,
                 grid_search_folds=0,
                 test_bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 test_size=50),
            # `test_bin_ranges` is not the same length as `bin_ranges`
            dict(games=set(['Dota_2', 'Arma_3']),
                 test_games=set(['Counter_Strike']),
                 folds=0,
                 grid_search_folds=0,
                 bin_ranges=[(0.0, 200.1), (200.2, 1896.7), (1896.8, 9041.4),
                             (9041.5, 16435.0)],
                 test_bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 test_size=50),
            # `sampling` is an invalid value (e.g. not "even" or
            # "stratified")
            dict(games=set(['Dota_2']),
                 folds=0,
                 grid_search_folds=0,
                 sampling='random'),
            # `folds` is set to 0, but `fold_size` is set to a non-zero value
            dict(games=set(['Dota_2']),
                 folds=0,
                 fold_size=100,
                 grid_search_folds=0),
            # `grid_search_folds` is set to 0, but
            # `grid_search_fold_size` is set to a non-zero value
            dict(games=set(['Dota_2']),
                 folds=0,
                 grid_search_folds=0,
                 grid_search_fold_size=50),
            # `fold_size` is set to 0, but `folds` is set to a non-zero value
            dict(games=set(['Dota_2']),
                 folds=5,
                 fold_size=0,
                 grid_search_folds=0),
            # `grid_search_fold_size` is set to 0, but `grid_search_folds` is
            # set to a non-zero value
            dict(games=set(['Dota_2']),
                 folds=0,
                 grid_search_folds=5,
                 grid_search_fold_size=0)
            ]

        # The number of folds and fold/test set size parameters must be
        # non-negative, so test setting them to a negative value
        kwargs += [dict(games=set(['Dota_2']),
                        folds=-1,
                        grid_search_folds=0),
                   dict(games=set(['Dota_2']),
                        folds=0,
                        grid_search_folds=-1),
                   dict(games=set(['Dota_2']),
                        folds=5,
                        fold_size=-1,
                        grid_search_folds=0),
                   dict(games=set(['Dota_2']),
                        folds=0,
                        grid_search_folds=3,
                        grid_search_fold_size=-1),
                   dict(games=set(['Dota_2']),
                        folds=0,
                        grid_search_folds=0,
                        test_size=-1)]

        for _kwargs in kwargs:
            assert_raises(ValueError,
                          ExperimentalData,
                          db=self.db,
                          prediction_label=self.prediction_label,
                          **_kwargs)

    def test_ExperimentalData_valid(self):
        """
        Test the `ExperimentalData` class given valid arguments.
        """

        # Different combinations of arguments
        games_set = set(['Dota_2'])
        kwargs = [
            dict(folds=3,
                 fold_size=15,
                 grid_search_folds=3,
                 grid_search_fold_size=15,
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 sampling='even'),
            # Setting `folds` to 0 (i.e., not generating a main training
            # set)
            dict(folds=0,
                 grid_search_folds=3,
                 grid_search_fold_size=15,
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)]),
            # Setting `grid_search_folds` to 0 (i.e., not generating a
            # grid search set)
            dict(folds=3,
                 fold_size=15,
                 grid_search_folds=0,
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)]),
            # Specifying a set of test games (equal to games) and test
            # bin ranges
            dict(folds=3,
                 fold_size=15,
                 grid_search_folds=0,
                 test_games=games_set,
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 test_bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)]),
            # Specifying a set of test games (different from games),
            # test bin ranges, and `test_size`
            dict(folds=3,
                 fold_size=15,
                 grid_search_folds=0,
                 test_size=30,
                 test_games=set(['Dota_2', 'Arma_3']),
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 test_bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)]),
            # Specifying a set of test games (completely different from
            # games) and test bin ranges
            dict(folds=3,
                 fold_size=15,
                 grid_search_folds=0,
                 test_size=30,
                 test_games=set(['Football_Manager_2015', 'Arma_3']),
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 test_bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)]),
        ]

        for _kwargs in kwargs:
            
            # Make the dataset with the given combination of arguments
            exp_data = ExperimentalData(db=self.db,
                                        prediction_label=self.prediction_label,
                                        games=games_set,
                                        **_kwargs)

            # Each data fold should contain no more than the
            # ceiling of the corresponding fold size parameter
            for partition in ['training', 'grid_search']:
                prefix = 'grid_search_' if partition == 'grid_search' else ''
                n_folds = eval('exp_data.{0}_set'.format(partition))
                if n_folds:
                    n_folds = len(n_folds)

                # If the corresponding "folds" parameter was not 0
                if _kwargs['{0}folds'.format(prefix)]:

                    # The number of folds should be less than or equal
                    # to the corresponding "folds" input parameter value
                    # (and should not be 0 in any case)
                    assert (n_folds
                            and n_folds <= _kwargs['{0}folds'.format(prefix)])

                    # Each fold's number of samples should be less than
                    # or equal to the value of the corresponding
                    # "fold_size" input parameter (and should definitely
                    # not be empty)
                    for fold in eval('exp_data.{0}_set'.format(partition)):
                        assert (len(fold) and
                                len(fold) <= _kwargs['{0}fold_size'.format(prefix)])

                # Otherwise
                else:

                    # There should be no folds in the dataset
                    assert not n_folds

            # The `test_set` attribute should be an empty array if no
            # test set was generated; otherwise, it should contain of an
            # array of IDs that is less than or equal to the value of
            # the `test_size` parameter value
            test_set = exp_data.test_set
            test_set_length = len(test_set)
            test_games = _kwargs.get('test_games', None)
            test_size = _kwargs.get('test_size', 0)
            if test_size:
                assert (test_set_length
                        and test_set_length <= test_size)
            else:
                assert not test_set_length

            # No sample ID should occur twice
            samples = []
            if len(exp_data.test_set):
                samples.extend(exp_data.test_set)
            if exp_data.training_set:
                samples.extend(chain(*exp_data.training_set))
            if exp_data.grid_search_set:
                samples.extend(chain(*exp_data.grid_search_set))
            sample_counter = Counter(samples)
            assert all(sample_counter[_id] == 1 for _id in sample_counter)

            # The `sampling` attribute should reflect the value passed
            # in as the parameter value (or the default value)
            assert_equal(_kwargs.get('sampling', 'stratified'),
                         exp_data.sampling)


class CVExperimentConfigTestCase(unittest.TestCase):
    """
    Test the `CVExperimentConfig` class.
    """

    db = connect_to_db('localhost', 37017)
    prediction_label = 'total_game_hours'

    def test_CVExperimenConfig_invalid(self):
        """
        Test the `CVExperimentConfig` class.
        """

        learner_abbrs = ['perc', 'pagr']
        learners = [LEARNER_DICT[learner] for learner in learner_abbrs]
        non_nlp_features = parse_non_nlp_features_string('all', 'total_game_hours')
        param_grids = [DEFAULT_PARAM_GRIDS[LEARNER_DICT[learner]]
                       for learner in learner_abbrs]
        valid_kwargs = dict(db=self.db,
                            games=set(['Dota_2']),
                            learners=learners,
                            param_grids=param_grids,
                            training_rounds=10,
                            training_samples_per_round=100,
                            grid_search_samples_per_fold=50,
                            non_nlp_features=non_nlp_features,
                            prediction_label=self.prediction_label,
                            objective='pearson_r',
                            data_sampling='even',
                            grid_search_folds=5,
                            hashed_features=100000,
                            nlp_features=True,
                            bin_ranges=[(0.0, 225.1), (225.2, 2026.2),
                                        (2026.3, 16435.0)],
                            lognormal=False,
                            power_transform=False,
                            majority_baseline=True,
                            rescale=True)
        invalid_kwargs_list = [
            # Invalid `db` value
            dict(db='db',
                 **{p: v for p, v in valid_kwargs.items() if p != 'db'}),
            # Invalid games in `games` parameter value
            dict(games=set(['Dota']),
                 **{p: v for p, v in valid_kwargs.items() if p != 'games'}),
            # Invalid `learners` parameter value
            dict(learners=learner_abbrs,
                 **{p: v for p, v in valid_kwargs.items() if p != 'learners'}),
            # Invalid parameter grids in `param_grids` parameter value
            dict(param_grids=[dict(a=1, b=2), dict(c='g', d=True)],
                 **{p: v for p, v in valid_kwargs.items() if p != 'param_grids'}),
            # `learners` and `param_grids` of unequal size
            dict(learners=[learners[0]],
                 **{p: v for p, v in valid_kwargs.items() if p != 'learners'}),
            # Invalid `training_rounds` parameter value (must be int)
            dict(training_rounds=2.0,
                 **{p: v for p, v in valid_kwargs.items() if p != 'training_rounds'}),
            # Invalid `training_rounds` parameter value (must be greater
            # than 0)
            dict(training_rounds=0,
                 **{p: v for p, v in valid_kwargs.items() if p != 'training_rounds'})
            ]
        for kwargs in invalid_kwargs_list:
            assert_raises(SchemaError, CVExperimentConfig, **kwargs)
