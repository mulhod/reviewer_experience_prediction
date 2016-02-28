"""
Test various functions/classes in the `experiments` module + the
`CVConfig` class in the `util.cv_learn` module.

A MongoDB database must be up and running and accessible at port 37017
on `localhost`. 
"""
from os import (unlink,
                makedirs)
from shutil import rmtree
from os.path import (join,
                     exists,
                     dirname,
                     realpath)
from itertools import chain
from collections import Counter

import numpy as np
from schema import SchemaError
from nose2.compat import unittest
from nose.tools import (assert_equal,
                        assert_raises)
from pymongo.collection import Collection
from pymongo.errors import (AutoReconnect,
                            ConnectionFailure)

from util.cv_learn import CVConfig
from src.mongodb import connect_to_db
from src.datasets import validate_bin_ranges
from src.experiments import (make_cursor,
                             ExperimentalData)
from src import (LABELS,
                 LEARNER_DICT,
                 LEARNER_DICT_KEYS,
                 DEFAULT_PARAM_GRIDS,
                 OBJ_FUNC_ABBRS_DICT,
                 parse_non_nlp_features_string)

this_dir = dirname(realpath(__file__))

class MakeCursorTestCase(unittest.TestCase):
    """
    Test the `make_cursor` function.
    """

    db = None
    ids = None

    def setUp(self):

        # Connect to MongoDB collection
        try:
            self.db = connect_to_db('localhost', 37017)
        except AutoReconnect as e:
            raise ConnectionFailure('Could not connect to MongoDB client. Make '
                                    'sure a tunnel is set up (or some other method'
                                    ' is used) before running the tests.')

        # IDs to test
        self.ids = ['5690a60fe76db81bef5c46f8', '5690a60fe76db81bef5c275f',
                    '5690a60fe76db81bef5c49e9', '5690a60fe76db81bef5c3a67',
                    '5690a60fe76db81bef5c2d26', '5690a60fe76db81bef5c2756',
                    '5690a60fe76db81bef5c2bc9', '5690a60fe76db81bef5c3ab1',
                    '5690a60fe76db81bef5c3a71', '5690a60fe76db81bef5c2edf',
                    '5690a60fe76db81bef5c2f72', '5690a60fe76db81bef5c4305',
                    '5690a60fe76db81bef5c3ee9', '5690a60fe76db81bef5c4ab6',
                    '5690a60fe76db81bef5c43cf', '5690a60fe76db81bef5c47f1',
                    '5690a60fe76db81bef5c2b0b', '5690a60fe76db81bef5c4920',
                    '5690a60fe76db81bef5c49d9', '5690a60fe76db81bef5c3048',
                    '5690a60fe76db81bef5c4057', '5690a60fe76db81bef5c3902',
                    '5690a60fe76db81bef5c2702', '5690a60fe76db81bef5c461d',
                    '5690a60fe76db81bef5c4b2d', '5690a60fe76db81bef5c3176',
                    '5690a60fe76db81bef5c338a', '5690a60fe76db81bef5c2c01',
                    '5690a60fe76db81bef5c3836', '5690a60fe76db81bef5c3b07']

    def test_valid_id_strings_input(self):
        """
        Test `make_cursor` with valid sets of ID strings.
        """

        for _type in [list, np.array, iter, set, dict]:
            if _type is dict:
                ids = dict(zip(self.ids, self.ids))
            else:
                ids = _type(self.ids)
            make_cursor(self.db, id_strings=ids)


class ExperimentalDataTestCase(unittest.TestCase):
    """
    Test the `ExperimentalData` class.
    """

    try:
        db = connect_to_db('localhost', 37017)
    except AutoReconnect as e:
        raise ConnectionFailure('Could not connect to MongoDB client. Make '
                                'sure a tunnel is set up (or some other method'
                                ' is used) before running the tests.')
    prediction_label = 'total_game_hours'

    def test_ExperimentalData_invalid(self):
        """
        Test the `ExperimentalData` class given invalid arguments.
        """

        kwargs = [
            # `games` contains unrecognized entries
            dict(games=set(['Dota']),
                 folds=0,
                 fold_size=0,
                 grid_search_folds=0,
                 grid_search_fold_size=0),
            # `test_games` contains unrecognized entries
            dict(games=set(['Dota_2']),
                 test_games=set(['Dota']),
                 folds=0,
                 fold_size=0,
                 grid_search_folds=0,
                 grid_search_fold_size=0),
            # `games` is empty
            dict(games=set(),
                 folds=0,
                 fold_size=0,
                 grid_search_folds=0,
                 grid_search_fold_size=0),
            # `batch_size` < 1
            dict(games=set(['Dota_2']),
                 folds=0,
                 fold_size=0,
                 grid_search_folds=0,
                 grid_search_fold_size=0,
                 batch_size=0),
            # `test_bin_ranges` is specified but not `bin_ranges`
            dict(games=set(['Dota_2', 'Arma_3']),
                 test_games=set(['Counter_Strike']),
                 folds=0,
                 fold_size=0,
                 grid_search_folds=0,
                 grid_search_fold_size=0,
                 test_bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 test_size=50),
            # `test_bin_ranges` is not the same length as `bin_ranges`
            dict(games=set(['Dota_2', 'Arma_3']),
                 test_games=set(['Counter_Strike']),
                 folds=0,
                 fold_size=0,
                 grid_search_folds=0,
                 grid_search_fold_size=0,
                 bin_ranges=[(0.0, 200.1), (200.2, 1896.7), (1896.8, 9041.4),
                             (9041.5, 16435.0)],
                 test_bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 test_size=50),
            # `sampling` is an invalid value (e.g. not "even" or
            # "stratified")
            dict(games=set(['Dota_2']),
                 folds=0,
                 fold_size=0,
                 grid_search_folds=0,
                 grid_search_fold_size=0,
                 sampling='random'),
            # `folds` is set to 0, but `fold_size` is set to a non-zero value
            dict(games=set(['Dota_2']),
                 folds=0,
                 fold_size=100,
                 grid_search_folds=0,
                 grid_search_fold_size=0),
            # `grid_search_folds` is set to 0, but
            # `grid_search_fold_size` is set to a non-zero value
            dict(games=set(['Dota_2']),
                 folds=0,
                 fold_size=0,
                 grid_search_folds=0,
                 grid_search_fold_size=50),
            # `fold_size` is set to 0, but `folds` is set to a non-zero value
            dict(games=set(['Dota_2']),
                 folds=5,
                 fold_size=0,
                 grid_search_folds=0,
                 grid_search_fold_size=0),
            # `grid_search_fold_size` is set to 0, but `grid_search_folds` is
            # set to a non-zero value
            dict(games=set(['Dota_2']),
                 folds=0,
                 fold_size=0,
                 grid_search_folds=5,
                 grid_search_fold_size=0)
            ]

        # The number of folds and fold/test set size parameters must be
        # non-negative, so test setting them to a negative value
        kwargs += [dict(games=set(['Dota_2']),
                        folds=-1,
                        fold_size=0,
                        grid_search_folds=0,
                        grid_search_fold_size=0),
                   dict(games=set(['Dota_2']),
                        folds=0,
                        fold_size=0,
                        grid_search_folds=-1,
                        grid_search_fold_size=0),
                   dict(games=set(['Dota_2']),
                        folds=5,
                        fold_size=-1,
                        grid_search_folds=0,
                        grid_search_fold_size=0),
                   dict(games=set(['Dota_2']),
                        folds=0,
                        fold_size=0,
                        grid_search_folds=3,
                        grid_search_fold_size=-1),
                   dict(games=set(['Dota_2']),
                        folds=0,
                        fold_size=0,
                        grid_search_folds=0,
                        grid_search_fold_size=0,
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
                 fold_size=0,
                 grid_search_folds=3,
                 grid_search_fold_size=15,
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)]),
            # Setting `grid_search_folds` to 0 (i.e., not generating a
            # grid search set)
            dict(folds=3,
                 fold_size=15,
                 grid_search_folds=0,
                 grid_search_fold_size=0,
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)]),
            # Specifying a set of test games (equal to games) and test
            # bin ranges
            dict(folds=3,
                 fold_size=15,
                 grid_search_folds=0,
                 grid_search_fold_size=0,
                 test_games=games_set,
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 test_bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)]),
            # Specifying a set of test games (different from games),
            # test bin ranges, and `test_size`
            dict(folds=3,
                 fold_size=15,
                 grid_search_folds=0,
                 grid_search_fold_size=0,
                 test_size=30,
                 test_games=set(['Dota_2', 'Arma_3']),
                 bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)],
                 test_bin_ranges=[(0.0, 225.1), (225.2, 2026.2), (2026.3, 16435.0)]),
            # Specifying a set of test games (completely different from
            # games) and test bin ranges
            dict(folds=3,
                 fold_size=15,
                 grid_search_folds=0,
                 grid_search_fold_size=0,
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


class CVConfigTestCase(unittest.TestCase):
    """
    Test the `CVConfig` class.
    """

    def setUp(self):
        try:
            self.db = connect_to_db('localhost', 37017)
        except AutoReconnect as e:
            raise ConnectionFailure('Could not connect to MongoDB client. Make'
                                    'sure a tunnel is set up (or some other '
                                    'method is used) before running the '
                                    'tests.')
        self.prediction_label = 'total_game_hours'
        self.output_path = join(this_dir, 'test_output')
        if exists(self.output_path):
            rmtree(self.output_path)
        makedirs(self.output_path)

    def tearDown(self):
        rmtree(self.output_path)

    def test_CVConfig_invalid(self):
        """
        Test invalid parameter values for the `CVConfig` class.
        """

        learners = ['perc', 'pagr']
        non_nlp_features = parse_non_nlp_features_string('all', 'total_game_hours')
        param_grids = [DEFAULT_PARAM_GRIDS[LEARNER_DICT[learner]]
                       for learner in learners]
        valid_kwargs = dict(db=self.db,
                            games=set(['Dota_2']),
                            learners=learners,
                            param_grids=param_grids,
                            training_rounds=10,
                            training_samples_per_round=100,
                            grid_search_samples_per_fold=50,
                            non_nlp_features=non_nlp_features,
                            prediction_label=self.prediction_label,
                            output_path=self.output_path,
                            objective='pearson_r',
                            data_sampling='even',
                            grid_search_folds=5,
                            hashed_features=100000,
                            nlp_features=True,
                            bin_ranges=[(0.0, 225.1), (225.2, 2026.2),
                                        (2026.3, 16435.0)],
                            lognormal=False,
                            power_transform=None,
                            majority_baseline=True,
                            rescale=True,
                            n_jobs=1)
        
        # Combinations of parameters that should cause a `SchemaError`
        invalid_kwargs_list = [
            # Invalid `db` value
            dict(db='db',
                 **{p: v for p, v in valid_kwargs.items() if p != 'db'}),
            # Invalid games in `games` parameter value
            dict(games={'Dota'},
                 **{p: v for p, v in valid_kwargs.items() if p != 'games'}),
            # Invalid `learners` parameter value (unrecognized learner
            # abbreviations)
            dict(learners=['perceptron', 'passiveagressive'],
                 **{p: v for p, v in valid_kwargs.items() if p != 'learners'}),
            # Invalid `learners` parameter value (empty)
            dict(learners=[],
                 **{p: v for p, v in valid_kwargs.items() if p != 'learners'}),
            # Invalid parameter grids in `param_grids` parameter value
            dict(param_grids=[[dict(a=1, b=2), dict(c='g', d=True)]],
                 **{p: v for p, v in valid_kwargs.items() if p != 'param_grids'}),
            # `learners`/`param_grids` unequal in length
            dict(learners=['perc', 'pagr'],
                 param_grids=[DEFAULT_PARAM_GRIDS[LEARNER_DICT[learner]]
                              for learner in ['perc', 'pagr', 'mbkm']],
                 bin_ranges=None,
                 **{p: v for p, v in valid_kwargs.items() if not p
                    in ['learners', 'param_grids', 'bin_ranges']}),
            # Invalid `training_rounds` parameter value (must be int)
            dict(training_rounds=2.0,
                 **{p: v for p, v in valid_kwargs.items() if p != 'training_rounds'}),
            # Invalid `training_rounds` parameter value (must be greater
            # than 1)
            dict(training_rounds=1,
                 **{p: v for p, v in valid_kwargs.items() if p != 'training_rounds'}),
            # Invalid `training_samples_per_round` parameter value (must
            # be int)
            dict(training_samples_per_round=1.0,
                 **{p: v for p, v in valid_kwargs.items()
                    if p != 'training_samples_per_round'}),
            # Invalid `training_samples_per_round` parameter value (must
            # be greater than 0)
            dict(training_samples_per_round=0.0,
                 **{p: v for p, v in valid_kwargs.items()
                    if p != 'training_samples_per_round'}),
            # Invalid `grid_search_samples_per_fold` parameter value
            # (must be int)
            dict(grid_search_samples_per_fold=50.0,
                 **{p: v for p, v in valid_kwargs.items()
                    if p != 'grid_search_samples_per_fold'}),
            # Invalid `grid_search_samples_per_fold` parameter value
            # (must be greater than 1)
            dict(grid_search_samples_per_fold=0.0,
                 **{p: v for p, v in valid_kwargs.items()
                    if p != 'grid_search_samples_per_fold'}),
            # Invalid `non_nlp_features` parameter value (must be set of
            # valid features)
            dict(non_nlp_features={'total_game_hours_last_three_weeks'},
                 **{p: v for p, v in valid_kwargs.items()
                    if p != 'non_nlp_features'}),
            # Invalid `prediction_label` parameter value (must be in set
            # of valid features)
            dict(prediction_label='total_game_hours_last_three_weeks',
                 **{p: v for p, v in valid_kwargs.items()
                    if p != 'prediction_label'}),
            # Invalid `objective` parameter value (must be in set of
            # of valid objective function names)
            dict(objective='pearson',
                 **{p: v for p, v in valid_kwargs.items()
                    if p != 'objective'}),
            # Invalid `output_path` parameter value (must be string)
            dict(output_path=None,
                 **{p: v for p, v in valid_kwargs.items()
                    if p != 'output_path'}),
            # Invalid `output_path` parameter value (must exist)
            dict(output_path=join(self.output_path, 'does_not_exist'),
                 **{p: v for p, v in valid_kwargs.items()
                    if p != 'output_path'}),
            # Invalid `data_sampling` parameter value (must be in set of
            # of valid sampling methods)
            dict(data_sampling='equal',
                 **{p: v for p, v in valid_kwargs.items()
                    if p != 'data_sampling'}),
            # Invalid `grid_search_folds` parameter value (must be int)
            dict(grid_search_folds=0.0,
                 **{p: v for p, v in valid_kwargs.items()
                    if p != 'grid_search_folds'}),
            # Invalid `grid_search_folds` parameter value (must be
            # greater than 1)
            dict(grid_search_folds=1,
                 **{p: v for p, v in valid_kwargs.items()
                    if p != 'grid_search_folds'}),
            # Invalid `hashed_features` parameter value (must be
            # non-negative or None)
            dict(hashed_features=-1,
                 **{p: v for p, v in valid_kwargs.items()
                    if p != 'hashed_features'}),
            # Invalid `hashed_features` parameter value (must be
            # non-negative)
            dict(hashed_features=False,
                 **{p: v for p, v in valid_kwargs.items()
                    if p != 'hashed_features'}),
            # Invalid `nlp_features` parameter value (must be boolean or
            # None)
            dict(nlp_features=1,
                 **{p: v for p, v in valid_kwargs.items() if p != 'nlp_features'}),
            # Invalid `bin_ranges` parameter value (must be list of
            # tuples -- or None)
            dict(bin_ranges=[[0.2, 100.3], [100.5, 200.6]],
                 **{p: v for p, v in valid_kwargs.items() if p != 'bin_ranges'}),
            # Invalid `bin_ranges` parameter value (must be list of
            # tuples containing floats -- or None)
            dict(bin_ranges=[(0, 99), (100, 200)],
                 **{p: v for p, v in valid_kwargs.items() if p != 'bin_ranges'}),
            # Invalid `bin_ranges` parameter value (must be valid list
            # of bin ranges)
            dict(bin_ranges=[(0.9, 99.7), (99.9, 0.2)],
                 **{p: v for p, v in valid_kwargs.items() if p != 'bin_ranges'}),
            # Invalid `lognormal` parameter value (must be boolean or
            # None)
            dict(lognormal=0,
                 **{p: v for p, v in valid_kwargs.items() if p != 'lognormal'}),
            # Invalid `power_transform` parameter value (must be float
            # or None)
            dict(power_transform=False,
                 **{p: v for p, v in valid_kwargs.items()
                    if p != 'power_transform'}),
            # Invalid `power_transform` parameter value (must be float
            # or None)
            dict(power_transform=3,
                 **{p: v for p, v in valid_kwargs.items()
                    if p != 'power_transform'}),
            # Invalid `power_transform` parameter value (must be float
            # that is not equal to 0.0)
            dict(power_transform=0.0,
                 **{p: v for p, v in valid_kwargs.items()
                    if p != 'power_transform'}),
            # The `power_transform` and `lognormal` parameter values
            # were set as 2.0 and True, respectively, i.e., both were
            # set
            dict(power_transform=2.0,
                 lognormal=True,
                 **{p: v for p, v in valid_kwargs.items()
                    if not p in ['power_transform', 'lognormal']}),
            # Invalid `majority_baseline` parameter value (must be
            # boolean or None)
            dict(majority_baseline=0,
                 **{p: v for p, v in valid_kwargs.items()
                    if p != 'majority_baseline'}),
            # Invalid `rescale` parameter value (must be boolean or None)
            dict(rescale=0,
                 **{p: v for p, v in valid_kwargs.items() if p != 'rescale'}),
            # `learners` and `param_grids` of unequal size
            dict(learners=[learners[0]],
                 **{p: v for p, v in valid_kwargs.items() if p != 'learners'}),
            # `n_jobs` is not of type int
            dict(n_jobs=True,
                 **{p: v for p, v in valid_kwargs.items() if p != 'n_jobs'}),
            # `n_jobs` is less than 1
            dict(n_jobs=0,
                 **{p: v for p, v in valid_kwargs.items() if p != 'n_jobs'})
            ]
        for kwargs in invalid_kwargs_list:
            assert_raises(SchemaError, CVConfig, **kwargs)

    def test_CVConfig_valid(self):
        """
        Test valid parameter values for the `CVConfig` class.
        """

        learners = ['perc', 'pagr']
        non_nlp_features = parse_non_nlp_features_string('all', 'total_game_hours')
        param_grids = [DEFAULT_PARAM_GRIDS[LEARNER_DICT[learner]]
                       for learner in learners]
        valid_kwargs = dict(db=self.db,
                            games=set(['Dota_2']),
                            learners=learners,
                            param_grids=param_grids,
                            training_rounds=10,
                            training_samples_per_round=100,
                            grid_search_samples_per_fold=50,
                            non_nlp_features=non_nlp_features,
                            prediction_label=self.prediction_label,
                            output_path=self.output_path,
                            objective='pearson_r',
                            data_sampling='stratified',
                            grid_search_folds=5,
                            hashed_features=100000,
                            nlp_features=True,
                            bin_ranges=[(0.0, 225.1), (225.2, 2026.2),
                                        (2026.3, 16435.0)],
                            lognormal=False,
                            power_transform=None,
                            majority_baseline=True,
                            rescale=True,
                            n_jobs=4)
        default_params = set(['objective', 'data_sampling', 'grid_search_folds',
                              'hashed_features', 'nlp_features', 'bin_ranges',
                              'lognormal', 'power_transform', 'majority_baseline',
                              'rescale', 'n_jobs'])

        # Combinations of parameters
        valid_kwargs_list = [
            # Only specify non-default parameters
            dict(**{p: v for p, v in valid_kwargs.items() if not p
                    in default_params}),
            # Only specify non-default parameters + `objective`
            dict(**{p: v for p, v in valid_kwargs.items() if not p
                    in default_params.difference(['objective'])}),
            # Only specify non-default parameters + `data_sampling`
            dict(**{p: v for p, v in valid_kwargs.items() if not p
                    in default_params.difference(['data_sampling'])}),
            # Only specify non-default parameters + `grid_search_folds`
            dict(**{p: v for p, v in valid_kwargs.items() if not p
                    in default_params.difference(['grid_search_folds'])}),
            # Only specify non-default parameters + `hashed_features`
            dict(**{p: v for p, v in valid_kwargs.items() if not p
                    in default_params.difference(['hashed_features'])}),
            # Only specify non-default parameters + `nlp_features`
            dict(**{p: v for p, v in valid_kwargs.items() if not p
                    in default_params.difference(['nlp_features'])}),
            # Only specify non-default parameters + `bin_ranges`
            dict(**{p: v for p, v in valid_kwargs.items() if not p
                    in default_params.difference(['bin_ranges'])}),
            # Only specify non-default parameters + `lognormal`
            dict(**{p: v for p, v in valid_kwargs.items() if not p
                    in default_params.difference(['lognormal'])}),
            # Only specify non-default parameters + `power_transform`
            dict(**{p: v for p, v in valid_kwargs.items() if not p
                    in default_params.difference(['power_transform'])}),
            # Only specify non-default parameters + `majority_baseline`
            dict(**{p: v for p, v in valid_kwargs.items() if not p
                    in default_params.difference(['majority_baseline'])}),
            # Only specify non-default parameters + `rescale`
            dict(**{p: v for p, v in valid_kwargs.items() if not p
                    in default_params.difference(['rescale'])}),
            # Only specify non-default parameters + `n_jobs`
            dict(**{p: v for p, v in valid_kwargs.items() if not p
                    in default_params.difference(['n_jobs'])})
            ]
        for kwargs in valid_kwargs_list:

            # Make the configuration object
            cfg = CVConfig(**kwargs).validated

            # `db`
            assert_equal(cfg['db'], kwargs['db'])

            # `games`
            assert_equal(cfg['games'], kwargs['games'])

            # `learners`
            assert_equal(cfg['learners'], kwargs['learners'])
            assert (isinstance(cfg['learners'], list)
                    and all(learner in LEARNER_DICT_KEYS for learner in learners))
            assert_equal(len(cfg['learners']), len(cfg['param_grids']))

            # `param_grids`
            assert (isinstance(cfg['param_grids'], list)
                    and all(isinstance(pgrids_list, list) for pgrids_list
                            in cfg['param_grids'])
                    and all(all(isinstance(pgrid, dict) for pgrid in pgrids_list)
                            for pgrids_list in cfg['param_grids'])
                    and all(all(all(isinstance(param, str)
                                    for param in pgrid)
                                for pgrid in pgrids_list)
                            for pgrids_list in cfg['param_grids'])
                    and all(all(all(isinstance(pgrid[param], str)
                                    for param in pgrid)
                                for pgrid in pgrids_list)
                            for pgrids_list in cfg['param_grids'])
                    and len(cfg['param_grids']) > 0)

            # `training_rounds`, `training_samples_per_round`,
            # `grid_search_samples_per_fold`, and `grid_search_folds`
            assert cfg['training_rounds'] > 1
            assert cfg['training_samples_per_round'] > 0
            assert cfg['grid_search_samples_per_fold'] > 1
            if 'grid_search_folds' in kwargs:
                assert 'grid_search_folds' in cfg
                assert cfg['grid_search_folds'] > 1
                assert_equal(cfg['grid_search_folds'], kwargs['grid_search_folds'])
            else:
                assert_equal(cfg['grid_search_folds'], 5)

            # `nlp_features`, `non_nlp_features`, and `prediction_label`
            assert (isinstance(cfg['non_nlp_features'], set)
                    and cfg['non_nlp_features'].issubset(LABELS))
            assert (isinstance(cfg['prediction_label'], str)
                    and cfg['prediction_label'] in LABELS
                    and not cfg['prediction_label'] in cfg['non_nlp_features'])
            if 'nlp_features' in kwargs:
                assert 'nlp_features' in cfg
                assert isinstance(cfg['nlp_features'], bool)
                assert_equal(cfg['nlp_features'], kwargs['nlp_features'])
            else:
                assert_equal(cfg['nlp_features'], True)

            # `objective`
            if 'objective' in kwargs:
                assert 'objective' in cfg
                assert cfg['objective'] in OBJ_FUNC_ABBRS_DICT
                assert_equal(cfg['objective'], kwargs['objective'])
            else:
                assert_equal(cfg['objective'], None)

            # `data_sampling`
            if 'data_sampling' in kwargs:
                assert 'data_sampling' in cfg
                assert cfg['data_sampling'] in ExperimentalData.sampling_options
                assert_equal(cfg['data_sampling'], kwargs['data_sampling'])
            else:
                assert_equal(cfg['data_sampling'], 'even')

            # `hashed_features`
            if 'hashed_features' in kwargs:
                assert 'hashed_features' in cfg
                if cfg['hashed_features'] is not None:
                    assert cfg['hashed_features'] > -1
                assert_equal(cfg['hashed_features'], kwargs['hashed_features'])
            else:
                assert_equal(cfg['hashed_features'], None)

            # `bin_ranges`
            if 'bin_ranges' in kwargs:
                assert 'bin_ranges' in cfg
                assert (isinstance(cfg['bin_ranges'], list)
                        and all((isinstance(bin_, tuple)
                                 and all(isinstance(val, float) for val in bin_))
                                for bin_ in cfg['bin_ranges']))
                assert_equal(cfg['bin_ranges'], kwargs['bin_ranges'])
                validate_bin_ranges(cfg['bin_ranges'])
            else:
                assert_equal(cfg['bin_ranges'], None)

            # `lognormal`
            if 'lognormal' in kwargs:
                assert 'lognormal' in cfg
                assert isinstance(cfg['lognormal'], bool)
                assert_equal(cfg['lognormal'], kwargs['lognormal'])
            else:
                assert_equal(cfg['lognormal'], False)

            # `power_transform`
            if 'power_transform' in kwargs:
                assert 'power_transform' in cfg
                assert (cfg['power_transform'] is None
                        or isinstance(cfg['power_transform'], bool))
                assert_equal(cfg['power_transform'], kwargs['power_transform'])
            else:
                assert_equal(cfg['power_transform'], None)

            # `majority_baseline`
            if 'majority_baseline' in kwargs:
                assert 'majority_baseline' in cfg
                assert isinstance(cfg['majority_baseline'], bool)
                assert_equal(cfg['majority_baseline'], kwargs['majority_baseline'])
            else:
                assert_equal(cfg['majority_baseline'], True)

            # `rescale`
            if 'rescale' in kwargs:
                assert 'rescale' in cfg
                assert isinstance(cfg['rescale'], bool)
                assert_equal(cfg['rescale'], kwargs['rescale'])
            else:
                assert_equal(cfg['rescale'], True)

            # `n_jobs`
            if 'n_jobs' in kwargs:
                assert 'n_jobs' in cfg
                assert isinstance(cfg['n_jobs'], int)
                assert_equal(cfg['n_jobs'], kwargs['n_jobs'])
            else:
                assert_equal(cfg['n_jobs'], 1)
