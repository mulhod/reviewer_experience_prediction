"""
Test utility functions in `src`, i.e., for parsing command-line
arguments, etc.
"""
from os.path import join
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np
from typing import (List,
                    Tuple,
                    Optional)
from nose2.compat import unittest
from nose.tools import assert_equal
from sklearn.cluster import MiniBatchKMeans
from sklearn.naive_bayes import (BernoulliNB,
                                 MultinomialNB)
from sklearn.linear_model import (Perceptron,
                                  PassiveAggressiveRegressor)

from src import (Learner,
                 ParamGrid,
                 LEARNER_DICT,
                 get_game_files,
                 LEARNER_DICT_KEYS,
                 parse_games_string,
                 DEFAULT_PARAM_GRIDS,
                 parse_learners_string,
                 find_default_param_grid,
                 parse_non_nlp_features_string)


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


class FindDefaultParamGridsTestCase(unittest.TestCase):
    """
    Tests for the `find_default_param_grid` function.
    """

    def check_find_default_param_grid_defaults(self,
                                               learners: List[Tuple[str, Learner]],
                                               param_grids: ParamGrid) -> None:
        """
        Assist in testing the `find_default_param_grid` function.
        
        :param learners: list of tuples consisting of a learner name
                         abbreviation and a corresponding learner class
        :type learners: list
        :param param_grids: dictionary mapping learner classes to
                            parameter grids
        :type param_grids: dict

        :returns: None
        :rtype: None
        """

        for learner_abbrev, learner in learners:
            assert_equal(find_default_param_grid(learner_abbrev, param_grids),
                         param_grids[learner])

    def test_find_default_param_grid(self):
        """
        Test the `find_default_param_grid` function.
        """

        custom_param_grids = \
            {MiniBatchKMeans: {'n_clusters': [4, 5, 6, 7, 9],
                               'init' : ['k-means++', 'random']},
             BernoulliNB: {'alpha': [0.1, 0.5, 1.0]},
             MultinomialNB: {'alpha': [0.5, 0.75, 1.0]},
             Perceptron: {'penalty': ['l2', 'l1', 'elasticnet'],
                          'alpha': [0.0001, 0.001, 0.01],
                          'n_iter': [5]},
             PassiveAggressiveRegressor: {'C': [0.01, 0.1, 1.0],
                                          'n_iter': [10],
                                          'loss': ['epsilon_insensitive']}}

        learners = [MiniBatchKMeans, BernoulliNB, MultinomialNB, Perceptron,
                    PassiveAggressiveRegressor]
        learner_abbrevs = ['mbkm', 'bnb', 'mnb', 'perc', 'pagr']
        for param_grids in [DEFAULT_PARAM_GRIDS, custom_param_grids]:
            yield (self.check_find_default_param_grid_defaults,
                   list(zip(learner_abbrevs, learners)),
                   param_grids)


class ParseLearnersStringTestCase(unittest.TestCase):
    """
    Tests for the `parse_learners_string` function.
    """

    learners = list(LEARNER_DICT_KEYS)

    def test_parse_learners_string_valid(self):
        """
        Use valid parameter values to tests `parse_learners_string`.
        """

        for i in range(1, len(self.learners) + 1):
            assert_equal(sorted(list(parse_learners_string(','.join(self.learners[:i])))),
                         sorted(self.learners[:i]))

    def test_parse_learners_string_invalid(self):
        """
        Use invalid parameter values to test `parse_learners_string`.
        """

        fake_and_real_names = (self.learners
                               + ['perceptron', 'multinomialnb', 'MNB', ''])
        np.random.shuffle(fake_and_real_names)
        for i in range(len(fake_and_real_names)):
            if set(self.learners).issuperset(fake_and_real_names[:i]): continue
            with self.assertRaises(ValueError):
                parse_learners_string(','.join(fake_and_real_names[:i]))
