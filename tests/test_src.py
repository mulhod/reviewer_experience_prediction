"""
Test utility functions in `src`, i.e., for parsing command-line
arguments, etc.
"""
from os.path import join
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np
from itertools import chain
from typing import (List,
                    Tuple,
                    Optional)
from nose2.compat import unittest
from nose.tools import (assert_equal,
                        assert_raises)
from sklearn.cluster import MiniBatchKMeans
from sklearn.naive_bayes import (BernoulliNB,
                                 MultinomialNB)
from sklearn.linear_model import (Perceptron,
                                  PassiveAggressiveRegressor)

from src import (LABELS,
                 Learner,
                 ParamGrid,
                 VALID_GAMES,
                 TIME_LABELS,
                 LEARNER_DICT,
                 get_game_files,
                 FRIENDS_LABELS,
                 HELPFUL_LABELS,
                 LEARNER_DICT_KEYS,
                 parse_games_string,
                 DEFAULT_PARAM_GRIDS,
                 ACHIEVEMENTS_LABELS,
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
        Use valid parameter values to test `parse_learners_string`.
        """

        # Test some combinations of valid learner-types (not all
        # combinations, though)
        for i in range(1, len(self.learners) + 1):
            assert_equal(
                sorted(parse_learners_string(','.join(self.learners[:i]))),
                sorted(self.learners[:i]))

    def test_parse_learners_string_invalid(self):
        """
        Use invalid parameter values to test `parse_learners_string`.
        """

        # Test some combinations of invalid (and possibly valid)
        # learner-types
        fake_and_real_names = (self.learners
                               + ['perceptron', 'multinomialnb', 'MNB', ''])
        np.random.shuffle(fake_and_real_names)
        for i in range(len(fake_and_real_names)):
            if set(self.learners).issuperset(fake_and_real_names[:i]): continue
            with self.assertRaises(ValueError):
                parse_learners_string(','.join(fake_and_real_names[:i]))


class ParseNonNLPFeaturesStringTestCase(unittest.TestCase):
    """
    Tests for the `parse_non_nlp_features_string` function.
    """

    labels = set(LABELS)
    label_groups = dict(TIME_LABELS=set(TIME_LABELS),
                        FRIENDS_LABELS=set(FRIENDS_LABELS),
                        HELPFUL_LABELS=set(HELPFUL_LABELS),
                        ACHIEVEMENTS_LABELS=set(ACHIEVEMENTS_LABELS),
                        OTHER=set([label for label in LABELS
                                   if not label in chain(TIME_LABELS,
                                                         FRIENDS_LABELS,
                                                         HELPFUL_LABELS,
                                                         ACHIEVEMENTS_LABELS)]))

    def test_parse_non_nlp_features_string_valid(self):
        """
        Use valid parameter values to test `parse_non_nlp_features_string`.
        """

        # Test some valid combinations (not all) of non-NLP features
        for label_group in self.label_groups:
            
            valid_prediction_labels = self.label_groups[label_group]

            # Pick one random label to use as the prediction label from
            # each group of labels
            group_labels = list(valid_prediction_labels)
            np.random.shuffle(group_labels)
            prediction_label = group_labels[0]

            if label_group != 'OTHER':
                valid_labels = list(self.labels.difference(valid_prediction_labels))
            else:
                valid_labels = [label for label in self.labels
                                if not label == prediction_label]

            for i in range(1, len(valid_labels) + 1):
                assert_equal(
                    sorted(parse_non_nlp_features_string(','.join(valid_labels[:i]),
                                                         prediction_label)),
                    sorted(valid_labels[:i]))

    def test_parse_non_nlp_features_string_unrecognized(self):
        """
        Use invalid parameter values to test `parse_non_nlp_features_string`.
        """

        # Use one of the time-related labels as the prediction label
        # and exclude all time-related labels from the input string
        label_group = 'TIME_LABELS'
        prediction_label = list(self.label_groups[label_group])[0]
        fake_and_real_features = \
            self.labels.difference(self.label_groups[label_group])

        # Add fake features to the set of input features and shuffle it
        fake_and_real_features.update({'hours', 'achievements', 'friends',
                                       'groups'})
        fake_and_real_features = list(fake_and_real_features)
        np.random.shuffle(fake_and_real_features)

        # Iterate through features and discard any set that doesn't
        # contain at least one unrecognized feature
        for i in range(len(fake_and_real_features)):
            if self.labels.issuperset(fake_and_real_features[:i]): continue
            with self.assertRaises(ValueError):
                parse_non_nlp_features_string(','.join(fake_and_real_features[:i]),
                                              prediction_label)

    def test_parse_non_nlp_features_string_group_conflict(self):
        """
        Use parameter values that represent a conflict to test whether
        or not `parse_non_nlp_features_string` will catch it.
        """

        for label_group in self.label_groups:

            # Skip 'OTHER' label group
            if label_group == 'OTHER': continue

            # Use one label from the label group as the prediction label
            group_labels = list(self.label_groups[label_group])
            prediction_label = group_labels[0]

            # Get a small set of labels from other groups
            other_group_labels = list(set(chain(*self.label_groups.values()))
                                      .difference(group_labels))
            np.random.shuffle(other_group_labels)
            other_group_labels = other_group_labels[:5]

            # Iterate through each group label that represents a
            # conflict (including the prediction label itself)
            for label in group_labels:
                labels = [label_ for label_ in group_labels if label_ != label]
                for i in range(len(labels)):
                    labels_ = labels[:i] + [label]
                    with self.assertRaises(ValueError):
                        parse_non_nlp_features_string(','.join(labels_),
                                                      prediction_label)
                    labels_ = labels_ + other_group_labels
                    np.random.shuffle(labels_)
                    with self.assertRaises(ValueError):
                        parse_non_nlp_features_string(','.join(labels_),
                                                      prediction_label)

    def test_parse_non_nlp_features_string_all(self):
        """
        Test `parse_non_nlp_features_string` when a value of "all" is
        used instead of a comma-separated list of labels (which should
        automatically remove any labels that conflict with the
        prediction label).
        """

        # Use one label from the label group as the prediction label
        label_group_name = 'TIME_LABELS'
        group_labels = self.label_groups[label_group_name]
        prediction_label = list(group_labels)[0]
        expected_labels = self.labels.difference(group_labels)

        assert_equal(sorted(parse_non_nlp_features_string('all',
                                                          prediction_label)),
                     sorted(expected_labels))

    def test_parse_non_nlp_features_string_none(self):
        """
        Test `parse_non_nlp_features_string` when a value of "none" is
        used instead of a comma-separated list of labels (return a set
        consisting of no labels).
        """

        # Use one label from the label group as the prediction label
        label_group_name = list(self.label_groups)[0]
        group_labels = self.label_groups[label_group_name]
        prediction_label = list(group_labels)[0]
        expected_labels = set()

        assert_equal(parse_non_nlp_features_string('none', prediction_label),
                     set())


class ParseGamesStringTestCase(unittest.TestCase):
    """
    Tests for the `parse_games_string` function.
    """

    games = list(VALID_GAMES)

    def test_parse_games_string_all(self):
        """
        Test `parse_games_string` using "all" as input.
        """

        assert_equal(sorted(parse_games_string("all")),
                     sorted(self.games))

    def test_parse_games_string_valid(self):
        """
        Test `parse_games_string` using inputs consisting of valid
        games.
        """

        # Iterate through the list of valid games and try successively
        # larger inputs
        for i in range(1, len(self.games) + 1):
            assert_equal(sorted(parse_games_string(','.join(self.games[:i]))),
                         sorted(self.games[:i]))

    def test_parse_games_string_unrecognized(self):
        """
        Test `parse_games_string` using input consisting of unrecognized
        games (and possibly recognized games).
        """

        invalid_games = ["game1", "game2", "game3", "game4"]
        games = self.games + invalid_games
        np.random.shuffle(games)

        # Iterate through the list of valid/unrecognized games and
        # ensure that exceptions are raised in the cases where
        # unrecognized games are included
        for i in range(1, len(games) + 1):
            for j in range(0, len(games), i):
                if len(games[j:j + i]) != i: continue
                if VALID_GAMES.issuperset(games[j:j + i]):
                    assert_equal(
                        sorted(parse_games_string(','.join(games[j:j + i]))),
                        sorted(games[j:j + i]))
                else:
                    with self.assertRaises(ValueError):
                        parse_games_string(','.join(games[j:j + i]))
