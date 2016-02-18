"""
Test utility functions in `src`, i.e., for parsing command-line
arguments, etc.
"""
from typing import (List,
                    Tuple)
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
                 parse_games_string,
                 DEFAULT_PARAM_GRIDS,
                 parse_learners_string,
                 find_default_param_grid,
                 parse_non_nlp_features_string)

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
