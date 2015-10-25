#!/usr/env python3.4
'''
:author: Matt Mulholland (mulhodm@gmail.com)
:date: 10/14/2015

Command-line utility for the IncrementalLearning class, which enables
one to run experiments on subsets of the data with a number of
different machine learning algorithms and parameter customizations,
etc.
'''
import logging
from sys import exit
from copy import copy
from operator import or_
from os import (listdir,
                makedirs)
from os.path import (join,
                     exists,
                     dirname,
                     realpath)
from warnings import filterwarnings

import numpy as np
import pandas as pd
from bson import BSON
from pymongo import collection
from skll.metrics import kappa
from scipy.stats import (mode,
                         pearsonr)
from sklearn.cluster import MiniBatchKMeans
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import (precision_score,
                             f1_score,
                             accuracy_score,
                             confusion_matrix)
from sklearn.naive_bayes import (BernoulliNB,
                                 MultinomialNB)
from argparse import (ArgumentParser,
                      ArgumentDefaultsHelpFormatter)
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import (Perceptron,
                                  PassiveAggressiveRegressor)

from data import APPID_DICT
from util.mongodb import connect_to_db

# Set up logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -'
                              ' %(message)s')
sh.setFormatter(formatter)
sh.setLevel(logging.INFO)
logger.addHandler(sh)

project_dir = dirname(dirname(realpath(__file__)))

# Filter out warnings since there will be a lot of
# "UndefinedMetricWarning" warnings when running IncrementalLearning
filterwarnings("ignore")

seed = 123456789

# Define default parameter grids
_DEFAULT_PARAM_GRIDS = \
    {MiniBatchKMeans: {'n_clusters': [4, 6, 8, 12],
                       'init' : ['k-means++', 'random'],
                       'random_state': [seed]},
     BernoulliNB: {'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]},
     MultinomialNB: {'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]},
     Perceptron: {'penalty': [None, 'l2', 'l1', 'elasticnet'],
                  'alpha': [0.0001, 0.001, 0.01, 0.1],
                  'n_iter': [5, 10],
                  'random_state': [seed]},
     PassiveAggressiveRegressor:
         {'C': [0.01, 0.1, 1.0, 10.0, 100.0],
          'n_iter': [5, 10],
          'random_state': [seed],
          'loss': ['epsilon_insensitive',
                   'squared_epsilon_insensitive']}}

learner_names_dict = {MiniBatchKMeans: 'MiniBatchKMeans',
                      BernoulliNB: 'BernoulliNB',
                      MultinomialNB: 'MultinomialNB',
                      Perceptron: 'Perceptron',
                      PassiveAggressiveRegressor:
                          'PassiveAggressiveRegressor'}
learner_abbrs_dict = {'mbkm': 'MiniBatchKMeans',
                      'bnb': 'BernoulliNB',
                      'mnb': 'MultinomialNB',
                      'perc': 'Perceptron',
                      'pagr': 'PassiveAggressiveRegressor'}
learner_dict_keys = frozenset(learner_abbrs_dict.keys())
learner_dict = {k: eval(learner_abbrs_dict[k]) for k in learner_dict_keys}
learner_abbrs_string = ', '.join(['"{}" ({})'.format(abbr,
                                                     learner)
                                  for abbr, learner
                                  in learner_abbrs_dict.items()])

obj_funcs = frozenset({'r', 'significance', 'precision_macro',
                       'precision_weighted', 'f1_macro', 'f1_weighted',
                       'accuracy', 'uwk', 'uwk_off_by_one', 'qwk',
                       'qwk_off_by_one', 'lwk', 'lwk_off_by_one'})

labels = frozenset({'num_guides', 'num_games_owned', 'num_friends',
                    'num_voted_helpfulness', 'num_groups',
                    'num_workshop_items', 'num_reviews', 'num_found_funny',
                    'friend_player_level', 'num_badges', 'num_found_helpful',
                    'num_screenshots', 'num_found_unhelpful',
                    'found_helpful_percentage', 'num_comments',
                    'total_game_hours', 'total_game_hours_bin',
                    'total_game_hours_last_two_weeks',
                    'num_achievements_percentage',
                    'num_achievements_possible'})
time_labels = frozenset({'total_game_hours', 'total_game_hours_bin',
                         'total_game_hours_last_two_weeks'})


def _find_default_param_grid(learner: str) -> dict:
    """
    Finds the default parameter grid for the specified learner.

    :param learner: abbreviated string representation of a learner
    :type learner: str
    :returns: dict
    """

    for key_cls, grid in _DEFAULT_PARAM_GRIDS.items():
        if issubclass(learner_dict[learner],
                      key_cls):
            return grid
    raise Exception('Unrecognized learner abbreviation: {}'.format(learner))


class IncrementalLearning:
    '''
    Class for conducting incremental learning experiments with a
    parameter grid and a learner.
    '''

    # Constants
    __game__ = 'game'
    __partition__ = 'partition'
    __training__ = 'training'
    __test__ = 'test'
    __nlp_feats__ = 'nlp_features'
    __achieve_prog__ = 'achievement_progress'
    __nan__ = float("NaN")
    __x__ = 'x'
    __y__ = 'y'
    __id_string__ = 'id_string'
    __id__ = 'id'
    __macro__ = 'macro'
    __weighted__ = 'weighted'
    __linear__ = 'linear'
    __quadratic__ = 'quadratic'
    __learning_round__ = 'learning_round'
    __prediction_label__ = 'prediction_label'
    __test_labels_and_preds__ = 'test_set_labels/test_set_predictions'
    __learner__ = 'learner'
    __learners_requiring_classes__ = frozenset({'BernoulliNB',
                                                'MultinomialNB',
                                                'Perceptron'})
    __params__ = 'params'
    __training_samples__ = 'training_samples'
    __r__ = 'pearson_r'
    __sig__ = 'significance'
    __prec_macro__ = 'precision_macro'
    __prec_weighted__ = 'precision_weighted'
    __f1_macro__ = 'f1_macro'
    __f1_weighted__ = 'f1_weighted'
    __acc__ = 'accuracy'
    __cnfmat__ = 'confusion_matrix'
    __printable_cnfmat__ = 'printable_confusion_matrix'
    __uwk__ = 'uwk'
    __uwk_off_by_one__ = 'uwk_off_by_one'
    __qwk__ = 'qwk'
    __qwk_off_by_one__ = 'qwk_off_by_one'
    __lwk__ = 'lwk'
    __lwk_off_by_one__ = 'lwk_off_by_one'
    __possible_non_nlp_features__ = copy(labels)
    __tab_join__ = '\t'.join
    __cnfmat_row__ = '{}{}\n'.format
    __cnfmat_header__ = ('confusion_matrix (rounded predictions) '
                         '(row=human, col=machine, labels={}):\n')
    __majority_label__ = 'majority_label'
    __majority_baseline_model__ = 'majority_baseline_model'

    def __init__(self, db: collection, game: str, learners, param_grids: dict,
                 round_size: int, non_nlp_features: list,
                 prediction_label: str, objective: str, test_limit=0,
                 rounds=0, majority_baseline=False):
        '''
        Initialize class.

        :param db: MongoDB database collection object
        :type db: collection
        :param game: game
        :type game: str
        :param learners: algorithms to use for learning
        :type learners: list of learners
        :param param_grids: list of dictionaries of parameters mapped
                            to lists of values (must be aligned with
                            list of learners)
        :type param_grids: dict
        :param round_size: number of training documents to extract in
                           each round
        :type round_size: int
        :param non_nlp_features: list of non-NLP features to add into
                                 the feature dictionaries 
        :type non_nlp_features: list of str
        :param prediction_label: feature to predict
        :type prediction_label: str
        :param objective: objective function to use in ranking the runs
        :type objective: str
        :param test_limit: limit for the number of test samples
                           (defaults to 0 for no limit)
        :type test_limit: int
        :param rounds: number of rounds of learning (0 for as many as
                       possible)
        :type rounds: int
        :param majority_baseline: evaluate a majority baseline model
        :type majority_baseline: boolean
        :returns: list of dict
        '''

        # Make sure parameters make some sense
        if round_size < 1:
            raise Exception('The round_size parameter should have a positive'
                            ' value.')
        if prediction_label in non_nlp_features:
            raise Exception('The prediction_label parameter ({}) cannot also '
                            'be in the list of non-NLP features to use in the'
                            ' model:\n\n{}\n.'.format(prediction_label,
                                                      non_nlp_features))
        if not prediction_label in self.__possible_non_nlp_features__:
            raise Exception('The prediction label must be in the set of '
                            'features that can be extracted/used, i.e.: {}.'
                            .format(self.__possible_non_nlp_features__))

        # MongoDB database
        self.db = db

        # Game
        self.game = game

        # Objective function
        self.obj_func = objective
        if not self.obj_func in obj_funcs:
            raise Exception('Unrecognized objective function used: {}. These '
                            'are the available objective functions: {}.'
                            .format(objective,
                                    ', '.join(obj_funcs)))

        # Learner-related variables
        self.vec = None
        self.param_grids = [list(ParameterGrid(param_grid)) for param_grid
                            in param_grids]
        self.learner_names = [learner_names_dict[learner]
                              for learner in learners]
        self.learner_lists = [[learner(**kwparams) for kwparams in param_grid]
                              for learner, param_grid
                              in zip(learners,
                                     self.param_grids)]
        self.learner_param_grid_stats = []
        for learner_list in self.learner_lists:
            self.learner_param_grid_stats.append([[] for _ in learner_list])

        # Information about what features to use for what purposes
        if all(feat in self.__possible_non_nlp_features__
               for feat in non_nlp_features):
            self.non_nlp_features = non_nlp_features
        self.prediction_label = prediction_label

        # Incremental learning-related variables
        self.round_size = round_size
        self.rounds = rounds
        self.round = 1
        self.NO_MORE_TRAINING_DATA = False

        # Test data-related variables
        self.training_cursor = None
        self.test_cursor = None
        self.test_limit = test_limit
        self.make_cursors()
        self.test_data = self.get_test_data()
        self.test_ids = [_data[self.__id__] for _data in self.test_data]
        self.test_feature_dicts = [_data[self.__x__] for _data
                                   in self.test_data]
        self.y_test = np.array([_data[self.__y__] for _data
                                in self.test_data])
        self.classes = np.unique(self.y_test)

        # Useful constants for use in make_printable_confusion_matrix
        self.cnfmat_desc = \
            self.__cnfmat_row__(self.__cnfmat_header__.format(self.classes),
                                self.__tab_join__([''] + [str(x) for x
                                                          in self.classes]))

        # Do incremental learning experiments
        logger.info('Incremental learning experiments initialized...')
        self.do_learning_rounds()
        self.learner_param_grid_stats = [[pd.DataFrame(param_grid)
                                          for param_grid in learner]
                                         for learner
                                         in self.learner_param_grid_stats]

        # Generate statistics for the majority baseline model
        if majority_baseline:
            self.majority_label = None
            self.majority_baseline_stats = None
            self.evaluate_majority_baseline_model()

    def make_cursors(self):
        '''
        Make cursor objects for the training/test sets.
        '''

        # Make training data cursor
        batch_size = 50
        self.training_cursor = \
            self.db.find({self.__game__: self.game,
                          self.__partition__: self.__training__},
                         timeout=False)
        self.training_cursor.batch_size = batch_size

        # Make test data cursor
        self.test_cursor = \
            self.db.find({self.__game__: self.game,
                          self.__partition__: self.__test__},
                         timeout=False)
        if self.test_limit:
            self.test_cursor = self.test_cursor.limit(self.test_limit)
        self.test_cursor.batch_size = batch_size

    def get_all_features(self, review_doc: dict) -> dict:
        '''
        Get all the features in a review document and put them together
        in a dictionary.

        :param review_doc: review document from Mongo database
        :type review_doc: dict
        :returns: dict
        '''

        _get = review_doc.get

        # Add in the NLP features
        features = {feat: val for feat, val
                    in BSON.decode(_get(self.__nlp_feats__)).items()
                    if val and val != self.__nan__}

        # Add in the non-NLP features (except for those that may be in
        # the 'achievement_progress' sub-dictionary of the review
        # dictionary
        features.update({feat: val
                         for feat, val in review_doc.items()
                         if feat in self.__possible_non_nlp_features__
                            and val
                            and val != self.__nan__})

        # Add in the features that may be in the 'achievement_progress'
        # sub-dictionary of the review dictionary
        features.update({feat: val for feat, val
                         in _get(self.__achieve_prog__,
                                 dict()).items()
                         if feat in self.__possible_non_nlp_features__
                            and val
                            and val != self.__nan__})

        # Add in the 'id_string' value just to make it easier to
        # process the results of this function
        features.update({self.__id_string__: _get(self.__id_string__)})
        return features

    def get_train_data_iteration(self) -> list:
        '''
        Get a list of training data dictionaries to use in model
        training.

        :returns: list of dict
        '''

        data = []
        i = 0
        while i < self.round_size:
            # Get a review document from the Mongo database
            try:
                review_doc = next(self.training_cursor)
            except StopIteration:
                self.NO_MORE_TRAINING_DATA = True
                break

            # Get dictionary containing all features needed + the ID
            # and the prediction label
            feature_dict = self.get_all_features(review_doc)
            _get = feature_dict.get

            # Get prediction label feature and remove it from feature
            # dictionary, skipping the document if it's not found or if
            # its value is None
            if _get(self.prediction_label):
                y_value = _get(self.prediction_label,
                               None)
                if y_value == None:
                    i += 1
                    continue
                del feature_dict[self.prediction_label]
            else:
                i += 1
                continue

            # Get ID and remove from feature dictionary
            id_string = _get(self.__id_string__)
            del feature_dict[self.__id_string__]

            # Put features, prediction label, and ID in a new
            # dictionary and append to list of data samples and then
            # increment the review counter
            data.append(dict(y=y_value,
                             id=id_string,
                             x=feature_dict))
            i += 1
        return data

    def get_test_data(self) -> list:
        '''
        Get a list of test data dictionaries to use in model
        evaluation.

        :returns: list of dict
        '''

        data = []
        for review_doc in self.test_cursor:
            # Get dictionary containing all features needed + the ID
            # and the prediction label
            feature_dict = self.get_all_features(review_doc)
            _get = feature_dict.get

            # Get prediction label feature and remove it from feature
            # dictionary, skipping the document if it's not found or if
            # its value is None
            if _get(self.prediction_label):
                y_value = _get(self.prediction_label,
                               None)
                if y_value == None:
                    i += 1
                    continue
                del feature_dict[self.prediction_label]
            else:
                i += 1
                continue

            # Get ID and remove from feature dictionary
            id_string = _get(self.__id_string__)
            del feature_dict[self.__id_string__]

            # Put features, prediction label, and ID in a new
            # dictionary and append to list of data samples and then
            # increment the review counter
            data.append(dict(y=y_value,
                             id=id_string,
                             x=feature_dict))
        return data

    def get_majority_baseline(self) -> np.array:
        '''
        Generate a majority baseline array of prediction labels.

        :returns: np.array
        '''

        self.majority_label = mode(self.y_test).mode[0]
        return np.array([self.majority_label]*len(self.y_test))

    def evaluate_majority_baseline_model(self) -> None:
        '''
        Evaluate the majority baseline model predictions.
        '''

        stats_dict = self.get_stats(self.get_majority_baseline())
        stats_dict.update({self.__prediction_label__: self.prediction_label,
                           self.__majority_label__: self.majority_label,
                           self.__learner__:
                               self.__majority_baseline_model__})
        self.majority_baseline_stats = pd.DataFrame([pd.Series(stats_dict)])

    def make_printable_confusion_matrix(self, y_preds) -> tuple:
        '''
        Produce a printable confusion matrix to use in the evaluation
        report.

        :param y_preds: array-like of predicted labels
        :type y_preds: array-like
        :returns: (str, np.ndarray)
        '''

        cnfmat = confusion_matrix(self.y_test,
                                  np.round(y_preds),
                                  labels=self.classes).tolist()
        res = str(self.cnfmat_desc)
        for row, label in zip(cnfmat,
                              self.classes):
            row = self.__tab_join__([str(x) for x in [label] + row])
            res = self.__cnfmat_row__(res, row)
        return res, cnfmat

    def fit_preds_in_scale(self, y_preds):
        '''
        Force values at either end of the scale to fit within the scale
        by adding to or truncating the values.

        :param y_preds: array-like of predicted labels
        :type y_preds: array-like
        :returns: array-like
        '''

        # Get low/high ends of the scale
        scale = sorted(self.classes)
        low = scale[0]
        high = scale[-1]

        i = 0
        while i < len(y_preds):
            if y_preds[i] < low:
                y_preds[i] = low
            elif y_preds[i] > high:
                y_preds[i] = high
            i += 1
        return y_preds

    def get_stats(self, y_preds) -> dict:
        """
        Get some statistics about the model's performance on the test
        set.

        :param y_preds: array-like of predicted labels
        :type y_preds: array-like
        :returns: dict
        """

        # Get Pearson r and significance
        r, sig = pearsonr(self.y_test,
                          y_preds)

        # Get confusion matrix (both the np.ndarray and the printable
        # one)
        printable_cnfmat, cnfmat = \
            self.make_printable_confusion_matrix(y_preds)

        return {self.__r__: r,
                self.__sig__: sig,
                self.__prec_macro__: precision_score(self.y_test,
                                                     y_preds,
                                                     labels=self.classes,
                                                     average=self.__macro__),
                self.__prec_weighted__:
                    precision_score(self.y_test,
                                    y_preds,
                                    labels=self.classes,
                                    average=self.__weighted__),
                self.__f1_macro__: f1_score(self.y_test,
                                            y_preds,
                                            labels=self.classes,
                                            average=self.__macro__),
                self.__f1_weighted__: f1_score(self.y_test,
                                               y_preds,
                                               labels=self.classes,
                                               average=self.__weighted__),
                self.__acc__: accuracy_score(self.y_test,
                                             y_preds,
                                             normalize=True),
                self.__cnfmat__: cnfmat,
                self.__printable_cnfmat__: printable_cnfmat,
                self.__uwk__: kappa(self.y_test,
                                    y_preds),
                self.__uwk_off_by_one__: kappa(self.y_test,
                                               y_preds,
                                               allow_off_by_one=True),
                self.__qwk__: kappa(self.y_test,
                                    y_preds,
                                    weights=self.__quadratic__),
                self.__qwk_off_by_one__: kappa(self.y_test,
                                               y_preds,
                                               weights=self.__quadratic__,
                                               allow_off_by_one=True),
                self.__lwk__: kappa(self.y_test,
                                    y_preds,
                                    weights=self.__linear__),
                self.__lwk_off_by_one__: kappa(self.y_test,
                                               y_preds,
                                               weights=self.__linear,
                                               allow_off_by_one=True)}

    def learning_round(self) -> None:
        '''
        Do learning rounds.
        '''

        # Get some training data
        train_data = self.get_train_data_iteration()
        samples = len(train_data)

        # Skip round if there are no more training samples to learn
        # from or if the number remaining is less than half the size of
        # the intended number of samples to be used in each round
        if (not samples
            or samples < self.round_size/2):
            return

        logger.info('Round {}...'.format(self.round))
        train_ids = np.array([_data[self.__id__] for _data in train_data])
        y_train = np.array([_data[self.__y__] for _data in train_data])
        train_feature_dicts = [_data[self.__x__] for _data in train_data]

        # Set _vec if not already set and fit it it the training
        # features, which will only need to be done the first time
        if self.vec == None:
            self.vec = DictVectorizer(sparse=True)
            X_train = self.vec.fit_transform(train_feature_dicts)
        else:
            X_train = self.vec.transform(train_feature_dicts)

        # Transform the test features
        X_test = self.vec.transform(self.test_feature_dicts)

        # Conduct a round of learning with each of the various learners
        for i, (learner_list,
                learner_name) in enumerate(zip(self.learner_lists,
                                               self.learner_names)):
            for j, learner in enumerate(learner_list):
                if (learner_name in self.__learners_requiring_classes__
                    and self.round == 1):
                    learner.partial_fit(X_train,
                                        y_train,
                                        classes=self.classes)
                else:
                    learner.partial_fit(X_train,
                                        y_train)

                # Make predictions on the test set, rounding the values
                y_test_preds = np.round(learner.predict(X_test))

                # "Rescale" the values (if necessary), forcing the
                # values should fit within the original scale
                y_test_preds = self.fit_preds_in_scale(y_test_preds)

                # Evaluate the new model, collecting metrics, etc., and
                # then store the round statistics
                stats_dict = self.get_stats(y_test_preds)
                stats_dict.update({self.__learning_round__: int(self.round),
                                   self.__prediction_label__:
                                       self.prediction_label,
                                   self.__test_labels_and_preds__:
                                       list(zip(self.y_test,
                                                y_test_preds)),
                                   self.__learner__: learner_name,
                                   self.__params__: learner.get_params(),
                                   self.__training_samples__: samples})
                (self.learner_param_grid_stats[i][j]
                 .append(pd.Series(stats_dict)))

        # Increment the round number
        self.round += 1

    def do_learning_rounds(self) -> None:
        '''
        Do rounds of learning.
        '''

        # If a certain number of rounds has been specified, try to do
        # that many rounds; otherwise, do as many as possible
        if self.rounds > 0:
            while self.round <= self.rounds:
                if self.NO_MORE_TRAINING_DATA:
                    break
                else:
                    self.learning_round()
        else:
            while True:
                if self.NO_MORE_TRAINING_DATA:
                    break
                self.learning_round()


def main():
    parser = ArgumentParser(description='Run incremental learning '
                                        'experiments.',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--game',
                        help='Game to use in experiments.',
                        choices=[game for game in APPID_DICT.keys()
                                 if not game.startswith('sample')],
                        required=True)
    parser.add_argument('--output_dir',
                        help='Directory in which to output data related to '
                             'the results of the conducted experiments.',
                        type=str,
                        required=True)
    parser.add_argument('--rounds',
                        help='The maximum number of rounds of learning to '
                             'conduct (the number of rounds will necessarily '
                             'be limited by the amount of training data and '
                             'the number of samples used per round). Use "0" '
                             'to do as many rounds as possible.',
                        type=int,
                        default=0)
    parser.add_argument('--samples_per_round',
                        help='The maximum number of training samples '
                             'to use in each round.',
                        type=int,
                        default=100)
    parser.add_argument('--test_limit',
                        help='Cap to set on the number of test reviews to use'
                             ' for evaluation.',
                        type=int,
                        default=1000)
    parser.add_argument('--y_label',
                        help='Label to predict.',
                        choices=labels,
                        default='total_game_hours_bin')
    parser.add_argument('--non_nlp_features',
                        help='Comma-separated list of non-NLP features to '
                             'combine with the NLP features in creating a '
                             'model. Use "all" to use all available '
                             'features or "none" to use no non-NLP features.',
                        type=str,
                        default="all")
    parser.add_argument('--learners',
                        help='Comma-separated list of learning algorithms to '
                             'try. Refer to list of learners above to find '
                             'out which abbreviations stand for which '
                             'learners. Set of available learners: {}. Use '
                             '"all" to include all available learners.'
                             .format(learner_abbrs_string),
                        type=str,
                        default='all')
    parser.add_argument('--obj_func',
                        help='Objective function to use in determining which '
                             'set of parameters resulted in the best '
                             'performance.',
                        choices=obj_funcs,
                        default='r')
    parser.add_argument('--evaluate_majority_baseline',
                        help='Evaluate the majority baseline model.',
                        action='store_true',
                        default=False)
    parser.add_argument('-dbhost', '--mongodb_host',
        help='Host that the MongoDB server is running on.',
        type=str,
        default='localhost')
    parser.add_argument('--mongodb_port', '-dbport',
        help='Port that the MongoDB server is running on.',
        type=int,
        default=37017)
    args = parser.parse_args()

    game_id = args.game
    rounds = args.rounds
    samples_per_round = args.samples_per_round
    non_nlp_features = args.non_nlp_features
    y_label = args.y_label
    learners = args.learners
    host = args.mongodb_host
    port = args.mongodb_port
    test_limit = args.test_limit
    output_dir = realpath(args.output_dir)
    obj_func = args.obj_func
    evaluate_majority_baseline = args.evaluate_majority_baseline

    logger.info('Game: {}'.format(game_id))
    logger.info('Maximum number of learning rounds to conduct: {}'
                .format(rounds if rounds > 0
                                  else "as many as possible"))
    logger.info('Maximum number of training samples to use in each eround: {}'
                .format(samples_per_round))

    # Get list of non-NLP features to include; if the prediction label
    # is one of the time features, take other time features out of the
    # set of non-NLP features since the information could be duplicated
    if non_nlp_features:
        if non_nlp_features == 'all':
            non_nlp_features = set(copy(labels))
            if y_label in time_labels:
                [non_nlp_features.remove(feat) for feat in time_labels]
            else:
                non_nlp_features.remove(y_label)
        elif non_nlp_features == "none":
            non_nlp_features = set()
        else:
            non_nlp_features = set(non_nlp_features.split(','))
            if any(not feat in labels for feat in non_nlp_features):
                logger.error('Found unrecognized feature in the list of '
                             'passed-in non-NLP features. Available features:'
                             ' {}. Exiting.'.format(', '.join(labels)))
                exit(1)
            if (y_label in time_labels
                and non_nlp_features.intersection(time_labels)):
                logger.error('The list of non-NLP features should not contain'
                             ' any of the time-related features if the "y" '
                             'label is itself a time-related feature. '
                             'Exiting.')
                exit(1)
    else:
        non_nlp_features = set()
    logger.info('Y label: {}'.format(y_label))
    logger.info('Non-NLP features to use: {}'
                .format(', '.join(non_nlp_features)))

    # Get set of learners to use
    if learners == 'all':
        learners = set(copy(learner_dict_keys))
    else:
        learners = set(learners.split(','))
        if not learners.issubset(learner_dict_keys):
            logger.error('Found unrecognized learner(s) in list of passed-in '
                         'learners: {}. Available learners: {}. Exiting.'
                         .format(', '.join(learners),
                                 learner_abbrs_string))
            exit(1)
    logger.info('Learners: {}'.format(', '.join([learner_abbrs_dict[learner]
                                                 for learner in learners])))
    logger.info('Using {} as the objective function'.format(obj_func))

    # Connect to running Mongo server
    logger.info('MongoDB host: {}'.format(host))
    logger.info('MongoDB port: {}'.format(port))
    db = connect_to_db(host=host,
                       port=port)

    # Log info about data
    logger.info('Game: {}'.format(game))
    logger.info('Limiting number of test reviews to {} or below'
                .format(test_limit))

    # Do learning experiments
    logger.info('Starting incremental learning experiments...')
    inc_learning = \
        IncrementalLearning(db,
                            game,
                            [learner_dict[learner] for learner in learners],
                            [_find_default_param_grid(learner)
                             for learner in learners],
                            train_cursor,
                            test_cursor,
                            samples_per_round,
                            non_nlp_features,
                            y_label,
                            obj_func,
                            test_limit=test_limit,
                            rounds=rounds,
                            majority_baseline=evaluate_majority_baseline)

    # Output results files to output directory
    logger.info('Output directory: {}'.format(output_dir))
    makedirs(output_dir,
             exist_ok=True)

    # Generate evaluation reports for the various learner/parameter
    # grid combinations
    logger.info('Generating reports for the incremental learning runs...')
    for learner_name, learner_param_grid_stats \
        in zip(inc_learning.learner_names,
               inc_learning.learner_param_grid_stats):
        for i, stats_df in enumerate(learner_param_grid_stats):
            stats_df.to_csv(join(output_dir,
                                 '{}_inc_learning_learner_stats_{}.csv'
                                 .format(learner_name, i)),
                            index=False)

    # Generate evaluation report for the majority baseline model, if
    # specified
    if evaluate_majority_baseline:
        logger.info('Generating report for the majority baseline model...')
        logger.info('Majority label: {}'.format(inc_learning.majority_label))
        (inc_learning.majority_baseline_stats
         .to_csv(join(output_dir,
                      'majority_baseline_model_stats.csv'),
                 index=False))

    logger.info('Complete.')


if __name__ == '__main__':
    main()
