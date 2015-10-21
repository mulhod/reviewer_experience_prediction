#!/usr/env python3.4
from sys import exit
from os import (listdir,
                makedirs)
from os.path import (join,
                     exists,
                     dirname,
                     realpath)
from operator import or_
import logging
# Set up logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -'
                              ' %(message)s')
sh.setFormatter(formatter)
sh.setLevel(logging.INFO)
logger.addHandler(sh)

from src.features import *
from util.datasets import *
from util.mongodb import *
from data import APPID_DICT

import numpy as np
import pandas as pd
from bson import BSON
from pymongo import cursor
from argparse import (ArgumentParser,
                      ArgumentDefaultsHelpFormatter)
from sklearn.grid_search import ParameterGrid
from sklearn.feature_extraction import DictVectorizer
from skll.metrics import kappa
from scipy.stats import pearsonr
from sklearn.metrics import (precision_score,
                             f1_score,
                             accuracy_score,
                             confusion_matrix)
from sklearn.cluster import MiniBatchKMeans
from sklearn.naive_bayes import (BernoulliNB,
                                 MultinomialNB)
from sklearn.linear_model import (Perceptron,
                                  SGDRegressor,
                                  PassiveAggressiveRegressor)

project_dir = dirname(dirname(realpath(__file__)))

# Filter out warnings since there will be a lot of
# "UndefinedMetricWarning" warnings when running IncrementalLearning
import warnings
warnings.filterwarnings("ignore")

seed = 123456789

# Define default parameter grids
_DEFAULT_PARAM_GRIDS = \
    {PCA: {'n_components': [None, 'mle'],
           'whiten': [True, False]},
     MiniBatchKMeans: {'n_clusters': [4, 6, 8, 12],
                       'init' : ['k-means++', 'random'],
                       'random_state': [seed]},
     BernoulliNB: {'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]},
     MultinomialNB: {'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]},
     Perceptron: {'penalty': [None, 'l2', 'l1', 'elasticnet'],
                  'alpha': [0.0001, 0.001, 0.01, 0.1],
                  'n_iter': [5, 10],
                  'random_state': [seed]},
     SGDRegressor: {'alpha': [0.000001, 0.00001, 0.0001, 0.001,
                              0.01],
                    'penalty': ['l1', 'l2', 'elasticnet']},
     PassiveAggressiveRegressor:
         {'C': [0.01, 0.1, 1.0, 10.0, 100.0],
          'n_iter': [5, 10],
          'random_state': [seed],
          'loss': ['epsilon_insensitive',
                   'squared_epsilon_insensitive']},
     IncrementalPCA: {'whiten': [True, False]},
     MiniBatchDictionaryLearning:
         {'n_components': [100, 500, 1000, 10000],
          'n_iter': [5, 10, 15],
          'fit_algorithm': ['lars'],
          'transform_algorithm': ['lasso_lars'],
          'random_state': [seed]}}

learner_dict = {'mbkm': MiniBatchKMeans,
                'bnb': BernoulliNB,
                'mnb': MultinomialNB,
                'perc': Perceptron,
                'sgd': SGDRegressor,
                'pagr': PassiveAggressiveRegressor}

obj_funcs = ['r', 'significance', 'precision_macro', 'precision_weighted',
             'f1_macro', 'f1_weighted', 'accuracy', 'uwk', 'uwk_off_by_one',
             'qwk', 'qwk_off_by_one', 'lwk', 'lwk_off_by_one']

labels = ['num_guides', 'num_games_owned', 'num_friends',
          'num_voted_helpfulness', 'num_groups', 'num_workshop_items',
          'num_reviews', 'num_found_funny', 'friend_player_level',
          'num_badges', 'num_found_helpful', 'num_screenshots',
          'num_found_unhelpful', 'found_helpful_percentage', 'num_comments',
          'total_game_hours', 'total_game_hours_bin',
          'total_game_hours_last_two_weeks', 'num_achievements_percentage',
          'num_achievements_possible']
time_labels = ['total_game_hours', 'total_game_hours_bin',
               'total_game_hours_last_two_weeks']

class IncrementalLearning:
    '''
    Class for conducting incremental learning experiments with a
    parameter grid and a learner.
    '''

    # Constants
    __nlp_feats__ = 'nlp_features'
    __achieve_prog__ = 'achievement_progress'
    __nan__ = float("NaN")
    __x__ = 'x'
    __y__ = 'y'
    __id_string__ = 'id_string'
    __id__ = 'id'
    __learning_round__ = 'learning_round'
    __prediction_label__ = 'prediction_label'
    __test_labels_and_preds__ = 'test_set_labels/test_set_predictions'
    __learner__ = 'learner'
    __learners_requiring_classes__ = ['BernoulliNB', 'MultinomialNB',
                                      'Perceptron']
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
    __possible_non_nlp_features__ = list(labels)
    __tab_join__ = '\t'.join
    __cnfmat_row__ = '{}{}\n'.format
    __cnfmat_header__ = ('confusion_matrix (rounded predictions) '
                         '(row=human, col=machine, labels={}):\n')

    def __init__(self, learners, param_grid: dict,
                 training_data_cursor: cursor, test_data_cursor: cursor,
                 round_size: int, non_nlp_features: list, prediction_label: str,
                 rounds=0):
        '''
        Initialize class.

        :param learners: algorithms to use for learning
        :type learners: list of learners
        :param param_grids: list of dictionaries of parameters mapped
                            to lists of values (must be aligned with
                            list of learners)
        :type param_grids: dict
        :param training_data_cursor: MongoDB cursor for training
                                     documents
        :type training_data_cursor: pymongo cursor object
        :param test_data_cursor: MongoDB cursor for test documents
        :type test_data_cursor: pymongo cursor object
        :param round_size: number of training documents to extract in
                           each round
        :type round_size: int
        :param non_nlp_features: list of non-NLP features to add into
                                 the feature dictionaries 
        :type non_nlp_features: list of str
        :param prediction_label: feature to predict
        :type prediction_label: str
        :param rounds: number of rounds of learning (0 for as many as
                       possible)
        :type rounds: int
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

        # Learner-related variables
        self.vec = None
        self.param_grids = [list(ParameterGrid(param_grid)) for param_grid
                            in param_grids]
        self.learner_names = [(str(learner.__class__)
                               .rsplit('.', 1)[1]
                               .strip("'>")) for learner in learners]
        self.learner_lists = [[learner(**kwparams) for kwparams in param_grid]
                              for learner, param_grid
                              in zip(learners,
                                     self.param_grids)]
        self.learner_param_grid_stats = []
        for learner_list in self.learner_lists:
            self.learner_param_grid_stats.append([[] for _ in learner_list])

        # Information about what features to use for what purposes
        if all([feat in self.__possible_non_nlp_features__
                for feat in non_nlp_features]):
            self.non_nlp_features = non_nlp_features
        self.prediction_label = prediction_label

        # Information about incremental learning
        self.round_size = round_size
        self.rounds = rounds
        self.round = 1
        self.NO_MORE_TRAINING_DATA = False

        # Training/test data variables
        self.training_cursor = training_data_cursor
        self.test_cursor = test_data_cursor
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
        self.learner_stats = [pd.DataFrame(learner_stats) for learner_stats
                              in self.learner_stats]

    def get_all_features(self, review_doc):
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
        # Add in the non-NLP features (except for those that may be in the
        # 'achievement_progress' sub-dictionary of the review dictionary
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
        # Add in the 'id_string' value just to make it easier to process the
        # results of this function
        features.update({self.__id_string__: _get(self.__id_string__)})
        return features

    def get_train_data_iteration(self):
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

    def get_test_data(self):
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

    def make_printable_confusion_matrix(self, y_pred):
        '''
        Produce a printable confusion matrix to use in the evaluation
        report.

        :param y_pred: array-like of predicted labels
        :type y_pred: array-like
        :returns: str, np.ndarray
        '''

        cnfmat = confusion_matrix(self.y_test,
                                  np.round(y_pred),
                                  labels=self.classes).tolist()
        res = str(self.cnfmat_desc)
        for row, label in zip(cnfmat,
                              self.classes):
            row = self.__tab_join__([str(x) for x in [label] + row])
            res = self.__cnfmat_row__(res, row)
        return res, cnfmat

    def get_stats(self, y_pred):
        """
        Get some statistics about the model's performance on the test
        set.

        :param y_pred: predictions
        :type y_pred: np.array
        :returns: dict
        """

        # Get Pearson r and significance
        r, sig = pearsonr(self.y_test,
                          y_pred)

        # Get confusion matrix (both the np.ndarray and the printable one)
        printable_cnfmat, cnfmat = \
            self.make_printable_confusion_matrix(y_pred)

        return {self.__r__: r,
                self.__sig__: sig,
                self.__prec_macro__: precision_score(self.y_test,
                                                     y_pred,
                                                     labels=self.classes,
                                                     average='macro'),
                self.__prec_weighted__: precision_score(self.y_test,
                                                        y_pred,
                                                        labels=self.classes,
                                                        average='weighted'),
                self.__f1_macro__: f1_score(self.y_test,
                                            y_pred,
                                            labels=self.classes,
                                            average='macro'),
                self.__f1_weighted__: f1_score(self.y_test,
                                               y_pred,
                                               labels=self.classes,
                                               average='weighted'),
                self.__acc__: accuracy_score(self.y_test,
                                             y_pred,
                                             normalize=True),
                self.__cnfmat__: cnfmat,
                self.__printable_cnfmat__: printable_cnfmat,
                self.__uwk__: kappa(self.y_test,
                                    y_pred),
                self.__uwk_off_by_one__: kappa(self.y_test,
                                               y_pred,
                                               allow_off_by_one=True),
                self.__qwk__: kappa(self.y_test,
                                    y_pred,
                                    weights='quadratic'),
                self.__qwk_off_by_one__: kappa(self.y_test,
                                               y_pred,
                                               weights='quadratic',
                                               allow_off_by_one=True),
                self.__lwk__: kappa(self.y_test,
                                    y_pred,
                                    weights='linear'),
                self.__lwk_off_by_one__: kappa(self.y_test,
                                               y_pred,
                                               weights='linear',
                                               allow_off_by_one=True)}

    def learning_round(self):
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

        # Update the various models with differing parameters
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
                y_test_preds = learner.predict(X_test)

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
                self.learner_param_grid_stats[j].append(pd.Series(stats_dict))

        # Increment the round number
        self.round += 1

    def do_learning_rounds(self):
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
                        choices=[labels],
                        default='total_game_hours_bin')
    parser.add_argument('--non_nlp_features',
                        help='Comma-separated list of non-NLP features to '
                             'combine with the NLP features in creating a '
                             'model. Use "all" to use all available '
                             'features or "none" to use no non-NLP features.',
                        default="all")
    parser.add_argument('--learners',
                        help='Comma-separated list of learning algorithms to '
                             'try. Refer to list of learners above to find '
                             'out which abbreviations stand for which '
                             'learners. Set of available learners: {}. Use '
                             '"all" to include all available learners.'
                             .format(', '.join(learners.keys()))
                        type=str,
                        default='all',
                        required=True)
    parser.add_argument('--obj_func',
                        help='Objective function to use in determining which '
                             'set of parameters caused the best performance.',
                        choices=obj_funcs,
                        default='r')
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
    port = args.mongodb_host
    test_limit = args.test_limit
    output_dir = realpath(args.output_dir)

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
            non_nlp_features = list(labels)
            if y_label in time_labels:
                for feat in time_labels:
                    del non_nlp_features[non_nlp_features.index(feat)]
            else:
                del non_nlp_features[non_nlp_features.index(y_label)]
        elif non_nlp_features = "none":
            non_nlp_features = []
        else:
            non_nlp_features = non_nlp_features.split(',')
            if any([not feat in labels for feat in non_nlp_features]):
                logger.error('Found unrecognized feature in the list of '
                             'passed-in non-NLP features. Available features:'
                             ' {}. Exiting.'.format(', '.join(labels)))
                exit(1)
            if (y_label in time_labels
                and any([feat in time_labels for feat in non_nlp_features])):
                logger.error('The list of non-NLP features should not contain'
                             ' any of the time-related features if the "y" '
                             'label is itself a time-related feature. '
                             'Exiting.')
                exit(1)
    else:
        non_nlp_features = []
    logger.info('Y label: {}'.format(y_label))
    logger.info('Non-NLP features to use: {}'
                .format(', '.join(non_nlp_features)))

    # Get set of learners to use
    if learners == 'all':
        learners = list(learner_dict.keys())
    else:
        learners = learners.split(',')
        if any([not learner in learner_dict.keys() for learner in learners]):
            logger.error('Found unrecognized learner in list of passed-in '
                         'learners. Available learners: {}. Exiting.'
                         .format(', '.join(learner_dict.keys())))
            exit(1)
    logger.info('Learners: {}'.format(', '.join(learners)))

    # Connect to running Mongo server
    logger.info('MongoDB host: {}'.format(host))
    logger.info('MongoDB port: {}'.format(port))
    db = connect_to_db(host=host,
                       port=port)

    # Get training/test data cursors
    batch_size = 50
    logger.info('Cursor batch size: {}'.format(batch_size))
    train_cursor = db.find({'game': game_id,
                            'partition': 'training'},
                           timeout=False)
    train_cursor.batch_size = batch_size
    test_cursor = db.find({'game': game_id,
                           'partition': 'test'},
                          timeout=False).limit(test_limit)
    test_cursor.batch_size = batch_size
    logger.info('Limiting number of test reviews to {} or below'
                .format(test_limit))

    # Do learning experiments
    logger.info('Starting incremental learning experiments...')
    inc_learning = \
        IncrementalLearning([learner_dict[learner] for learner in learners],
                            [_DEFAULT_PARAM_GRIDS[learner_dict[learner]]
                             for learner in learners],
                            train_cursor,
                            test_cursor,
                            samples_per_round,
                            non_nlp_features,
                            y_label,
                            rounds=rounds)

    # Output results files to output directory
    logger.info('Output directory: {}'.format(output_dir))
    makedirs(output_dir,
             exist_ok=True)
    


if __name__ == '__main__':
    main()
