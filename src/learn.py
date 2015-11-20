#!/usr/env python3.4
"""
:author: Matt Mulholland (mulhodm@gmail.com)
:date: 10/14/2015

Command-line utility for the IncrementalLearning class, which enables
one to run experiments on subsets of the data with a number of
different machine learning algorithms and parameter customizations,
etc.
"""
import logging
from json import dump
from os import makedirs
from itertools import chain
from os.path import (join,
                     isdir,
                     isfile,
                     exists,
                     dirname,
                     realpath)
from warnings import filterwarnings

import numpy as np
import pandas as pd
from bson import BSON
from pymongo import (ASCENDING,
                     collection)
from skll.metrics import kappa
from scipy.stats import (mode,
                         pearsonr,
                         linregress)
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
from src import log_format_string
from src import experiments as ex
from util.mongodb import connect_to_db
from util.datasets import (get_bin,
                           get_bin_ranges_helper)

# Filter out warnings since there will be a lot of
# "UndefinedMetricWarning" warnings when running IncrementalLearning
filterwarnings("ignore")

ORDERINGS = frozenset({'objective_last_round', 'objective_best_round',
                       'objective_slope'})

class RunExperiments:
    """
    Class for conducting sets of incremental learning experiments.
    """

    # Constants
    __game__ = 'game'
    __games__ = 'games'
    __test_games__ = 'test_games'
    __all_games__ = 'all_games'
    __partition__ = 'partition'
    __training__ = 'training'
    __test__ = 'test'
    __nlp_feats__ = 'nlp_features'
    __bin_ranges__ = 'bin_ranges'
    __achieve_prog__ = 'achievement_progress'
    __steam_id__ = 'steam_id_number'
    __in_op__ = '$in'
    __nan__ = float("NaN")
    __x__ = 'x'
    __y__ = 'y'
    __id_string__ = 'id_string'
    __id__ = 'id'
    __obj_id__ = '_id'
    __macro__ = 'macro'
    __weighted__ = 'weighted'
    __linear__ = 'linear'
    __quadratic__ = 'quadratic'
    __learning_round__ = 'learning_round'
    __prediction_label__ = 'prediction_label'
    __test_labels_and_preds__ = 'test_set_labels/test_set_predictions'
    __non_nlp_features__ = 'non-NLP features'
    __no_nlp_features__ = 'no NLP features'
    __learner__ = 'learner'
    __learners_requiring_classes__ = frozenset({'BernoulliNB', 'MultinomialNB',
                                                'Perceptron'})
    __learner_names__ = {MiniBatchKMeans: 'MiniBatchKMeans',
                         BernoulliNB: 'BernoulliNB',
                         MultinomialNB: 'MultinomialNB',
                         Perceptron: 'Perceptron',
                         PassiveAggressiveRegressor: 'PassiveAggressiveRegressor'}
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
    __possible_non_nlp_features__ = set(ex.LABELS)
    __orderings__ = ORDERINGS
    __tab_join__ = '\t'.join
    __cnfmat_row__ = '{0}{1}\n'.format
    __cnfmat_header__ = ('confusion_matrix (rounded predictions) '
                         '(row=actual, col=machine, labels={0}):\n')
    __majority_label__ = 'majority_label'
    __labels_string__ = ', '.join(ex.LABELS)
    __majority_baseline_model__ = 'majority_baseline_model'

    def __init__(self, db: collection, games: set, test_games: set, learners,
                 param_grids: dict, round_size: int, non_nlp_features: list,
                 prediction_label: str, objective: str,
                 logger: logging.RootLogger, no_nlp_features=False,
                 bin_ranges=None, test_limit=0, rounds=0,
                 majority_baseline=True):
        """
        Initialize class.

        :param db: MongoDB database collection object
        :type db: collection
        :param games: set of games to use for training models
        :type games: set
        :param test_games: set of games to use for evaluating models
        :type test_games: set
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
        :param logger: logger instance
        :type logger: logging.RootLogger
        :param no_nlp_features: leave out NLP features
        :type no_nlp_features: boolean
        :param bin_ranges: list of tuples representing the maximum and
                           minimum values corresponding to bins (for
                           splitting up the distribution of prediction
                           label values)
        :type bin_ranges: None or list of tuple
        :param test_limit: limit for the number of test samples
                           (defaults to 0 for no limit)
        :type test_limit: int
        :param rounds: number of rounds of learning (0 for as many as
                       possible)
        :type rounds: int
        :param majority_baseline: evaluate a majority baseline model
        :type majority_baseline: bool

        :returns: instance of IncrementalLearning class
        :rtype: IncrementalLearning

        :raises: ValueError
        """

        self.logger = logger

        # Make sure parameters make sense/are valid
        if round_size < 1:
            raise ValueError('The round_size parameter should have a positive'
                             ' value.')
        if prediction_label in non_nlp_features:
            raise ValueError('The prediction_label parameter ({0}) cannot '
                             'also be in the list of non-NLP features to use '
                             'in the model:\n\n{1}\n.'
                             .format(prediction_label,
                                     ', '.join(non_nlp_features)))
        if any(not feat in self.__possible_non_nlp_features__
               for feat in non_nlp_features):
            raise ValueError('All non-NLP features must be included in the '
                             'list of available non-NLP features: {0}.'
                             .format(self.__labels_string__))
        if not prediction_label in self.__possible_non_nlp_features__:
            raise ValueError('The prediction label must be in the set of '
                             'features that can be extracted/used, i.e.: {0}.'
                             .format(self.__labels_string__))
        if not all(_games.issubset(ex.VALID_GAMES) for _games
                   in [games, test_games]):
            raise ValueError('Unrecognized game(s)/test game(s): {0}. The '
                             'games must be in the following list of '
                             'available games: {1}.'
                             .format(', '.join(games.union(test_games)),
                                     ', '.join(APPID_DICT)))

        # MongoDB database
        self.db = db

        # Games
        self.games = games
        if not self.games:
            raise ValueError('The set of games must be greater than zero!')
        self.__games_string__ = ', '.join(self.games)
        self.__file_name_template__ = ('{0}_{1}_{2}_{3}.csv'
                                       .format(self.__games_string__, '{0}',
                                               '{1}', '{2}'))
        self.__report_name_template__ = (self.__file_name_template__
                                         .format('{0}', 'learning_stats', '{1}'))
        self.__model_weights_name_template__ = (self.__file_name_template__
                                                .format('{0}', 'model_weights',
                                                        '{1}'))
        if majority_baseline:
            self.__majority_baseline_report_name__ = \
                ('{0}_majority_baseline_model_stats.csv'
                 .format(self.__games_string__))
        self.test_games = test_games
        if not self.test_games:
            raise ValueError('The set of games must be greater than zero!')
        self.__test_games_string__ = ', '.join(self.test_games)
        self.bin_ranges = bin_ranges

        # Objective function
        self.objective = objective
        if not self.objective in ex.OBJ_FUNC_ABBRS_DICT:
            raise ValueError('Unrecognized objective function used: {0}. '
                             'These are the available objective functions: {1}.'
                             .format(self.objective, ex.OBJ_FUNC_ABBRS_STRING))

        # Learner-related variables
        self.vec = None
        self.param_grids = [list(ParameterGrid(param_grid)) for param_grid
                            in param_grids]
        self.learner_names = [self.__learner_names__[learner] for learner
                              in learners]
        self.learner_lists = [[learner(**kwparams) for kwparams in param_grid]
                              for learner, param_grid in zip(learners,
                                                             self.param_grids)]
        self.learner_param_grid_stats = []
        for learner_list in self.learner_lists:
            self.learner_param_grid_stats.append([[] for _ in learner_list])

        # Information about what features to use for what purposes
        self.non_nlp_features = non_nlp_features
        self.no_nlp_features = no_nlp_features
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
        self.logger.info('Setting up MongoDB cursors for training/evaluation '
                         'data...')
        self.make_cursors()
        self.logger.info('Extracting evaluation dataset...')
        self.test_data = self.get_test_data()
        self.test_ids = [_data[self.__id__] for _data in self.test_data]
        self.test_feature_dicts = [_data[self.__x__] for _data in self.test_data]
        self.y_test = np.array([_data[self.__y__] for _data in self.test_data])
        self.classes = np.unique(self.y_test)
        self.logger.info('Prediction label classes: {0}'
                         .format(', '.join([str(x) for x in self.classes])))

        # Useful constants for use in make_printable_confusion_matrix
        self.cnfmat_desc = \
            self.__cnfmat_row__(
                self.__cnfmat_header__.format(self.classes),
                self.__tab_join__([''] + [str(x) for x in self.classes])
                )

        # Do incremental learning experiments
        self.logger.info('Incremental learning experiments initialized...')
        self.do_learning_rounds()
        self.learner_param_grid_stats = \
            [[pd.DataFrame(param_grid) for param_grid in learner]
             for learner in self.learner_param_grid_stats]

        # Generate statistics for the majority baseline model
        if majority_baseline:
            self.majority_label = None
            self.majority_baseline_stats = None
            self.evaluate_majority_baseline_model()

    def make_cursors(self) -> None:
        """
        Make cursor objects for the training/test sets.

        :rtype: None
        """

        batch_size = 50
        self.logger.debug('Batch size of MongoDB cursors: {0}'.format(batch_size))
        sorting_args = [(self.__steam_id__, ASCENDING)]

        # Leave out the '_id' value and the 'nlp_features' value if
        # `self.no_nlp_features` is true
        projection = {self.__obj_id__: 0}
        if self.no_nlp_features:
            projection.update({self.__nlp_feats__: 0})

        # Make training data cursor
        if len(self.games) == 1:
            train_query = {self.__game__: list(self.games)[0],
                           self.__partition__: self.__training__}
        elif not ex.VALID_GAMES.difference(self.games):
            train_query = {self.__partition__: self.__training__}
        else:
            train_query = {self.__game__: {self.__in_op__: list(self.games)},
                           self.__partition__: self.__training__}
        self.training_cursor = (self.db
                                .find(train_query, projection, timeout=False)
                                .sort(sorting_args))
        self.training_cursor.batch_size = batch_size

        # Make test data cursor
        if len(self.test_games) == 1:
            test_query = {self.__game__: list(self.test_games)[0],
                          self.__partition__: self.__test__}
        elif not ex.VALID_GAMES.difference(self.test_games):
            test_query = {self.__partition__: self.__test__}
        else:
            test_query = {self.__game__: {self.__in_op__: list(self.test_games)},
                           self.__partition__: self.__test__}
        self.test_cursor = (self.db
                            .find(test_query, projection, timeout=False)
                            .sort(sorting_args))
        if self.test_limit:
            self.test_cursor = self.test_cursor.limit(self.test_limit)
        self.test_cursor.batch_size = batch_size

    def get_all_features(self, review_doc: dict):
        """
        Get all the features in a review document and put them together
        in a dictionary. If `self.no_nlp_features` is true, leave out
        NLP features. If bin_ranges is specified, convert the value of
        the prediction label to the bin index.

        :param review_doc: review document from Mongo database
        :type review_doc: dict

        :returns: feature dictionary
        :rtype: dict or None
        """

        _get = review_doc.get
        features = {}
        _update = features.update

        # Add in the NLP features
        if not self.no_nlp_features:
            _update({feat: val for feat, val
                     in BSON.decode(_get(self.__nlp_feats__)).items()
                     if val and val != self.__nan__})

        # Add in the non-NLP features (except for those that may be in
        # the 'achievement_progress' sub-dictionary of the review
        # dictionary)
        _update({feat: val for feat, val in review_doc.items()
                 if (feat in self.__possible_non_nlp_features__
                     and val and val != self.__nan__)})

        # Add in the features that may be in the 'achievement_progress'
        # sub-dictionary of the review document
        _update({feat: val for feat, val
                 in _get(self.__achieve_prog__, dict()).items()
                 if (feat in self.__possible_non_nlp_features__
                     and val and val != self.__nan__)})

        # Convert prediction label if `self.bin_ranges` is specified
        # and the prediction label is in the features dictionary;
        # if the prediction label is not in the dictionary, return
        _label = features.get(self.prediction_label)
        if _label:
            # If `self.bin_ranges` was specified, convert the value of the
            # prediction label (if present) to the corresponding bin
            if self.bin_ranges:
                features[self.prediction_label] = self.convert_value_to_bin(_label)
        else:
            return

        # Add in the 'id_string' value just to make it easier to
        # process the results of this function
        _update({self.__id_string__: _get(self.__id_string__)})
        return features

    def get_data(self, review_doc: dict):
        """
        Collect data from a MongoDB review document and return it in
        format needed for DictVectorizer.

        :param review_doc: document from the MongoDB reviews collection
        :type review_doc: 

        :returns: training/test sample
        :rtype: dict or None
        """

        # Get dictionary containing all features needed + the ID and
        # the prediction label
        feature_dict = self.get_all_features(review_doc)

        # Skip over any feature dictionaries that are empty (i.e., due
        # to the absence of the prediction label, or if for some reason
        # the dictionary is otherwise empty)
        if not feature_dict:
            return

        # Get prediction label and remove it from the feature
        # dictionary
        y_value = feature_dict[self.prediction_label]
        del feature_dict[self.prediction_label]

        # Get ID and remove from feature dictionary
        id_string = feature_dict[self.__id_string__]
        del feature_dict[self.__id_string__]

        # Only keep the non-NLP features that are supposed to be kept,
        # if any
        for feat in self.__possible_non_nlp_features__:
            if (not feat in self.non_nlp_features
                and feature_dict.get(feat, None) != None):
                del feature_dict[feat]

        # If, after taking out the prediction label and the ID, there
        # are no remaining features, return None
        if not feature_dict:
            return

        # Return dictionary of features
        return dict(y=y_value, id=id_string, x=feature_dict)

    def get_train_data_iteration(self) -> list:
        """
        Get a list of training data dictionaries to use in model
        training.

        :returns: list of sample dictionaries
        :rtype: list
        """

        data = []
        i = 0
        while i < self.round_size:
            # Get a review document from the Mongo database
            try:
                review_doc = next(self.training_cursor)
            except StopIteration:
                self.NO_MORE_TRAINING_DATA = True
                break

            # Get features, prediction label, and ID in a new
            # dictionary and append to list of data samples
            sample = self.get_data(review_doc)
            if sample:
                data.append(sample)
                i += 1

        return data

    def get_test_data(self) -> list:
        """
        Get a list of test data dictionaries to use in model
        evaluation.

        :returns: list of sample dictionaries
        :rtype: list
        """

        data = []
        for review_doc in self.test_cursor:
            # Get features, prediction label, and ID in a new
            # dictionary and append to list of data samples
            sample = self.get_data(review_doc)
            if sample:
                data.append(sample)

        return data

    def get_majority_baseline(self) -> np.array:
        """
        Generate a majority baseline array of prediction labels.

        :returns: array of prediction labels
        :rtype: np.array
        """

        self.majority_label = mode(self.y_test).mode[0]
        return np.array([self.majority_label]*len(self.y_test))

    def evaluate_majority_baseline_model(self) -> None:
        """
        Evaluate the majority baseline model predictions.

        :rtype: None
        """

        stats_dict = self.get_stats(self.get_majority_baseline())
        stats_dict.update({self.__test_games__:
                               ', '.join(self.test_games)
                               if self.test_games.difference(ex.VALID_GAMES)
                               else self.__all_games__,
                           self.__prediction_label__: self.prediction_label,
                           self.__majority_label__: self.majority_label,
                           self.__learner__: self.__majority_baseline_model__})
        if self.bin_ranges:
            stats_dict.update({self.__bin_ranges__: self.bin_ranges})
        self.majority_baseline_stats = pd.DataFrame([pd.Series(stats_dict)])

    def generate_majority_baseline_report(self, output_path) -> None:
        """
        Generate a CSV file reporting on the performance of the
        majority baseline model.

        :param output_path: path to destination directory
        :type str:

        :returns: None
        :rtype: None
        """

        (self.majority_baseline_stats
         .to_csv(join(output_path, self.__majority_baseline_report_name__),
                 index=False))

    def generate_learning_reports(self, output_path,
                                  ordering='objective_last_round') -> None:
        """
        Generate experimental reports for each run represented in the
        lists of input dataframes.

        The output files will have indices in their names, which simply
        correspond to the sequence in which they occur in the list of
        input dataframes.

        :param output_path: path to destination directory
        :type output_path: str
        :param ordering: ordering type for ranking the reports (see
                         `ORDERINGS`)
        :type ordering: str

        :rtype: None
        """

        # Rank the experiments by the given ordering type
        try:
            dfs = self.rank_experiments_by_objective(ordering=ordering)
        except ValueError:
            raise e

        for i, df in enumerate(dfs):
            learner_name = df[self.__learner__].iloc[0]
            df.to_csv(join(output_path,
                           self.__report_name_template__.format(learner_name,
                                                                i + 1)),
                      index=False)

    def convert_value_to_bin(self, val) -> int:
        """
        Conver the value to the index of the bin in which it resides.

        :param val: 
        :type val: int or float

        :returns: index of bin containg value
        :rtype: int
        """

        if not self.bin_ranges:
            return val
        return get_bin(self.bin_ranges, val)

    def make_printable_confusion_matrix(self, y_preds) -> tuple:
        """
        Produce a printable confusion matrix to use in the evaluation
        report.

        :param y_preds: array-like of predicted labels
        :type y_preds: array-like

        :returns: (printable confusion matrix: str, confusion matrix:
                  np.ndarray)
        :rtype: tuple
        """

        cnfmat = confusion_matrix(self.y_test, np.round(y_preds),
                                  labels=self.classes).tolist()
        res = str(self.cnfmat_desc)
        for row, label in zip(cnfmat, self.classes):
            row = self.__tab_join__([str(x) for x in [label] + row])
            res = self.__cnfmat_row__(res, row)

        return res, cnfmat

    def fit_preds_in_scale(self, y_preds):
        """
        Force values at either end of the scale to fit within the scale
        by adding to or truncating the values.

        :param y_preds: array-like of predicted labels
        :type y_preds: array-like

        :returns: array of prediction labels
        :rtype: array-like
        """

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

        :returns: statistics dictionary
        :rtype: dict
        """

        # Get Pearson r and significance
        r, sig = pearsonr(self.y_test, y_preds)

        # Get confusion matrix (both the np.ndarray and the printable
        # one)
        printable_cnfmat, cnfmat = self.make_printable_confusion_matrix(y_preds)

        return {self.__r__: r,
                self.__sig__: sig,
                self.__prec_macro__: precision_score(self.y_test, y_preds,
                                                     labels=self.classes,
                                                     average=self.__macro__),
                self.__prec_weighted__:
                    precision_score(self.y_test, y_preds, labels=self.classes,
                                    average=self.__weighted__),
                self.__f1_macro__: f1_score(self.y_test, y_preds,
                                            labels=self.classes,
                                            average=self.__macro__),
                self.__f1_weighted__: f1_score(self.y_test, y_preds,
                                               labels=self.classes,
                                               average=self.__weighted__),
                self.__acc__: accuracy_score(self.y_test, y_preds, normalize=True),
                self.__cnfmat__: cnfmat,
                self.__printable_cnfmat__: printable_cnfmat,
                self.__uwk__: kappa(self.y_test, y_preds),
                self.__uwk_off_by_one__: kappa(self.y_test, y_preds,
                                               allow_off_by_one=True),
                self.__qwk__: kappa(self.y_test, y_preds,
                                    weights=self.__quadratic__),
                self.__qwk_off_by_one__: kappa(self.y_test, y_preds,
                                               weights=self.__quadratic__,
                                               allow_off_by_one=True),
                self.__lwk__: kappa(self.y_test, y_preds, weights=self.__linear__),
                self.__lwk_off_by_one__: kappa(self.y_test, y_preds,
                                               weights=self.__linear__,
                                               allow_off_by_one=True)}

    def rank_experiments_by_objective(self, ordering: str) -> list:
        """
        Rank the experiments in relation to their performance in the
        objective function.

        :param ordering: ordering type (see `ORDERINGS`)
        :type ordering: str

        :returns: list of dataframes
        :rtype: list
        """

        if not ordering in self.__orderings__:
            raise ValueError('ordering parameter not in the set of possible '
                             'orderings: {0}'
                             .format(', '.join(self.__orderings__)))

        # Keep track of the performance
        dfs = []
        performances = []

        # Iterate over all experiments
        for learner_param_grid_stats in self.learner_param_grid_stats:
            for stats_df in learner_param_grid_stats:

                # Fill "na" values with 0
                stats_df = stats_df.fillna(value=0)
                dfs.append(stats_df)

                if ordering == 'objective_last_round':
                    # Get the performance in the last round
                    performances.append(stats_df[self.objective][len(stats_df) - 1])
                elif ordering == 'objective_best_round':
                    # Get the best performance (in any round)
                    performances.append(stats_df[self.objective].max())
                else:
                    # Get the slope of the performance as the learning
                    # round increases
                    regression = linregress(stats_df[self.__learning_round__],
                                            stats_df[self.objective])
                    performances.append(regression.slope)

        # Sort dataframes on ordering value and return
        return [df[1] for df in sorted(zip(performances, dfs), key=lambda x: x[0],
                                       reverse=True)]

    def get_sorted_features_for_learner(self, learner,
                                        filter_zero_features=True) -> list:
        """
        Get the best-performing features in a learner (excluding
        MiniBatchKMeans).

        :param learner: learner
        :type learner: learner instance
        :param filter_zero_features: filter out features with
                                     zero-valued coefficients
        :type filter_zero_features: bool

        :returns: list of sorted features (in dictionaries)
        :rtype: list
        """

        # Store feature coefficient tuples
        feature_coefs = []

        # Get tuples of feature + label/coefficient tuples, one for
        # each label
        for index, feat in enumerate(self.vec.get_feature_names()):

            # Get list of coefficient arrays for the different classes
            try:
                coef_indices = [learner.coef_[i][index] for i, _
                                in enumerate(self.classes)]
            except IndexError:
                self.logger.error('Could not get feature coefficients!')
                return None

            # Append feature coefficient tuple to list of tuples
            feature_coefs.append(tuple(list(chain([feat],
                                                  zip(self.classes,
                                                      coef_indices)))))

        # Unpack tuples of features and label/coefficient tuples into
        # one long list of feature/label/coefficient values, sort, and
        # filter out any tuples with zero weight
        features = []
        for i, _label in enumerate(self.classes):
            features.extend(
                [dict(feature=coefs[0], label=coefs[i + 1][0],
                      weight=coefs[i + 1][1])
                 for coefs in feature_coefs if coefs[i + 1][1]]
                )

        return sorted(features, key=lambda x: abs(x['weight']), reverse=True)

    def store_sorted_features(self, model_weights_path):
        """
        Store files with sorted lists of features and their associated
        coefficients from each model (for which introspection like this
        can be done, at least).

        :param model_weights_path: path to directory for model weights
                                   files
        :type model_weights_path: str
        :param : 
        :type : 

        :returns: 
        :rtype: 
        """

        makedirs(model_weights_path, exist_ok=True)

        # Generate feature weights files and a README.json providing
        # the parameters corresponding to each set of feature weights
        params_dict = {}
        for (learner_list, learner_name) in zip(self.learner_lists,
                                                self.learner_names):
            # Skip MiniBatchKMeans models
            if learner_name == 'MiniBatchKMeans':
                continue

            for i, learner in enumerate(learner_list):

                # Store the parameter grids to an indexed dictionary
                # so that a key can be output also
                params_dict.setdefault(learner_name, {})
                params_dict[learner_name][i] = learner.get_params()

                # Get dataframe of the features/coefficients
                df = pd.DataFrame(self.get_sorted_features_for_learner(learner))

                if not df.empty:
                    # Generate feature weights report
                    df.to_csv(join(model_weights_path,
                                   self.__model_weights_name_template__
                                   .format(learner_name, i + 1)))
                else:
                    self.logger.error('Could not generate features/feature '
                                      'coefficients dataframe for {0}...'
                                      .format(learner_name))

        # Save parameters file also
        dump(params_dict, open(join(model_weights_path, 'README.json'), 'w'),
             indent=4)

    def learning_round(self) -> None:
        """
        Do learning rounds.

        :rtype: None
        """

        # Get some training data
        train_data = self.get_train_data_iteration()
        samples = len(train_data)

        # Skip round if there are no more training samples to learn
        # from or if the number remaining is less than half the size of
        # the intended number of samples to be used in each round
        if (not samples
            or samples < self.round_size/2):
            return

        self.logger.info('Round {0}...'.format(self.round))
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
        for i, (learner_list, learner_name) in enumerate(zip(self.learner_lists,
                                                             self.learner_names)):
            for j, learner in enumerate(learner_list):
                # If the learner is MiniBatchKMeans, set the batch_size
                # parameter to the number of training samples
                if learner_name == 'MiniBatchKMeans':
                    learner.set_params(batch_size=samples)

                if (learner_name in self.__learners_requiring_classes__
                    and self.round == 1):
                    learner.partial_fit(X_train, y_train, classes=self.classes)
                else:
                    learner.partial_fit(X_train, y_train)

                # Make predictions on the test set, rounding the values
                y_test_preds = np.round(learner.predict(X_test))

                # "Rescale" the values (if necessary), forcing the
                # values should fit within the original scale
                y_test_preds = self.fit_preds_in_scale(y_test_preds)

                # Evaluate the new model, collecting metrics, etc., and
                # then store the round statistics
                stats_dict = self.get_stats(y_test_preds)
                stats_dict.update({self.__games__ if len(self.games) > 1
                                                  else self.__game__:
                                       ', '.join(self.games)
                                           if self.games.difference(ex.VALID_GAMES)
                                           else self.__all_games__,
                                   self.__test_games__:
                                       ', '.join(self.test_games)
                                       if self.test_games.difference(ex.VALID_GAMES)
                                       else self.__all_games__,
                                   self.__learning_round__: int(self.round),
                                   self.__prediction_label__: self.prediction_label,
                                   self.__test_labels_and_preds__:
                                       list(zip(self.y_test, y_test_preds)),
                                   self.__learner__: learner_name,
                                   self.__params__: learner.get_params(),
                                   self.__training_samples__: samples,
                                   self.__non_nlp_features__:
                                       ', '.join(self.non_nlp_features),
                                   self.__no_nlp_features__: self.no_nlp_features})
                if self.bin_ranges:
                    stats_dict.update({self.__bin_ranges__: self.bin_ranges})
                self.learner_param_grid_stats[i][j].append(pd.Series(stats_dict))

        # Increment the round number
        self.round += 1

    def do_learning_rounds(self) -> None:
        """
        Do rounds of learning.

        :rtype: None
        """

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


def main(argv=None):
    parser = ArgumentParser(description='Run incremental learning '
                                        'experiments.',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    _add_arg = parser.add_argument
    _add_arg('--games',
             help='Game(s) to use in experiments; or "all" to use data from '
                  'all games. If --test_games is not specified, then it is '
                  'assumed that the evaluation will be against data from the '
                  'same game(s).',
             type=str,
             required=True)
    _add_arg('--test_games',
             help='Game(s) to use for evaluation data (or "all" for data from'
                  ' all games). Only specify if the value is different from '
                  'that specified via --games.',
             type=str)
    _add_arg('--output_dir',
             help='Directory in which to output data related to the results '
                  'of the conducted experiments.',
             type=str,
             required=True)
    _add_arg('--rounds',
             help='The maximum number of rounds of learning to conduct (the '
                  'number of rounds will necessarily be limited by the amount'
                  ' of training data and the number of samples used per '
                  'round). Use "0" to do as many rounds as possible.',
             type=int,
             default=0)
    _add_arg('--samples_per_round',
             help='The maximum number of training samples to use in each '
                  'round.',
             type=int,
             default=100)
    _add_arg('--test_limit',
             help='Cap to set on the number of test reviews to use for '
                  'evaluation.',
             type=int,
             default=1000)
    _add_arg('--prediction_label',
             help='Label to predict.',
             choices=ex.LABELS,
             default='total_game_hours_bin')
    _add_arg('--non_nlp_features',
             help='Comma-separated list of non-NLP features to combine with '
                  'the NLP features in creating a model. Use "all" to use all'
                  ' available features, "none" to use no non-NLP features. If'
                  ' --only_non_nlp_features is used, NLP features will be '
                  'left out entirely.',
             type=str,
             default='none')
    _add_arg('--only_non_nlp_features',
             help="Don't use any NLP features.",
             action='store_true',
             default=False)
    _add_arg('--learners',
             help='Comma-separated list of learning algorithms to try. Refer '
                  'to list of learners above to find out which abbreviations '
                  'stand for which learners. Set of available learners: {0}. '
                  'Use "all" to include all available learners.'
                  .format(ex.LEARNER_ABBRS_STRING),
             type=str,
             default='all')
    _add_arg('--nbins',
             help='Number of bins to split up the distribution of prediction '
                  'label values into. Use 0 (or don\'t specify) if the values'
                  ' should not be collapsed into bins. Note: Only use this '
                  'option (and --bin_factor below) if the prediction labels '
                  'are numeric.',
             type=int,
             default=0)
    _add_arg('--bin_factor',
             help='Factor by which to multiply the size of each bin. Defaults'
                  ' to 1.0.',
             type=float,
             required=False)
    _add_arg('--obj_func',
             help='Objective function to use in determining which learner/set'
                  ' of parameters resulted in the best performance.',
             choices=ex.OBJ_FUNC_ABBRS_DICT.keys(),
             default='qwk')
    _add_arg('--order_outputs_by',
             help='Order output reports by best last round objective '
                  'performance, best learning round objective performance, or'
                  ' by best objective slope.',
             choices=ORDERINGS,
             default='objective_last_round')
    _add_arg('--evaluate_majority_baseline',
             help='Evaluate the majority baseline model.',
             action='store_true',
             default=True)
    _add_arg('--save_best_features',
             help='Get the best features from each model and write them out '
                  'to files.',
             action='store_true',
             default=False)
    _add_arg('-dbhost', '--mongodb_host',
             help='Host that the MongoDB server is running on.',
             type=str,
             default='localhost')
    _add_arg('--mongodb_port', '-dbport',
             help='Port that the MongoDB server is running on.',
             type=int,
             default=37017)
    _add_arg('-log', '--log_file_path',
             help='Path to feature extraction log file. If no path is '
                  'specified, then a "logs" directory will be created within '
                  'the directory specified via the --output_dir argument.',
             type=str,
             required=False)
    args = parser.parse_args()

    # Command-line arguments and flags
    games = ex.parse_games_string(args.games)
    test_games = ex.parse_games_string(args.test_games
                                       if args.test_games
                                       else args.games)
    rounds = args.rounds
    samples_per_round = args.samples_per_round
    prediction_label = args.prediction_label
    non_nlp_features = ex.parse_non_nlp_features_string(args.non_nlp_features,
                                                        prediction_label)
    only_non_nlp_features = args.only_non_nlp_features
    nbins = args.nbins
    bin_factor = args.bin_factor
    learners = ex.parse_learners_string(args.learners)
    host = args.mongodb_host
    port = args.mongodb_port
    test_limit = args.test_limit
    if not isfile(realpath(args.output_dir)):
        output_dir = realpath(args.output_dir)
    else:
        raise FileExistsError('The specified output destination is the name '
                              'of a currently existing file.')
    obj_func = args.obj_func
    ordering = args.order_outputs_by
    evaluate_majority_baseline = args.evaluate_majority_baseline
    save_best_features = args.save_best_features
    if args.log_file_path:
        if isdir(realpath(args.log_file_path)):
            raise FileExistsError('The specified log file path is the name of'
                                  ' a currently existing directory.')
        else:
            log_file_path = realpath(args.log_file_path)
    else:
        log_file_path = join(output_dir, 'logs', 'learn.log')
    log_dir = dirname(log_file_path)

    # Output results files to output directory
    makedirs(output_dir, exist_ok=True)
    makedirs(log_dir, exist_ok=True)

    # Set up logger
    logger = logging.getLogger('learn')
    logging_debug = logging.DEBUG
    logger.setLevel(logging_debug)
    loginfo = logger.info
    logdebug = logger.debug
    formatter = logging.Formatter(log_format_string)
    sh = logging.StreamHandler()
    sh.setLevel(logging_debug)
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging_debug)
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)

    # Log a bunch of job attributes
    loginfo('Output directory: {0}'.format(output_dir))
    if games == test_games:
        loginfo('Game{0} to train/evaluate models on: {1}'
                .format('s' if len(games) > 1 else '',
                        ', '.join(games) if ex.VALID_GAMES.difference(games)
                                         else 'all games'))
    else:
        loginfo('Game{0} to train models on: {1}'
                .format('s' if len(games) > 1 else '',
                        ', '.join(games) if ex.VALID_GAMES.difference(games)
                                         else 'all games'))
        loginfo('Game{0} to evaluate models against: {1}'
                .format('s' if len(test_games) > 1 else '',
                        ', '.join(test_games)
                        if ex.VALID_GAMES.difference(test_games)
                        else 'all games'))
    loginfo('Maximum number of learning rounds to conduct: {0}'
            .format(rounds if rounds > 0 else "as many as possible"))
    loginfo('Maximum number of training samples to use in each round: {0}'
            .format(samples_per_round))
    loginfo('Prediction label: {0}'.format(prediction_label))
    loginfo('Non-NLP features to use: {0}'
            .format(', '.join(non_nlp_features) if non_nlp_features else 'none'))
    if only_non_nlp_features:
        if not non_nlp_features:
            raise ValueError('No features to train a model on since the '
                             '--only_non_nlp_features flag was used and the '
                             'set of non-NLP features being used is empty.')
        loginfo('Leaving out all NLP features.')
    if nbins == 0:
        if bin_factor:
            raise ValueError('--bin_factor should not be specified if --nbins'
                             ' is not specified or set to 0.')
        bin_ranges = None
    else:
        if (bin_factor
            and bin_factor <= 0):
            raise ValueError('--bin_factor should be set to a positive, '
                             'non-zero value.')
        elif not bin_factor:
            bin_factor = 1.0
        loginfo('Number of bins to split up the distribution of prediction '
                'label values into: {}'.format(nbins))
        loginfo("Factor by which to multiply each succeeding bin's size: {}"
                .format(bin_factor))
    loginfo('Learners: {0}'.format(', '.join([ex.LEARNER_ABBRS_DICT[learner]
                                              for learner in learners])))
    loginfo('Using {0} as the objective function'.format(obj_func))

    # Connect to running Mongo server
    loginfo('MongoDB host: {0}'.format(host))
    loginfo('MongoDB port: {0}'.format(port))
    loginfo('Limiting number of test reviews to {0} or below'
                .format(test_limit))
    db = connect_to_db(host=host, port=port)

    # Check to see if the database has the proper index and, if not,
    # index the database here
    index_name = 'steam_id_number_1'
    if not index_name in db.index_information():
        logdebug('Creating index on the "steam_id_number" key...')
        db.create_index('steam_id_number', ASCENDING)

    if nbins:
        # Get ranges of prediction label distribution bins given the
        # number of bins and the factor by which they should be
        # multiplied as the index increases
        bin_ranges = get_bin_ranges_helper(db, games, prediction_label, nbins,
                                           bin_factor)
    if bin_ranges:
        loginfo('Bin ranges (nbins = {0}, bin_factor = {1}): {2}'
                .format(nbins, bin_factor, bin_ranges))

    # Do learning experiments
    loginfo('Starting incremental learning experiments...')
    experiments = RunExperiments(db,
                                 games,
                                 test_games,
                                 [ex.LEARNER_DICT[learner] for learner in learners],
                                 [ex.find_default_param_grid(learner)
                                 for learner in learners],
                                 samples_per_round,
                                 non_nlp_features,
                                 prediction_label,
                                 obj_func,
                                 logger,
                                 no_nlp_features=only_non_nlp_features,
                                 bin_ranges=bin_ranges,
                                 test_limit=test_limit,
                                 rounds=rounds,
                                 majority_baseline=evaluate_majority_baseline)

    # Generate evaluation reports for the various learner/parameter
    # grid combinations, ranking experiments in terms of their
    # performance with respect to the objective function in the last
    # round of learning, their best performance (in any round), or the
    # slope of their performance as the round increases
    loginfo('Generating reports for the incremental learning runs ordered by '
            '{0}...'.format(ordering))
    experiments.generate_learning_reports(output_dir, ordering)

    # Generate evaluation report for the majority baseline model, if
    # specified
    if evaluate_majority_baseline:
        loginfo('Generating report for the majority baseline model...')
        loginfo('Majority label: {0}'.format(experiments.majority_label))
        experiments.generate_majority_baseline_report(output_dir)

    # Save the best-performing features
    if save_best_features:
        loginfo('Generating feature weights output files...')
        model_weights_dir = join(output_dir, 'model_weights')
        makedirs(model_weights_dir, exist_ok=True)
        experiments.store_sorted_features(model_weights_dir)

    loginfo('Complete.')


if __name__ == '__main__':
    main()
