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
from copy import copy
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
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.linear_model import (Perceptron,
                                  PassiveAggressiveRegressor)

from data import APPID_DICT
from src import log_format_string
from src import experiments as ex
from src.mongodb import connect_to_db
from src import (LABELS,
                 VALID_GAMES,
                 LEARNER_DICT,
                 LABELS_STRING,
                 LEARNER_ABBRS_DICT,
                 OBJ_FUNC_ABBRS_DICT,
                 LEARNER_ABBRS_STRING,
                 OBJ_FUNC_ABBRS_STRING)
from src.datasets import (get_bin,
                          parse_games_string,
                          compute_label_value,
                          validate_bin_ranges,
                          parse_learners_string,
                          get_bin_ranges_helper,
                          find_default_param_grid,
                          parse_non_nlp_features_string)

# Filter out warnings since there will be a lot of
# "UndefinedMetricWarning" warnings when running IncrementalLearning
filterwarnings("ignore")


class RunExperiments:
    """
    Class for conducting sets of incremental learning experiments.
    """

    # Constant strings
    _game = 'game'
    _games = 'games'
    _test_game = 'test_game'
    _test_games = 'test_games'
    _all_games = 'all_games'
    _partition = 'partition'
    _training = 'training'
    _test = 'test'
    _nlp_feats = 'nlp_features'
    _bin_ranges = 'bin_ranges'
    _achieve_prog = 'achievement_progress'
    _steam_id = 'steam_id_number'
    _in_op = '$in'
    _x = 'x'
    _y = 'y'
    _id_string = 'id_string'
    _id = 'id'
    _obj_id = '_id'
    _macro = 'macro'
    _weighted = 'weighted'
    _linear = 'linear'
    _quadratic = 'quadratic'
    _learning_round = 'learning_round'
    _prediction_label = 'prediction_label'
    _test_labels_and_preds = 'test_set_labels/test_set_predictions'
    _non_nlp_features = 'non-NLP features'
    _no_nlp_features = 'no NLP features'
    _transformation = 'transformation'
    _learner = 'learner'
    _params = 'params'
    _training_samples = 'training_samples'
    _r = 'pearson_r'
    _sig = 'significance'
    _prec_macro = 'precision_macro'
    _prec_weighted = 'precision_weighted'
    _f1_macro = 'f1_macro'
    _f1_weighted = 'f1_weighted'
    _acc = 'accuracy'
    _cnfmat = 'confusion_matrix'
    _printable_cnfmat = 'printable_confusion_matrix'
    _uwk = 'uwk'
    _uwk_off_by_one = 'uwk_off_by_one'
    _qwk = 'qwk'
    _qwk_off_by_one = 'qwk_off_by_one'
    _lwk = 'lwk'
    _lwk_off_by_one = 'lwk_off_by_one'
    _majority_label = 'majority_label'
    _majority_baseline_model = 'majority_baseline_model'
    _cnfmat_header = ('confusion_matrix (rounded predictions) '
                      '(row=actual, col=machine, labels={0}):\n')
    _report_name_template = '{0}_{1}_{2}_{3}.csv'
    _available_labels_string = LABELS_STRING
    _labels_list = None

    # Constant values
    _nan = float("NaN")
    _n_features_feature_hashing = 2 ** 18
    _default_cursor_batch_size = 50

    # Constant methods
    _tab_join = '\t'.join
    _cnfmat_row = '{0}{1}\n'.format

    # Available learners, labels, orderings
    _learners_requiring_classes = frozenset({'BernoulliNB', 'MultinomialNB',
                                             'Perceptron'})
    _no_introspection_learners_dict = {'MiniBatchKMeans': MiniBatchKMeans,
         'PassiveAggressiveRegressor': PassiveAggressiveRegressor}
    _learner_names = {MiniBatchKMeans: 'MiniBatchKMeans',
                      BernoulliNB: 'BernoulliNB',
                      MultinomialNB: 'MultinomialNB',
                      Perceptron: 'Perceptron',
                      PassiveAggressiveRegressor: 'PassiveAggressiveRegressor'}
    _orderings = frozenset({'objective_last_round', 'objective_best_round',
                            'objective_slope'})
    _possible_non_nlp_features = set(LABELS)

    def __init__(self, db: collection, games: set, test_games: set, learners,
                 param_grids: dict, samples_per_round: int, non_nlp_features: list,
                 prediction_label: str, objective: str, logger: logging.RootLogger,
                 hashed_features: int = None, no_nlp_features: bool = False,
                 bin_ranges: list = None, lognormal: bool = False,
                 power_transform: float = None, max_test_samples: int = 0,
                 max_rounds: int = 0,
                 majority_baseline: bool = True) -> 'RunExperiments':
        """
        Initialize object.

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
        :param samples_per_round: number of training documents to
                                  extract in each round
        :type samples_per_round: int
        :param non_nlp_features: list of non-NLP features to add into
                                 the feature dictionaries 
        :type non_nlp_features: list
        :param prediction_label: feature to predict
        :type prediction_label: str
        :param objective: objective function to use in ranking the runs
        :type objective: str
        :param logger: logger instance
        :type logger: logging.RootLogger
        :param hashed_features: use FeatureHasher in place of
                                DictVectorizer and use the given number
                                of features (must be positive number or
                                0, which will set it to the default
                                number of features for feature hashing)
        :type hashed_features: int
        :param no_nlp_features: leave out NLP features
        :type no_nlp_features: bool
        :param bin_ranges: list of tuples representing the maximum and
                           minimum values corresponding to bins (for
                           splitting up the distribution of prediction
                           label values)
        :type bin_ranges: list or None
        :param lognormal: transform raw label values using `ln` (default:
                          False)
        :type lognormal: bool
        :param power_transform: power by which to transform raw label
                                values (default: None)
        :type power_transform: float or None
        :param max_test_samples: limit for the number of test samples
                                 (defaults to 0 for no limit)
        :type max_test_samples: int
        :param max_rounds: number of rounds of learning (0 for as many
                           as possible)
        :type max_rounds: int
        :param majority_baseline: evaluate a majority baseline model
        :type majority_baseline: bool

        :returns: instance of RunExperiments class
        :rtype: RunExperiments

        :raises ValueError: if the input parameters result in conflicts
                            or are invalid
        """

        self.logger = logger

        # Make sure parameters make sense/are valid
        if samples_per_round < 1:
            raise ValueError('The samples_per_round parameter should have a '
                             'positive value.')
        if prediction_label in non_nlp_features:
            raise ValueError('The prediction_label parameter ({0}) cannot '
                             'also be in the list of non-NLP features to use '
                             'in the model:\n\n{1}\n.'
                             .format(prediction_label,
                                     ', '.join(non_nlp_features)))
        if any(not feat in self._possible_non_nlp_features for feat
               in non_nlp_features):
            raise ValueError('All non-NLP features must be included in the '
                             'list of available non-NLP features: {0}.'
                             .format(self._available_labels_string))
        if not prediction_label in self._possible_non_nlp_features:
            raise ValueError('The prediction label must be in the set of '
                             'features that can be extracted/used, i.e.: {0}.'
                             .format(self._available_labels_string))
        if not all(games_.issubset(VALID_GAMES) for games_ in [games, test_games]):
            raise ValueError('Unrecognized game(s)/test game(s): {0}. The '
                             'games must be in the following list of '
                             'available games: {1}.'
                             .format(', '.join(games.union(test_games)),
                                     ', '.join(APPID_DICT)))
        if hashed_features != None:
            if hashed_features < 0:
                raise ValueError('Cannot use non-positive value, {0}, for the'
                                 ' "hashed_features" parameter.'
                                 .format(hashed_features))
            else:
                if hashed_features == 0:
                    hashed_features = self._n_features_feature_hashing
        if lognormal and power_transform:
            raise ValueError('Both "lognormal" and "power_transform" were '
                             'specified simultaneously.')
        if bin_ranges:
            try:
                validate_bin_ranges(bin_ranges)
            except ValueError as e:
                logerr('"bin_ranges" failed validation: {0}'.format(bin_ranges))
                raise e

        # Incremental learning-related attributes
        self.samples_per_round = samples_per_round
        self.max_rounds = max_rounds
        self.round = 1
        self.NO_MORE_TRAINING_DATA = False
        self.batch_size = (samples_per_round
                           if samples_per_round < self._default_cursor_batch_size
                           else self._default_cursor_batch_size)

        # MongoDB database
        self.db = db

        # Games
        self.games = games
        if not self.games:
            raise ValueError('The set of games must be greater than zero!')
        self._games_string = ', '.join(self.games)

        # Templates for report file names
        self._report_name_template = ('{0}_{1}_{2}_{3}.csv'
                                      .format(self._games_string,
                                              '{0}', '{1}', '{2}'))
        self._stats_name_template = (self._report_name_template
                                     .format('{0}', 'stats', '{1}'))
        self._model_weights_name_template = (self._report_name_template
                                             .format('{0}', 'model_weights', '{1}'))
        if majority_baseline:
            self._majority_baseline_report_name = \
                ('{0}_majority_baseline_model_stats.csv'
                 .format(self._games_string))
        self.test_games = test_games
        if not self.test_games:
            raise ValueError('The set of games must be greater than zero!')
        self._test_games_string = ', '.join(self.test_games)
        self.bin_ranges = bin_ranges
        self.lognormal = lognormal
        self.power_transform = power_transform
        if self.lognormal or self.power_transform:
            self._transformation_string = ('ln' if self.lognormal
                                           else 'x**{0}'.format(power_transform))
        else:
            self._transformation_string = 'None'

        # Objective function
        self.objective = objective
        if not self.objective in OBJ_FUNC_ABBRS_DICT:
            raise ValueError('Unrecognized objective function used: {0}. '
                             'These are the available objective functions: {1}.'
                             .format(self.objective, OBJ_FUNC_ABBRS_STRING))

        # Learner-related variables
        self.vec = None
        self.hashed_features = hashed_features
        self.param_grids = [list(ParameterGrid(param_grid)) for param_grid
                            in param_grids]
        self.learner_names = [self._learner_names[learner] for learner in learners]
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

        # Data- and database-related variables
        self.training_cursor = None
        self.test_cursor = None
        self.max_test_samples = max_test_samples
        self.logger.info('Setting up MongoDB cursor for training data...')
        self.sorting_args = [(self._steam_id, ASCENDING)]
        self.projection = None
        self.make_train_cursor()
        self.logger.info('Extracting evaluation dataset...')
        self.test_data = self.get_test_data()
        self.test_ids = [data_[self._id] for data_ in self.test_data]
        self.test_feature_dicts = [data_[self._x] for data_ in self.test_data]
        self.y_test = np.array([data_[self._y] for data_ in self.test_data])
        self.classes = np.unique(self.y_test)
        self._labels_list = [str(cls) for cls in self.classes]
        self.logger.info('Prediction label classes: {0}'
                         .format(', '.join(self._labels_list)))

        # Useful constants for use in make_printable_confusion_matrix
        self.cnfmat_desc = self._cnfmat_row(self._cnfmat_header.format(self.classes),
                                            self._tab_join([''] + self._labels_list))

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

    def make_train_cursor(self) -> None:
        """
        Make cursor for the training sets.

        :rtype: None
        """

        self.logger.debug('Batch size of MongoDB training cursor: {0}'
                          .format(self.batch_size))

        # Leave out the '_id' value and the 'nlp_features' value if
        # `self.no_nlp_features` is true
        self.projection = {self._obj_id: 0}
        if self.no_nlp_features:
            self.projection.update({self._nlp_feats: 0})

        # Make training data cursor
        games = list(self.games)
        if len(games) == 1:
            train_query = {self._game: games[0],
                           self._partition: self._training}
        else:
            train_query = {self._game: {self._in_op: games},
                           self._partition: self._training}
        self.training_cursor = (self.db
                                .find(train_query, self.projection, timeout=False)
                                .sort(self.sorting_args))
        self.training_cursor.batch_size = self.batch_size

    def get_all_features(self, review_doc: dict) -> dict:
        """
        Get all the features in a review document and put them together
        in a dictionary. If `self.no_nlp_features` is true, leave out
        NLP features. If `bin_ranges` is specified, convert the value
        of the prediction label to the bin index.

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
                     in BSON.decode(_get(self._nlp_feats)).items()
                     if val and val != self._nan})

        # Add in the non-NLP features (except for those that may be in
        # the 'achievement_progress' sub-dictionary of the review
        # dictionary)
        _update({feat: val for feat, val in review_doc.items()
                 if (feat in self._possible_non_nlp_features
                     and val and val != self._nan)})

        # Add in the features that may be in the 'achievement_progress'
        # sub-dictionary of the review document
        _update({feat: val for feat, val
                 in _get(self._achieve_prog, dict()).items()
                 if (feat in self._possible_non_nlp_features
                     and val and val != self._nan)})

        # Return None if the prediction label isn't present
        if not features.get(self.prediction_label):
            return

        # Add in the 'id_string' value just to make it easier to
        # process the results of this function
        _update({self._id_string: _get(self._id_string)})
        return features

    def get_data(self, review_doc: dict) -> dict:
        """
        Collect data from a MongoDB review document and return it in
        format needed for DictVectorizer.

        :param review_doc: document from the MongoDB reviews collection
        :type review_doc: dict

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

        # Get prediction label, apply transformations, and remove it
        # from the feature dictionary
        y_value = self.transform_value(feature_dict[self.prediction_label])
        del feature_dict[self.prediction_label]

        # Get ID and remove from feature dictionary
        id_string = feature_dict[self._id_string]
        del feature_dict[self._id_string]

        # Only keep the non-NLP features that are supposed to be kept,
        # if any
        for feat in self._possible_non_nlp_features:
            if (not feat in self.non_nlp_features
                and feature_dict.get(feat, None) != None):
                del feature_dict[feat]

        # If, after taking out the prediction label and the ID, there
        # are no remaining features, return None
        if not feature_dict:
            return

        # Return dictionary of prediction label value, ID string, and
        # features
        return {self._y: y_value, self._id: id_string, self._x: feature_dict}

    def get_train_data_iteration(self) -> list:
        """
        Get a list of training data dictionaries to use in model
        training.

        :returns: list of sample dictionaries
        :rtype: list
        """

        data = []
        i = 0
        while i < self.samples_per_round:
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

        # Generate the base query
        games = list(self.test_games)
        if len(games) == 1:
            test_query = {self._game: games[0], self._partition: self._test}
        else:
            test_query = {self._game: {self._in_op: games},
                          self._partition: self._test}

        data = []
        j = 0
        for id_string \
            in ex.evenly_distribute_samples(self.db, self.prediction_label,
                                            games, bin_ranges=self.bin_ranges,
                                            lognormal=self.lognormal,
                                            power_transform=self.power_transform):
            # Get a review document from the Mongo database
            _test_query = copy(test_query)
            _test_query[self._id_string] = id_string

            # Get features, prediction label, and ID in a new
            # dictionary and append to list of data samples
            sample = self.get_data(next(self.db
                                        .find(_test_query, self.projection,
                                              timeout=False)))
            if sample:
                data.append(sample)
                j += 1

            if j == self.max_test_samples:
                break

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

        :returns: None
        :rtype: None
        """

        stats_dict = self.get_stats(self.get_majority_baseline())
        stats_dict.update({self._test_games:
                               ', '.join(self.test_games)
                               if VALID_GAMES.difference(self.test_games)
                               else self._all_games,
                           self._prediction_label: self.prediction_label,
                           self._majority_label: self.majority_label,
                           self._learner: self._majority_baseline_model,
                           self._transformation: self._transformation_string})
        if self.bin_ranges:
            stats_dict.update({self._bin_ranges: self.bin_ranges})
        self.majority_baseline_stats = pd.DataFrame([pd.Series(stats_dict)])

    def generate_majority_baseline_report(self, output_path: str) -> None:
        """
        Generate a CSV file reporting on the performance of the
        majority baseline model.

        :param output_path: path to destination directory
        :type: str

        :returns: None
        :rtype: None
        """

        (self.majority_baseline_stats
         .to_csv(join(output_path, self._majority_baseline_report_name), index=False))

    def generate_learning_reports(self, output_path: str,
                                  ordering: str = 'objective_last_round') -> None:
        """
        Generate experimental reports for each run represented in the
        lists of input dataframes.

        The output files will have indices in their names, which simply
        correspond to the sequence in which they occur in the list of
        input dataframes.

        :param output_path: path to destination directory
        :type output_path: str
        :param ordering: ordering type for ranking the reports (see
                         `RunExperiments._orderings`)
        :type ordering: str

        :returns: None
        :rtype: None
        """

        # Rank the experiments by the given ordering type
        try:
            dfs = self.rank_experiments_by_objective(ordering=ordering)
        except ValueError:
            raise e

        for i, df in enumerate(dfs):
            learner_name = df[self._learner].iloc[0]
            df.to_csv(join(output_path,
                           self._stats_name_template.format(learner_name, i + 1)),
                      index=False)

    def transform_value(self, val: float) -> int:
        """
        Convert the value to the index of the bin in which it resides.

        :param val: raw value
        :type val: float

        :returns: index of bin containing value
        :rtype: int
        """

        # Apply transformations (multiply by 100 if percentage and/or
        # natural log/power transformation) if specified
        val = compute_label_value(val, self.prediction_label,
                                  lognormal=self.lognormal,
                                  power_transform=self.power_transform)

        # Convert value to bin-transformed value
        if not self.bin_ranges:
            return val
        return get_bin(self.bin_ranges, val)

    def make_printable_confusion_matrix(self, y_preds: np.array) -> tuple:
        """
        Produce a printable confusion matrix to use in the evaluation
        report.

        :param y_preds: array of predicted labels
        :type y_preds: np.array

        :returns: tuple consisting of a confusion matrix string and a
                  confusion matrix multi-dimensional array
        :rtype: tuple
        """

        cnfmat = confusion_matrix(self.y_test, np.round(y_preds),
                                  labels=self.classes).tolist()
        res = str(self.cnfmat_desc)
        for row, label in zip(cnfmat, self.classes):
            row = self._tab_join([str(x) for x in [label] + row])
            res = self._cnfmat_row(res, row)

        return res, cnfmat

    def fit_preds_in_scale(self, y_preds: np.array) -> np.array:
        """
        Force values at either end of the scale to fit within the scale
        by adding to or truncating the values.

        :param y_preds: array of predicted labels
        :type y_preds: np.array

        :returns: array of predicted labels
        :rtype: np.array
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

    def get_stats(self, y_preds: np.array) -> dict:
        """
        Get some statistics about the model's performance on the test
        set.

        :param y_preds: array of predicted labels
        :type y_preds: np.array

        :returns: statistics dictionary
        :rtype: dict
        """

        # Get Pearson r and significance
        r, sig = pearsonr(self.y_test, y_preds)

        # Get confusion matrix (both the np.ndarray and the printable
        # one)
        printable_cnfmat, cnfmat = self.make_printable_confusion_matrix(y_preds)

        return {self._r: r,
                self._sig: sig,
                self._prec_macro: precision_score(self.y_test, y_preds,
                                                  labels=self.classes,
                                                  average=self._macro),
                self._prec_weighted:
                    precision_score(self.y_test, y_preds, labels=self.classes,
                                    average=self._weighted),
                self._f1_macro: f1_score(self.y_test, y_preds,
                                         labels=self.classes,
                                         average=self._macro),
                self._f1_weighted: f1_score(self.y_test, y_preds,
                                            labels=self.classes,
                                            average=self._weighted),
                self._acc: accuracy_score(self.y_test, y_preds, normalize=True),
                self._cnfmat: cnfmat,
                self._printable_cnfmat: printable_cnfmat,
                self._uwk: kappa(self.y_test, y_preds),
                self._uwk_off_by_one: kappa(self.y_test, y_preds,
                                            allow_off_by_one=True),
                self._qwk: kappa(self.y_test, y_preds,
                                 weights=self._quadratic),
                self._qwk_off_by_one: kappa(self.y_test, y_preds,
                                            weights=self._quadratic,
                                            allow_off_by_one=True),
                self._lwk: kappa(self.y_test, y_preds, weights=self._linear),
                self._lwk_off_by_one: kappa(self.y_test, y_preds,
                                            weights=self._linear,
                                            allow_off_by_one=True)}

    def rank_experiments_by_objective(self, ordering: str) -> list:
        """
        Rank the experiments in relation to their performance in the
        objective function.

        :param ordering: ordering type (see
                         `RunExperiments._orderings`)
        :type ordering: str

        :returns: list of dataframes
        :rtype: list
        """

        if not ordering in self._orderings:
            raise ValueError('ordering parameter not in the set of possible '
                             'orderings: {0}'.format(', '.join(self._orderings)))

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
                    regression = linregress(stats_df[self._learning_round],
                                            stats_df[self.objective])
                    performances.append(regression.slope)

        # Sort dataframes on ordering value and return
        return [df[1] for df
                in sorted(zip(performances, dfs), key=lambda x: x[0], reverse=True)]

    def get_sorted_features_for_learner(self, learner,
                                        filter_zero_features: bool = True) -> list:
        """
        Get the best-performing features in a learner (excluding
        MiniBatchKMeans).

        :param learner: learner (can not be of type MiniBatchKMeans or
                        PassiveAggressiveRegressor, among others)
        :type learner: learner instance
        :param filter_zero_features: filter out features with
                                     zero-valued coefficients
        :type filter_zero_features: bool

        :returns: list of sorted features (in dictionaries)
        :rtype: list

        :raises: ValueError
        """

        # Raise exception if learner class is not supported
        if any(issubclass(type(learner), cls) for cls
               in self._no_introspection_learners_dict.values()):
            raise ValueError('Can not get feature weights for learners of '
                             'type {0}'.format(type(learner)))

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
                return []

            # Append feature coefficient tuple to list of tuples
            feature_coefs.append(tuple(list(chain([feat],
                                                  zip(self.classes,
                                                      coef_indices)))))

        # Unpack tuples of features and label/coefficient tuples into
        # one long list of feature/label/coefficient values, sort, and
        # filter out any tuples with zero weight
        features = []
        for i in range(1, len(self.classes) + 1):
            features.extend([dict(feature=coefs[0], label=coefs[i][0],
                                  weight=coefs[i][1])
                             for coefs in feature_coefs if coefs[i][1]])

        return sorted(features, key=lambda x: abs(x['weight']), reverse=True)

    def store_sorted_features(self, model_weights_path: str) -> None:
        """
        Store files with sorted lists of features and their associated
        coefficients from each model (for which introspection like this
        can be done, at least).

        :param model_weights_path: path to directory for model weights
                                   files
        :type model_weights_path: str

        :returns: None
        :rtype: None
        """

        makedirs(model_weights_path, exist_ok=True)

        # Generate feature weights files and a README.json providing
        # the parameters corresponding to each set of feature weights
        params_dict = {}
        for (learner_list, learner_name) in zip(self.learner_lists,
                                                self.learner_names):
            # Skip MiniBatchKMeans/PassiveAggressiveRegressor models
            if learner_name in self._no_introspection_learners_dict:
                continue

            for i, learner in enumerate(learner_list):

                # Get dataframe of the features/coefficients
                sorted_features = self.get_sorted_features_for_learner(learner)

                if sorted_features:
                    # Generate feature weights report
                    (pd.DataFrame(sorted_features)
                     .to_csv(join(model_weights_path,
                                  self._model_weights_name_template
                                  .format(learner_name, i + 1)),
                             index=False))

                    # Store the parameter grids to an indexed
                    # dictionary so that a key can be output also
                    params_dict.setdefault(learner_name, {})
                    params_dict[learner_name][i] = learner.get_params()
                else:
                    self.logger.error('Could not generate features/feature '
                                      'coefficients dataframe for {0}...'
                                      .format(learner_name))

        # Save parameters file also
        dump(params_dict,
             open(join(model_weights_path, 'model_params_readme.json'), 'w'),
             indent=4)

    def learning_round(self) -> None:
        """
        Do learning rounds.

        :returns: None
        :rtype: None
        """

        # Get some training data
        train_data = self.get_train_data_iteration()
        samples = len(train_data)

        # Skip round if there are no more training samples to learn
        # from or if the number remaining is less than half the size of
        # the intended number of samples to be used in each round
        if (not samples
            or samples < self.samples_per_round/2):
            return

        self.logger.info('Round {0}...'.format(self.round))
        train_ids = np.array([data_[self._id] for data_ in train_data])
        y_train = np.array([data_[self._y] for data_ in train_data])
        train_feature_dicts = [data_[self._x] for data_ in train_data]

        # Set `vec` if not already set and fit it it the training
        # features, which will only need to be done the first time
        if self.vec == None:
            if not self.hashed_features:
                self.vec = DictVectorizer(sparse=True)
            else:
                self.vec = FeatureHasher(n_features=self.hashed_features,
                                         non_negative=True)
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

                if (learner_name in self._learners_requiring_classes
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
                stats_dict.update({self._games if len(self.games) > 1
                                   else self._game:
                                       ', '.join(self.games)
                                       if VALID_GAMES.difference(self.games)
                                       else self._all_games,
                                   self._test_games if len(self.test_games) > 1
                                   else self._test_game:
                                       ', '.join(self.test_games)
                                       if VALID_GAMES.difference(self.test_games)
                                       else self._all_games,
                                   self._learning_round: int(self.round),
                                   self._prediction_label: self.prediction_label,
                                   self._test_labels_and_preds:
                                       list(zip(self.y_test, y_test_preds)),
                                   self._learner: learner_name,
                                   self._params: learner.get_params(),
                                   self._training_samples: samples,
                                   self._non_nlp_features:
                                       ', '.join(self.non_nlp_features),
                                   self._no_nlp_features: self.no_nlp_features,
                                   self._transformation: self._transformation_string})
                if self.bin_ranges:
                    stats_dict.update({self._bin_ranges: self.bin_ranges})
                self.learner_param_grid_stats[i][j].append(pd.Series(stats_dict))

        # Increment the round number
        self.round += 1

    def do_learning_rounds(self) -> None:
        """
        Do rounds of learning.

        :returns: None
        :rtype: None
        """

        # If a certain number of rounds has been specified, try to do
        # that many rounds; otherwise, do as many as possible
        if self.max_rounds > 0:
            while self.round <= self.max_rounds:
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
    _add_arg('-g', '--games',
             help='Game(s) to use in experiments; or "all" to use data from '
                  'all games. If --test_games is not specified, then it is '
                  'assumed that the evaluation will be against data from the '
                  'same game(s).',
             type=str,
             required=True)
    _add_arg('-test', '--test_games',
             help='Game(s) to use for evaluation data (or "all" for data from'
                  ' all games). Only specify if the value is different from '
                  'that specified via --games.',
             type=str)
    _add_arg('-out', '--output_dir',
             help='Directory in which to output data related to the results '
                  'of the conducted experiments.',
             type=str,
             required=True)
    _add_arg('-nrounds', '--max_rounds',
             help='The maximum number of rounds of learning to conduct (the '
                  'number of rounds will necessarily be limited by the amount'
                  ' of training data and the number of samples used per '
                  'round). Use "0" to do as many rounds as possible.',
             type=int,
             default=0)
    _add_arg('-ntrain', '--max_samples_per_round',
             help='The maximum number of training samples to use in each '
                  'round.',
             type=int,
             default=100)
    _add_arg('-ntest', '--max_test_samples',
             help='Cap to set on the number of test samples to use for '
                  'evaluation.',
             type=int,
             default=1000)
    _add_arg('-label', '--prediction_label',
             help='Label to predict.',
             choices=LABELS,
             default='total_game_hours_bin')
    _add_arg('-non_nlp', '--non_nlp_features',
             help='Comma-separated list of non-NLP features to combine with '
                  'the NLP features in creating a model. Use "all" to use all'
                  ' available features, "none" to use no non-NLP features. If'
                  ' --only_non_nlp_features is used, NLP features will be '
                  'left out entirely.',
             type=str,
             default='none')
    _add_arg('-only_non_nlp', '--only_non_nlp_features',
             help="Don't use any NLP features.",
             action='store_true',
             default=False)
    _add_arg('-l', '--learners',
             help='Comma-separated list of learning algorithms to try. Refer '
                  'to list of learners above to find out which abbreviations '
                  'stand for which learners. Set of available learners: {0}. '
                  'Use "all" to include all available learners.'
                  .format(LEARNER_ABBRS_STRING),
             type=str,
             default='all')
    _add_arg('-bin', '--nbins',
             help='Number of bins to split up the distribution of prediction '
                  'label values into. Use 0 (or don\'t specify) if the values'
                  ' should not be collapsed into bins. Note: Only use this '
                  'option (and --bin_factor below) if the prediction labels '
                  'are numeric.',
             type=int,
             default=0)
    _add_arg('-factor', '--bin_factor',
             help='Factor by which to multiply the size of each bin. Defaults'
                  ' to 1.0 if --nbins is specified.',
             type=float,
             required=False)
    _add_arg('--lognormal',
             help='Transform raw label values with log before doing anything '
                  'else, whether it be binning the values or learning from '
                  'them.',
             action='store_true',
             default=False)
    _add_arg('--power_transform',
             help='Transform raw label values via `x**power` where `power` is'
                  ' the value specified and `x` is the raw label value before'
                  ' doing anything else, whether it be binning the values or '
                  'learning from them.',
             type=float,
             default=None)
    _add_arg('-feature_hasher', '--use_feature_hasher',
             help='Use FeatureHasher to be more memory-efficient.',
             action='store_true',
             default=False)
    _add_arg('-obj', '--obj_func',
             help='Objective function to use in determining which learner/set'
                  ' of parameters resulted in the best performance.',
             choices=OBJ_FUNC_ABBRS_DICT.keys(),
             default='qwk')
    _add_arg('-order_by', '--order_outputs_by',
             help='Order output reports by best last round objective '
                  'performance, best learning round objective performance, or'
                  ' by best objective slope.',
             choices=RunExperiments._orderings,
             default='objective_last_round')
    _add_arg('-baseline', '--evaluate_majority_baseline',
             help='Evaluate the majority baseline model.',
             action='store_true',
             default=True)
    _add_arg('-save_best', '--save_best_features',
             help='Get the best features from each model and write them out '
                  'to files.',
             action='store_true',
             default=False)
    _add_arg('-dbhost', '--mongodb_host',
             help='Host that the MongoDB server is running on.',
             type=str,
             default='localhost')
    _add_arg('-dbport', '--mongodb_port',
             help='Port that the MongoDB server is running on.',
             type=int,
             default=37017)
    _add_arg('-log', '--log_file_path',
             help='Path to feature extraction log file. If no path is '
                  'specified, then a "logs" directory will be created within '
                  'the directory specified via the --output_dir argument and '
                  'a log will automatically be stored.',
             type=str,
             required=False)
    args = parser.parse_args()

    # Command-line arguments and flags
    games = parse_games_string(args.games)
    test_games = parse_games_string(args.test_games) if args.test_games else games
    max_rounds = args.max_rounds
    max_samples_per_round = args.max_samples_per_round
    prediction_label = args.prediction_label
    non_nlp_features = parse_non_nlp_features_string(args.non_nlp_features,
                                                     prediction_label)
    only_non_nlp_features = args.only_non_nlp_features
    nbins = args.nbins
    bin_factor = args.bin_factor
    lognormal = args.lognormal
    power_transform = args.power_transform
    feature_hashing = args.use_feature_hasher
    learners = parse_learners_string(args.learners)
    host = args.mongodb_host
    port = args.mongodb_port
    max_test_samples = args.max_test_samples
    obj_func = args.obj_func
    ordering = args.order_outputs_by
    evaluate_majority_baseline = args.evaluate_majority_baseline
    save_best_features = args.save_best_features

    # Validate the input arguments
    if not isfile(realpath(args.output_dir)):
        output_dir = realpath(args.output_dir)
    else:
        raise FileExistsError('The specified output destination is the name '
                              'of a currently existing file.')
    if save_best_features:
        if learners.issubset(RunExperiments._no_introspection_learners_dict):
            loginfo('The specified set of learners do not work with the '
                    'current way of extracting features from models and, '
                    'thus, -save_best/--save_best_features, will be ignored.')
            save_best_features = False
        if feature_hashing:
            raise ValueError('The --save_best_features/-save_best option '
                             'cannot be used in conjunction with the '
                             '--use_feature_hasher/-feature_hasher option. '
                             'Exiting.')
    if args.log_file_path:
        if isdir(realpath(args.log_file_path)):
            raise FileExistsError('The specified log file path is the name of'
                                  ' a currently existing directory.')
        else:
            log_file_path = realpath(args.log_file_path)
    else:
        log_file_path = join(output_dir, 'logs', 'learn.log')
    log_dir = dirname(log_file_path)
    if lognormal and power_transform:
        raise ValueError('Both "lognormal" and "power_transform" were '
                         'specified simultaneously.')

    # Output results files to output directory
    makedirs(output_dir, exist_ok=True)
    makedirs(log_dir, exist_ok=True)

    # Set up logger
    logger = logging.getLogger('learn')
    logging_debug = logging.DEBUG
    logger.setLevel(logging_debug)
    loginfo = logger.info
    logerr = logger.error
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
                        ', '.join(games) if VALID_GAMES.difference(games)
                        else 'all games'))
    else:
        loginfo('Game{0} to train models on: {1}'
                .format('s' if len(games) > 1 else '',
                        ', '.join(games) if VALID_GAMES.difference(games)
                        else 'all games'))
        loginfo('Game{0} to evaluate models against: {1}'
                .format('s' if len(test_games) > 1 else '',
                        ', '.join(test_games)
                        if VALID_GAMES.difference(test_games)
                        else 'all games'))
    loginfo('Maximum number of learning rounds to conduct: {0}'
            .format(max_rounds if max_rounds > 0 else "as many as possible"))
    loginfo('Maximum number of training samples to use in each round: {0}'
            .format(max_samples_per_round))
    loginfo('Prediction label: {0}'.format(prediction_label))
    loginfo('Lognormal transformation: {0}'.format(lognormal))
    loginfo('Power transformation: {0}'.format(power_transform))
    loginfo('Non-NLP features to use: {0}'
            .format(', '.join(non_nlp_features) if non_nlp_features else 'none'))
    if only_non_nlp_features:
        if not non_nlp_features:
            raise ValueError('No features to train a model on since the '
                             '--only_non_nlp_features flag was used and the '
                             'set of non-NLP features being used is empty.')
        loginfo('Leaving out all NLP features')
    if nbins == 0:
        if bin_factor:
            raise ValueError('--bin_factor should not be specified if --nbins'
                             ' is not specified or set to 0.')
        bin_ranges = None
    else:
        if bin_factor and bin_factor <= 0:
            raise ValueError('--bin_factor should be set to a positive, '
                             'non-zero value.')
        elif not bin_factor:
            bin_factor = 1.0
        loginfo('Number of bins to split up the distribution of prediction '
                'label values into: {}'.format(nbins))
        loginfo("Factor by which to multiply each succeeding bin's size: {}"
                .format(bin_factor))
    if feature_hashing:
        loginfo('Using feature hashing to increase memory efficiency')
    loginfo('Learners: {0}'.format(', '.join([LEARNER_ABBRS_DICT[learner]
                                              for learner in learners])))
    loginfo('Using {0} as the objective function'.format(obj_func))

    # Connect to running Mongo server
    loginfo('MongoDB host: {0}'.format(host))
    loginfo('MongoDB port: {0}'.format(port))
    loginfo('Limiting number of test reviews to {0} or below'
            .format(max_test_samples))
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
                                           bin_factor, lognormal=lognormal,
                                           power_transform=power_transform)
        if lognormal or power_transform:
            transformation = ('lognormal' if lognormal
                              else 'x**{0}'.format(power_transform))
        loginfo('Bin ranges (nbins = {0}, bin_factor = {1}, {2}): {3}'
                .format(nbins, bin_factor,
                        '{0} transformation'.format(transformation),
                        bin_ranges))

    # Do learning experiments
    loginfo('Starting incremental learning experiments...')
    try:
        experiments = RunExperiments(db,
                                     games,
                                     test_games,
                                     [LEARNER_DICT[learner] for learner in learners],
                                     [find_default_param_grid(learner) for learner in learners],
                                     max_samples_per_round,
                                     non_nlp_features,
                                     prediction_label,
                                     obj_func,
                                     logger,
                                     hashed_features=0 if feature_hashing else None,
                                     no_nlp_features=only_non_nlp_features,
                                     bin_ranges=bin_ranges,
                                     lognormal=lognormal,
                                     power_transform=power_transform,
                                     max_test_samples=max_test_samples,
                                     max_rounds=max_rounds,
                                     majority_baseline=evaluate_majority_baseline)
    except ValueError as e:
        logerr('Encountered a ValueError while instantiating the '
               '`RunExperiments` object: {0}'.format(e))
        raise e

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
        loginfo('Generating feature coefficient output files for each model '
                '(after all learning rounds)...')
        model_weights_dir = join(output_dir, 'model_weights')
        makedirs(model_weights_dir, exist_ok=True)
        experiments.store_sorted_features(model_weights_dir)

    loginfo('Complete.')


if __name__ == '__main__':
    main()
