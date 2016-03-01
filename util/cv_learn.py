"""
:author: Matt Mulholland (mulhodm@gmail.com)
:date: 10/14/2015

Command-line utility utilizing the RunCVExperiments class, which enables
one to run cross-validation experiments incrementally with a number of
different machine learning algorithms and parameter customizations, etc.
"""
import logging
from copy import copy
from json import dump
from os import (unlink,
                makedirs)
from shutil import rmtree
from itertools import chain
from tempfile import mkdtemp
from os.path import (join,
                     isdir,
                     isfile,
                     exists,
                     dirname,
                     realpath)
from warnings import filterwarnings

from tables import (Atom,
                    Filters,
                    open_file)
import numpy as np
import pandas as pd
from typing import (Any,
                    Dict,
                    List,
                    Union,
                    Optional,
                    Iterable)
from pymongo import ASCENDING
from sklearn.externals import joblib
from sklearn.metrics import make_scorer
from schema import (Or,
                    And,
                    Schema,
                    SchemaError,
                    Optional as Default)
from pymongo.collection import Collection
from sklearn.cluster import MiniBatchKMeans
from pymongo.errors import ConnectionFailure
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import (BernoulliNB,
                                 MultinomialNB)
from skll.metrics import (kappa,
                          pearson,
                          spearman,
                          kendall_tau,
                          f1_score_least_frequent)
from argparse import (ArgumentParser,
                      ArgumentDefaultsHelpFormatter)
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction import (FeatureHasher,
                                        DictVectorizer)
from sklearn.linear_model import (Perceptron,
                                  PassiveAggressiveRegressor)

from src.mongodb import connect_to_db
from src import (LABELS,
                 Scorer,
                 Learner,
                 Numeric,
                 BinRanges,
                 ParamGrid,
                 formatter,
                 Vectorizer,
                 VALID_GAMES,
                 LEARNER_DICT,
                 LABELS_STRING,
                 experiments as ex,
                 LEARNER_DICT_KEYS,
                 parse_games_string,
                 LEARNER_ABBRS_DICT,
                 OBJ_FUNC_ABBRS_DICT,
                 LEARNER_ABBRS_STRING,
                 OBJ_FUNC_ABBRS_STRING,
                 parse_learners_string,
                 find_default_param_grid,
                 parse_non_nlp_features_string)
from src.datasets import (validate_bin_ranges,
                          get_bin_ranges_helper)

# Filter out warnings since there will be a lot of
# "UndefinedMetricWarning" warnings when running `RunCVExperiments`
filterwarnings("ignore")

# Set up logger
logger = logging.getLogger('util.cv_learn')
logging_debug = logging.DEBUG
logger.setLevel(logging_debug)
loginfo = logger.info
logerr = logger.error
logdebug = logger.debug
sh = logging.StreamHandler()
sh.setLevel(logging_debug)
sh.setFormatter(formatter)
logger.addHandler(sh)


class CVConfig(object):
    """
    Class for representing a set of configuration options for use with
    the `RunCVExperiments` class.
    """

    # Default value to use for the `hashed_features` parameter if 0 is
    # passed in.
    _n_features_feature_hashing = 2 ** 18

    def __init__(self,
                 db: Collection,
                 games: set,
                 learners: List[str],
                 param_grids: List[ParamGrid],
                 training_rounds: int,
                 training_samples_per_round: int,
                 grid_search_samples_per_fold: int,
                 non_nlp_features: set,
                 prediction_label: str,
                 output_path: str,
                 objective: str = None,
                 data_sampling: str = 'even',
                 grid_search_folds: int = 5,
                 hashed_features: Optional[int] = None,
                 nlp_features: bool = True,
                 bin_ranges: Optional[BinRanges] = None,
                 lognormal: bool = False,
                 power_transform: Optional[float] = None,
                 majority_baseline: bool = True,
                 rescale: bool = True,
                 n_jobs: int = 1) -> 'CVConfig':
        """
        Initialize object.

        :param db: MongoDB database collection object
        :type db: Collection
        :param games: set of games to use for training models
        :type games: set
        :param learners: list of abbreviated names corresponding to
                         the available learning algorithms (see
                         `src.LEARNER_ABBRS_DICT`, etc.)
        :type learners: list
        :param param_grids: list of lists of dictionaries of parameters
                            mapped to lists of values (must be aligned
                            with list of learners)
        :type param_grids: list
        :param training_rounds: number of training rounds to do (in
                                addition to the grid search round)
        :type training_rounds: int
        :param training_samples_per_round: number of training samples
                                           to use in each training round
        :type training_samples_per_round: int
        :param grid_search_samples_per_fold: number of samples to use
                                             for each grid search fold
        :type grid_search_samples_per_fold: int
        :param non_nlp_features: set of non-NLP features to add into the
                                 feature dictionaries 
        :type non_nlp_features: set
        :param prediction_label: feature to predict
        :type prediction_label: str
        :param objective: objective function to use in ranking the runs;
                          if left unspecified, the objective will be
                          decided in `GridSearchCV` and will be either
                          accuracy for classification or r2 for
                          regression
        :param output_path: path for output reports, etc.
        :type output_path: str
        :type objective: str or None
        :param data_sampling: how the data should be sampled (i.e.,
                              either 'even' or 'stratified')
        :type data_sampling: str
        :param grid_search_folds: number of grid search folds to use
                                  (default: 5)
        :type grid_search_folds: int
        :param hashed_features: use FeatureHasher in place of
                                DictVectorizer and use the given number
                                of features (must be positive number or
                                0, which will set it to the default
                                number of features for feature hashing)
        :type hashed_features: int
        :param nlp_features: include NLP features (default: True)
        :type nlp_features: bool
        :param bin_ranges: list of tuples representing the maximum and
                           minimum values corresponding to bins (for
                           splitting up the distribution of prediction
                           label values)
        :type bin_ranges: list or None
        :param lognormal: transform raw label values using `ln` (default:
                          False)
        :type lognormal: bool
        :param power_transform: power by which to transform raw label
                                values (default: False)
        :type power_transform: float or None
        :param majority_baseline: evaluate a majority baseline model
        :type majority_baseline: bool
        :param rescale: whether or not to rescale the predicted values
                        based on the input value distribution (defaults
                        to True, but set to False if this is a
                        classification experiment)
        :type rescale: bool
        :param njobs: value of `n_jobs` parameter, which is passed into
                      the learners (where applicable)
        :type n_jobs: int

        :returns: instance of `CVConfig` class
        :rtype: CVConfig

        :raises SchemaError, ValueError: if the input parameters result
                                         in conflicts or are invalid
        """

        # Get dicionary of parameters (but remove "self" since that
        # doesn't need to be validated and remove values set to None
        # since they will be dealt with automatically)
        params = dict(locals())
        del params['self']
        for param in list(params):
            if params[param] is None:
                del params[param]

        # Schema
        exp_schema = Schema(
            {'db': Collection,
             'games': And(set, lambda x: x.issubset(VALID_GAMES)),
             'learners': And([str],
                             lambda learners: all(learner in LEARNER_DICT_KEYS
                                                  for learner in learners)),
             'param_grids': [[{str: list}]],
             'training_rounds': And(int, lambda x: x > 1),
             'training_samples_per_round': And(int, lambda x: x > 0),
             'grid_search_samples_per_fold': And(int, lambda x: x > 1),
             'non_nlp_features': And({str}, lambda x: LABELS.issuperset(x)),
             'prediction_label':
                 And(str,
                     lambda x: x in LABELS and not x in params['non_nlp_features']),
             'output_path': And(str, lambda x: isdir(output_path)),
             Default('objective', default=None): lambda x: x in OBJ_FUNC_ABBRS_DICT,
             Default('data_sampling', default='even'):
                And(str, lambda x: x in ex.ExperimentalData.sampling_options),
             Default('grid_search_folds', default=5): And(int, lambda x: x > 1),
             Default('hashed_features', default=None):
                Or(None,
                   lambda x: not isinstance(x, bool)
                             and isinstance(x, int)
                             and x > -1),
             Default('nlp_features', default=True): bool,
             Default('bin_ranges', default=None):
                Or(None,
                   And([(float, float)],
                       lambda x: validate_bin_ranges(x) is None)),
             Default('lognormal', default=False): bool,
             Default('power_transform', default=None):
                Or(None, And(float, lambda x: x != 0.0)),
             Default('majority_baseline', default=True): bool,
             Default('rescale', default=True): bool,
             Default('n_jobs', default=1): And(int, lambda x: x > 0)
             }
            )

        # Validate the schema
        try:
            self.validated = exp_schema.validate(params)
        except (ValueError, SchemaError) as e:
            msg = ('The set of passed-in parameters was not able to be '
                   'validated and/or the bin ranges values, if specified, were'
                   ' not able to be validated.')
            logerr('{0}:\n\n{1}'.format(msg, e))
            raise e

        # Set up the experiment
        self._further_validate_and_setup()

    def _further_validate_and_setup(self) -> None:
        """
        Further validate the experiment's configuration settings and set
        up certain configuration settings, such as setting the total
        number of hashed features to use, etc.

        :returns: None
        :rtype: None
        """

        # Make sure parameters make sense/are valid
        if len(self.validated['learners']) != len(self.validated['param_grids']):
            raise SchemaError(autos=None,
                              errors='The lists of of learners and parameter '
                                     'grids must be the same size.')
        if (self.validated['hashed_features'] is not None
            and self.validated['hashed_features'] == 0):
                self.validated['hashed_features'] = self._n_features_feature_hashing
        if self.validated['lognormal'] and self.validated['power_transform']:
            raise SchemaError(autos=None,
                              errors='Both "lognormal" and "power_transform" '
                                     'were set simultaneously.')
        if len(self.validated['learners']) != len(self.validated['param_grids']):
            raise SchemaError(autos=None,
                              errors='The "learners" and "param_grids" '
                                      'parameters were both set and the '
                                      'lengths of the lists are unequal.')


class RunCVExperiments(object):
    """
    Class for conducting sets of incremental cross-validation
    experiments.
    """

    # Constants
    default_cursor_batch_size_ = 50

    def __init__(self, config: CVConfig) -> 'RunCVExperiments':
        """
        Initialize object.

        :param config: an `CVConfig` instance containing configuration
                       options relating to the experiment, etc.
        :type config: CVConfig
        """

        # Experiment configuration settings
        self.cfg_ = pd.Series(config.validated)
        cfg = self.cfg_

        # Games
        if not cfg.games:
            raise ValueError('The set of games must be greater than zero!')
        self.games_string_ = ', '.join(cfg.games)

        # Output path and output file names/templates
        self.stats_report_path_ = join(cfg.output_path, 'cv_stats.csv')
        self.aggregated_stats_report_path_ = join(cfg.output_path,
                                                  'cv_stats_aggregated.csv')
        self.model_weights_path_template_ = join(cfg.output_path,
                                                 '{0}_model_weights_{1}.csv')
        self.model_path_template_ = join(cfg.output_path, '{0}_{1}.model')
        if cfg.majority_baseline:
            self.majority_baseline_report_path_ = join(cfg.output_path,
                                                       'maj_baseline_stats.csv')
        if cfg.lognormal or cfg.power_transform:
            self.transformation_string_ = ('ln' if cfg.lognormal
                                           else 'x**{0}'.format(cfg.power_transform))
        else:
            self.transformation_string_ = 'None'

        # Objective function
        if not cfg.objective in OBJ_FUNC_ABBRS_DICT:
            raise ValueError('Unrecognized objective function used: {0}. '
                             'These are the available objective functions: {1}.'
                             .format(cfg.objective, OBJ_FUNC_ABBRS_STRING))

        # Data-set- and database-related variables
        self.batch_size_ = \
            (cfg.training_samples_per_round
             if cfg.training_samples_per_round < self.default_cursor_batch_size_
             else self.default_cursor_batch_size_)
        self.projection_ = {'_id': 0}
        if not cfg.nlp_features:
            self.projection_['nlp_features'] = 0
        self.data_ = self._generate_experimental_data()

        # Create and fit a vectorizer with all possible samples
        train_ids = list(chain(*self.data_.training_set))
        grid_search_ids = list(chain(*self.data_.grid_search_set))
        all_ids = train_ids + grid_search_ids
        self.vec_ = self._make_vectorizer(all_ids,
                                          hashed_features=cfg.hashed_features)

        # Store all of the labels used for grid search and training (for
        # majority label identification)
        self.y_all_ = []

        # Learner-related variables
        self.learners_ = [LEARNER_DICT[learner] for learner in cfg.learners]
        self.learner_names_ = [LEARNER_ABBRS_DICT[learner] for learner
                               in cfg.learners]

        # Make a temporary directory for storing compressed datasets
        self.temp_dir = mkdtemp()

        # Make a filter for dataset compression
        self.filter = Filters(complevel=5, complib='blosc')

        # Do grid search round
        loginfo('Executing parameter grid search learning round...')
        self.gs_cv_folds_ = None
        self.learner_gs_cv_dict_ = self._do_grid_search_round()

        # Do incremental learning experiments
        loginfo('Incremental learning cross-validation experiments '
                'initialized...')
        self._do_training_cross_validation()
        self.training_cv_aggregated_stats_ = \
            ex.aggregate_cross_validation_experiments_stats(self.cv_learner_stats_)

        # Generate a report with the results from the cross-validation
        # experiments
        self.generate_learning_reports()

        # Generate statistics for the majority baseline model
        if cfg.majority_baseline:
            self._majority_baseline_stats = self._evaluate_majority_baseline_model()

        self._tear_down()

    def _tear_down(self, keep: bool = True) -> None:
        """
        Remove the temporary directory where compressed datasets are
        stored if `keep` is False.
        
        :param keep: boolean value signifying whether or not to delete
        """

        if not keep:
            rmtree(self.temp_dir)

    def _resolve_objective_function(self) -> Scorer:
        """
        Resolve value of parameter to be passed in to the `scoring`
        parameter in `GridSearchCV`, which can be `None`, a string, or a
        callable.

        :returns: a value to pass into the `scoring` parameter in
                  `GridSearchCV`, which can be None to use the default,
                  a string value that represents one of the scoring
                  functions, or a custom scorer function (via
                  `make_scorer`)
        :rtype: str, None, callable
        """

        objective = self.cfg_.objective
        if objective == 'accuracy':
            return make_scorer(ex.accuracy_score_round_inputs)
        if objective.startswith('precision'):
            if objective.endswith('macro'):
                return make_scorer(ex.precision_score_round_inputs,
                                   average='macro')
            elif objective.endswith('weighted'):
                return make_scorer(ex.precision_score_round_inputs,
                                   average='weighted')
        if objective.startswith('f1'):
            if objective.endswith('macro'):
                return make_scorer(ex.f1_score_round_inputs,
                                   average='macro')
            elif objective.endswith('weighted'):
                return make_scorer(ex.f1_score_round_inputs,
                                   average='weighted')
            elif objective.endswith('least_frequent'):
                return make_scorer(ex.f1_score_least_frequent_round_inputs)
        if objective == 'pearson_r':
            return make_scorer(pearson)
        if objective == 'spearman':
            return make_scorer(spearman)
        if objective == 'kendall_tau':
            return make_scorer(kendall_tau)
        if objective.startswith('uwk'):
            if objective == 'uwk':
                return make_scorer(ex.kappa_round_inputs)
            return make_scorer(ex.kappa_round_inputs,
                               allow_off_by_one=True)
        if objective.startswith('lwk'):
            if objective == 'lwk':
                return make_scorer(ex.kappa_round_inputs,
                                   weights='linear')
            return make_scorer(ex.kappa_round_inputs,
                               weights='linear',
                               allow_off_by_one=True)
        if objective.startswith('qwk'):
            if objective == 'qwk':
                return make_scorer(ex.kappa_round_inputs,
                                   weights='quadratic')
            return make_scorer(ex.kappa_round_inputs,
                               weights='quadratic',
                               allow_off_by_one=True)
        return objective

    def _generate_experimental_data(self):
        """
        Call `src.experiments.ExperimentalData` to generate a set of
        data to be used for grid search, training, etc.
        """

        loginfo('Extracting dataset...')
        cfg = self.cfg_
        return ex.ExperimentalData(db=cfg.db,
                                   prediction_label=cfg.prediction_label,
                                   games=cfg.games,
                                   folds=cfg.training_rounds,
                                   fold_size=cfg.training_samples_per_round,
                                   grid_search_folds=cfg.grid_search_folds,
                                   grid_search_fold_size=
                                       cfg.grid_search_samples_per_fold,
                                   sampling=cfg.data_sampling,
                                   lognormal=cfg.lognormal,
                                   power_transform=cfg.power_transform,
                                   bin_ranges=cfg.bin_ranges,
                                   batch_size=self.batch_size_)

    def _make_vectorizer(self, ids: List[str],
                         hashed_features: Optional[int] = None) -> Vectorizer:
        """
        Make a vectorizer.

        :param ids: a list of sample ID strings with which to fit the
                    vectorizer
        :type ids: list
        :param hashed_features: if feature hasing is being used, provide
                                the number of features to use;
                                otherwise, the value should be False
        :type hashed_features: bool or int

        :returns: a vectorizer, i.e., DictVectorizer or FeatureHasher
        :rtype: Vectorizer

        :raises ValueError: if the value of `hashed_features` is not
                            greater than zero or `ids` is empty
        """

        if hashed_features:
            if hashed_features < 1:
                raise ValueError('The value of "hashed_features" should be a '
                                 'positive integer, preferably a very large '
                                 'integer.')
            vec = FeatureHasher(n_features=hashed_features,
                                non_negative=True)
        else:
            vec = DictVectorizer(sparse=True)

        if not ids:
            raise ValueError('The "ids" parameter is empty.')

        vec.fit(self._generate_samples(ids, 'x'))

        return vec

    def _generate_samples(self, ids: List[str], key: Optional[str] = None) \
        -> Iterable[Union[Dict[str, Any], str, Numeric]]:
        """
        Generate feature dictionaries for the review samples in the
        given cursor.

        Provides a lower-memory way of fitting a vectorizer, for
        example.

        :param ids: list of ID strings
        :type ids: list
        :param key: yield only the value of the specified key (if a key
                    is specified), can be the following values: 'y',
                    'x', or 'id'
        :type key: str or None

        :yields: feature dictionary
        :ytype: dict, str, int, float, etc.
        """

        cfg = self.cfg_
        for doc in ex.make_cursor(cfg.db,
                                  projection=self.projection_,
                                  batch_size=self.batch_size_,
                                  id_strings=ids):
            sample = ex.get_data_point(doc,
                                       prediction_label=cfg.prediction_label,
                                       nlp_features=cfg.nlp_features,
                                       non_nlp_features=cfg.non_nlp_features,
                                       lognormal=cfg.lognormal,
                                       power_transform=cfg.power_transform,
                                       bin_ranges=cfg.bin_ranges)

            # Either yield the sample given the specified key or yield
            # the whole sample (or, if the sample is equal to None,
            # continue)
            if not sample:
                continue
            yield sample.get(key, sample)

    def _do_grid_search_round(self) -> Dict[str, GridSearchCV]:
        """
        Do grid search round.

        :returns: dictionary of learner names mapped to already-fitted
                  `GridSearchCV` instances, including attributes such as
                  `best_estimator_`
        :rtype: dict
        """

        cfg = self.cfg_

        # Get the data to use, vectorizing the sample feature dictionaries
        grid_search_all_ids = list(chain(*self.data_.grid_search_set))
        y_train = list(self._generate_samples(grid_search_all_ids, 'y'))
        X_train = self.vec_.transform(self._generate_samples(grid_search_all_ids, 'x'))

        # Update `self.y_all_` with all of the sample labels used during
        # grid search cross-validation (for majority label
        # identification)
        self.y_all_.extend(y_train)

        # Store the training data in a compressed table and then remove
        # `y_train` and `X_train`
        grid_search_compressed_file_path = join(self.temp_dir, "grid_search.h5")
        grid_search_compressed_file = \
            open_file(grid_search_compressed_file_path, mode="w")
        X_train_carray = (grid_search_compressed_file
                          .create_carray(grid_search_compressed_file.root,
                                         'X_train_carray',
                                         Atom.from_dtype(X_train.dtype),
                                         shape=X_train.shape,
                                         filters=self.filter))
        X_train_carray[:] = X_train.todense()
        y_train_carray = (grid_search_compressed_file
                          .create_carray(grid_search_compressed_file.root,
                                         'y_train_carray',
                                         Atom.from_dtype(y_train.dtype),
                                         shape=y_train.shape,
                                         filters=self.filter))
        y_train_carray[:] = y_train.todense()
        grid_search_compressed_file.close()
        del y_train
        del X_train

        # Read in the datasets from the compressed file using the
        # `H5FD_CORE` driver
        grid_search_compressed_file = \
            open_file(grid_search_compressed_file_path, mode='r',
                      driver='H5FD_CORE')
        X_train_carray = grid_search_compressed_file.root.X_train_carray
        y_train_carray = grid_search_compressed_file.root.y_train_carray

        # Make a `StratifiedKFold` object using the list of labels
        # NOTE: This will effectively redistribute the samples in the
        # various grid search folds, but it will maintain the
        # distribution of labels. Furthermore, due to the use of the
        # `RandomState` object, it should always happen in the exact
        # same way.
        prng = np.random.RandomState(12345)
        self.gs_cv_folds_ = StratifiedKFold(y=y_train_carray,
                                            n_folds=self.data_.grid_search_folds,
                                            shuffle=True,
                                            random_state=prng)

        # Iterate over the learners/parameter grids, executing the grid search
        # cross-validation for each
        loginfo('Doing a grid search cross-validation round with {0} folds for'
                ' each learner and each corresponding parameter grid.'
                .format(self.data_.grid_search_folds))
        n_jobs_learners = ['Perceptron', 'SGDClassifier',
                           'PassiveAggressiveClassifier']
        learner_gs_cv_dict = {}
        for learner, learner_name, param_grids in zip(self.learners_,
                                                      self.learner_names_,
                                                      cfg.param_grids):

            loginfo('Grid search cross-validation for {0}...'
                    .format(learner_name))

            # If the learner is `MiniBatchKMeans`, set the `batch_size`
            # parameter to the number of training samples
            if learner_name == 'MiniBatchKMeans':
                for param_grid in param_grids:
                    param_grid['batch_size'] = [len(y_train)]

            # If learner is of any of the learner types in
            # `n_jobs_learners`, add in the `n_jobs` parameter specified
            # in the config (but only do so if that `n_jobs` value is
            # greater than 1 since it won't matter because 1 is the
            # default, anyway)
            if cfg.n_jobs > 1:
                if learner_name in n_jobs_learners:
                    for param_grid in param_grids:
                        param_grid['n_jobs'] = [cfg.n_jobs]

            # Make `GridSearchCV` instance
            folds_diff = cfg.grid_search_folds - self.data_.grid_search_folds
            if (self.data_.grid_search_folds < 2
                or folds_diff/cfg.grid_search_folds > 0.25):
                msg = ('Either there weren\'t enough folds after collecting '
                       'data (via `ExperimentalData`) to do the grid search '
                       'round or the number of folds had to be reduced to such'
                       ' a degree that it would mean a +25\% reduction in the '
                       'total number of folds used during the grid search '
                       'round.')
                logerr(msg)
                raise ValueError(msg)
            gs_cv = GridSearchCV(learner(),
                                 param_grids,
                                 cv=self.gs_cv_folds_,
                                 scoring=self._resolve_objective_function())

            # Do the grid search cross-validation
            gs_cv.fit(X_train_carray, y_train_carray)
            learner_gs_cv_dict[learner_name] = gs_cv

        # Close the compressed dataset file
        grid_search_compressed_file.close()

        return learner_gs_cv_dict

    def _do_training_cross_validation(self) -> None:
        """
        Do cross-validation with training data. Each train/test split will
        represent an individual incremental learning experiment, i.e., starting
        with the best estimator from the grid search round, learn little by
        little from batches of training samples and evaluate on the held-out
        partition of data.

        :returns: None
        :rtype: None
        """

        cfg = self.cfg_

        # Get dictionary mapping learner names to the corresponding
        # "best" estimator instances from the grid search round
        self.best_estimators_gs_cv_dict = \
            {learner_name: learner_gs_cv.best_estimator_
             for learner_name, learner_gs_cv in self.learner_gs_cv_dict_.items()}

        # Make a dictionary mapping each learner name to a list of
        # individual copies of the grid search cross-validation round's
        # best estimator instances, with the length of the list equal to
        # the number of folds in the training set since each of these
        # estimator instances will be incrementally improved upon and
        # evaluated
        self.cv_learners_ = [[copy(self.best_estimators_gs_cv_dict[learner_name])
                              for learner_name in self.learner_names_]
                             for _ in range(self.data_.folds)]
        self.cv_learners_ = dict(zip(self.learner_names_,
                                     zip(*self.cv_learners_)))
        self.cv_learners_ = {k: list(v) for k, v in self.cv_learners_.items()}

        # Make a list of empty lists corresponding to each learner,
        # which will be used to hold the performance stats for each
        # cross-validation leave-one-fold-out sub-experiment
        self.cv_learner_stats_ = [[] for _ in cfg.learners]

        # For each fold of the training set, train on all of the other
        # folds and evaluate on the one left out fold
        y_training_set_all = []
        training_compressed_file_path = join(self.temp_dir, "training.h5")
        test_compressed_file_path = join(self.temp_dir, "test.h5")
        for i, held_out_fold in enumerate(self.data_.training_set):

            loginfo('Cross-validation sub-experiment #{0} in progress'
                    .format(i + 1))

            # Use each training fold (except for the held-out set) to
            # incrementally build up the model
            training_folds = self.data_.training_set[:i] + self.data_.training_set[i + 1:]
            y_train_all = []
            for training_fold in training_folds:

                # Get the training data
                y_train = list(self._generate_samples(training_fold, 'y'))
                X_train = self.vec_.transform(self._generate_samples(training_fold, 'x'))

                # Store the actual input values so that rescaling can be
                # done later
                y_train_all.extend(y_train)

                # Store the data in compressed form on disk and delete
                # `y_train`/`X_train` so that it is picked up by garbage
                # collection
                if exists(training_compressed_file_path):
                    unlink(training_compressed_file_path)
                training_compressed_file = \
                    open_file(training_compressed_file_path, mode="w")
                X_train_carray = (training_compressed_file
                                  .create_carray(training_compressed_file.root,
                                                 'X_train_carray',
                                                 Atom.from_dtype(X_train.dtype),
                                                 shape=X_train.shape,
                                                 filters=self.filter))
                X_train_carray[:] = X_train.todense()
                y_train_carray = (training_compressed_file
                                  .create_carray(training_compressed_file.root,
                                                 'y_train_carray',
                                                 Atom.from_dtype(y_train.dtype),
                                                 shape=y_train.shape,
                                                 filters=self.filter))
                y_train_carray[:] = y_train.todense()
                training_compressed_file.close()
                del X_train
                del y_train

                # Read in the datasets from the compressed file using the
                # `H5FD_CORE` driver
                training_compressed_file = \
                    open_file(training_compressed_file_path, mode='r',
                              driver='H5FD_CORE')
                X_train_carray = training_compressed_file.root.X_train_carray
                y_train_carray = training_compressed_file.root.y_train_carray

                # Iterate over the learners
                for learner_name in self.learner_names_:

                    # Partially fit each estimator with the new training data
                    self.cv_learners_[learner_name][i].partial_fit(X_train_carray,
                                                                   y_train_carray)

                # Close `test_compressed_file`
                training_compressed_file.close()

            # Get mean and standard deviation for actual values
            y_train_all = np.array(y_train_all)
            y_train_mean = y_train_all.mean()
            y_train_std = y_train_all.std()

            # Get test data
            y_test = list(self._generate_samples(held_out_fold, 'y'))
            X_test = self.vec_.transform(self._generate_samples(held_out_fold, 'x'))

            # Add test labels to `y_training_set_all`
            y_training_set_all.extend(y_test)

            # Store the test data in compressed form on disk and delete
            # `X_test` so that it is picked up by garbage collection
            if exists(test_compressed_file_path):
                unlink(test_compressed_file_path)
            test_compressed_file = open_file(test_compressed_file_path, mode="w")
            X_test_carray = (test_compressed_file
                             .create_carray(test_compressed_file.root,
                                            'X_test_carray',
                                            Atom.from_dtype(X_test.dtype),
                                            shape=X_test.shape,
                                            filters=self.filter))
            X_test_carray[:] = X_test.todense()
            test_compressed_file.close()
            del X_test

            # Read in the datasets from the compressed file using the
            # `H5FD_CORE` driver
            test_compressed_file = \
                open_file(test_compressed_file_path, mode='r', driver='H5FD_CORE')
            X_test_carray = test_compressed_file.root.X_test_carray

            # Make predictions with the modified estimators
            for j, learner_name in enumerate(self.learner_names_):

                # Make predictions with the given estimator,rounding the
                # predictions
                y_test_preds = np.round(self.cv_learners_[learner_name][i]
                                        .predict(X_test_carray))

                # Rescale the predicted values based on the
                # mean/standard deviation of the actual values and
                # fit the predicted values within the original scale
                # (i.e., no predicted values should be outside the range
                # of possible values)
                y_test_preds_dict = \
                    ex.rescale_preds_and_fit_in_scale(y_test_preds,
                                                      self.data_.classes,
                                                      y_train_mean,
                                                      y_train_std)

                if cfg.rescale:
                    y_test_preds = y_test_preds_dict['rescaled']
                else:
                    y_test_preds = y_test_preds_dict['fitted_only']

                # Evaluate the predictions and add to list of evaluation
                # reports for each learner
                (self.cv_learner_stats_[j]
                 .append(ex.evaluate_predictions_from_learning_round(
                             y_test=y_test,
                             y_test_preds=y_test_preds,
                             classes=self.data_.classes,
                             prediction_label=cfg.prediction_label,
                             non_nlp_features=cfg.non_nlp_features,
                             nlp_features=cfg.nlp_features,
                             learner=self.cv_learners_[learner_name][i],
                             learner_name=learner_name,
                             games=cfg.games,
                             test_games=cfg.games,
                             _round=i + 1,
                             iteration_rounds=self.data_.folds,
                             n_train_samples=len(y_train_all),
                             n_test_samples=len(held_out_fold),
                             rescaled=cfg.rescale,
                             transformation_string=self.transformation_string_,
                             bin_ranges=cfg.bin_ranges)))

            # Close `test_compressed_file`
            test_compressed_file.close()

        # Update `self.y_all_` with all of the sample labels used during
        # cross-validation (for majority label identification)
        self.y_all_.extend(y_training_set_all)

    def _get_majority_baseline(self) -> np.ndarray:
        """
        Generate a majority baseline array of prediction labels.

        :returns: array of prediction labels
        :rtype: np.ndarray
        """

        self._majority_label = max(set(self.y_all_), key=self.y_all_.count)
        return np.array([self._majority_label]*len(self.y_all_))

    def _evaluate_majority_baseline_model(self) -> pd.Series:
        """
        Evaluate the majority baseline model predictions.

        :returns: a Series containing the majority label system's
                  performance metrics and attributes
        :rtype: pd.Series
        """

        cfg = self.cfg_
        stats_dict = ex.compute_evaluation_metrics(self.y_all_,
                                                   self._get_majority_baseline(),
                                                   self.data_.classes)
        stats_dict.update({'games' if len(cfg.games) > 1 else 'game':
                               self.games_string_
                               if VALID_GAMES.difference(cfg.games)
                               else 'all_games',
                           'prediction_label': cfg.prediction_label,
                           'majority_label': self._majority_label,
                           'learner': 'majority_baseline_model',
                           'transformation': self.transformation_string_})
        if cfg.bin_ranges:
            stats_dict.update({'bin_ranges': cfg.bin_ranges})
        return pd.Series(stats_dict)

    def generate_majority_baseline_report(self) -> None:
        """
        Generate a CSV file reporting on the performance of the
        majority baseline model.

        :returns: None
        :rtype: None
        """

        self._majority_baseline_stats.to_csv(self.majority_baseline_report_path_)

    def generate_learning_reports(self) -> None:
        """
        Generate report for the cross-validation experiments.

        :returns: None
        :rtype: None
        """

        # Generate a report consisting of the evaluation metrics for
        # each sub-experiment comprising each cross-validation
        # experiment for each learner
        (pd.DataFrame(list(chain(*self.cv_learner_stats_)))
         .to_csv(self.stats_report_path_,
                 index=False))

        # Generate a report consisting of the aggregated evaluation
        # metrics from each cross-validation experiment with each
        # learner
        (self.training_cv_aggregated_stats_
         .to_csv(self.aggregated_stats_report_path_,
                 index=False))

    def store_sorted_features(self) -> None:
        """
        Store files with sorted lists of features and their associated
        coefficients from each model.

        :returns: None
        :rtype: None
        """

        makedirs(dirname(self.model_weights_path_template_), exist_ok=True)

        # Generate feature weights files and a README.json providing
        # the parameters corresponding to each set of feature weights
        params_dict = {}
        for learner_name in self.cv_learners_:

            # Skip MiniBatchKMeans models
            if learner_name == 'MiniBatchKMeans':
                logdebug('Skipping MiniBatchKMeans learner instances since '
                         'coefficients can not be extracted from them.')
                continue

            for i, estimator in enumerate(self.cv_learners_[learner_name]):

                # Get dataframe of the features/coefficients
                try:
                    ex.print_model_weights(estimator,
                                           learner_name,
                                           self.data_.classes,
                                           self.cfg_.games,
                                           self.vec_,
                                           self.model_weights_path_template_
                                           .format(learner_name, i + 1))
                    params_dict.setdefault(learner_name, {})
                    params_dict[learner_name][i] = estimator.get_params()
                except ValueError:
                    logerr('Could not generate features/feature coefficients '
                           'dataframe for {0}...'.format(learner_name))

        # Save parameters file also
        if params_dict:
            dump(params_dict,
                 open(join(dirname(self.model_weights_path_template_),
                           'model_params_readme.json'), 'w'),
                 indent=4)

    def store_models(self) -> None:
        """
        Save the learners to disk.

        :returns: None
        :rtype: None
        """

        # Iterate over the learner types (for which there will be
        # separate instances for each sub-experiment of the
        # cross-validation experiment)
        for learner_name in self.cv_learners_:
            loginfo('Saving {0} model files to disk...'.format(learner_name))
            for i, estimator in enumerate(self.cv_learners_[learner_name]):
                loginfo('Saving {0} model file #{1}'.format(learner_name, i + 1))
                joblib.dump(estimator,
                            self.model_path_template_.format(learner_name, i + 1))

def main(argv=None):
    parser = ArgumentParser(description='Run incremental learning '
                                        'experiments.',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    _add_arg = parser.add_argument
    _add_arg('--games',
             help='Game(s) to use in experiments; or "all" to use data from '
                  'all games.',
             type=str,
             required=True)
    _add_arg('--out_dir',
             help='Directory in which to output data related to the results '
                  'of the conducted experiments.',
             type=str,
             required=True)
    _add_arg('--train_rounds',
             help='The maximum number of rounds of learning to conduct (the '
                  'number of rounds will necessarily be limited by the amount'
                  ' of training data and the number of samples used per '
                  'round). Use "0" to do as many rounds as possible.',
             type=int,
             default=0)
    _add_arg('--train_samples_per_round',
             help='The maximum number of training samples to use in each '
                  'round.',
             type=int,
             default=100)
    _add_arg('--grid_search_folds',
             help='The maximum number of folds to use in the grid search '
                  'round.',
             type=int,
             default=5)
    _add_arg('--grid_search_samples_per_fold',
             help='The maximum number of training samples to use in each grid '
                  'search fold.',
             type=int,
             default=1000)
    _add_arg('--prediction_label',
             help='Label to predict.',
             choices=LABELS,
             default='total_game_hours')
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
    _add_arg('--data_sampling',
             help="Method used for sampling the data.",
             choices=ex.ExperimentalData.sampling_options,
             default='even')
    _add_arg('--learners',
             help='Comma-separated list of learning algorithms to try. Refer '
                  'to list of learners above to find out which abbreviations '
                  'stand for which learners. Set of available learners: {0}. '
                  'Use "all" to include all available learners.'
                  .format(LEARNER_ABBRS_STRING),
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
    _add_arg('--use_feature_hasher',
             help='Use FeatureHasher to be more memory-efficient.',
             action='store_true',
             default=False)
    _add_arg('--rescale_predictions',
             help='Rescale prediction values based on the mean/standard '
                  'deviation of the input values and fit all predictions into '
                  'the expected scale. Don\'t use if the experiment involves '
                  'labels rather than numeric values.',
             action='store_true',
             default=False)
    _add_arg('--objective',
             help='Objective function to use in determining which learner/set'
                  ' of parameters resulted in the best performance.',
             choices=OBJ_FUNC_ABBRS_DICT.keys(),
             default='qwk')
    _add_arg('--n_jobs',
             help='Value of "n_jobs" parameter to pass in to learners whose '
                  'tasks can be parallelized. Should be no more than the '
                  'number of cores (or virtual cores) for the machine that '
                  'this process is run on.',
             type=int,
             default=1)
    _add_arg('--evaluate_maj_baseline',
             help='Evaluate the majority baseline model.',
             action='store_true',
             default=False)
    _add_arg('--save_best_features',
             help='Get the best features from each model and write them out '
                  'to files.',
             action='store_true',
             default=False)
    _add_arg('--save_model_files',
             help='Save model files to disk.',
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
             help='Path to log file. If no path is specified, then a "logs" '
                  'directory will be created within the directory specified '
                  'via the --out_dir argument and a log will automatically be '
                  'stored.',
             type=str,
             required=False)
    args = parser.parse_args()

    # Command-line arguments and flags
    games = parse_games_string(args.games)
    train_rounds = args.train_rounds
    train_samples_per_round = args.train_samples_per_round
    grid_search_folds = args.grid_search_folds
    grid_search_samples_per_fold = args.grid_search_samples_per_fold
    prediction_label = args.prediction_label
    non_nlp_features = parse_non_nlp_features_string(args.non_nlp_features,
                                                     prediction_label)
    only_non_nlp_features = args.only_non_nlp_features
    nbins = args.nbins
    bin_factor = args.bin_factor
    lognormal = args.lognormal
    power_transform = args.power_transform
    feature_hashing = args.use_feature_hasher
    rescale_predictions = args.rescale_predictions
    data_sampling = args.data_sampling
    learners = parse_learners_string(args.learners)
    host = args.mongodb_host
    port = args.mongodb_port
    objective = args.objective
    n_jobs = args.n_jobs
    evaluate_maj_baseline = args.evaluate_maj_baseline
    save_best_features = args.save_best_features
    save_model_files = args.save_model_files

    # Validate the input arguments
    if isfile(realpath(args.out_dir)):
        raise FileExistsError('The specified output destination is the name '
                              'of a currently existing file.')
    else:
        output_path = realpath(args.out_dir)
        
    if save_best_features:
        if learners == ['mbkm']:
            loginfo('The specified set of learners do not work with the '
                    'current way of extracting features from models and, '
                    'thus, --save_best_features, will be ignored.')
            save_best_features = False
        if feature_hashing:
            raise ValueError('The --save_best_features option cannot be used '
                             'in conjunction with the --use_feature_hasher '
                             'option.')
    if args.log_file_path:
        if isdir(realpath(args.log_file_path)):
            raise FileExistsError('The specified log file path is the name of'
                                  ' a currently existing directory.')
        else:
            log_file_path = realpath(args.log_file_path)
    else:
        log_file_path = join(output_path, 'logs', 'learn.log')
    log_dir = dirname(log_file_path)
    if lognormal and power_transform:
        raise ValueError('Both "lognormal" and "power_transform" were '
                         'specified simultaneously.')

    # Output results files to output directory
    makedirs(output_path, exist_ok=True)
    makedirs(log_dir, exist_ok=True)

    # Set up file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging_debug)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Log a bunch of job attributes
    loginfo('Output directory: {0}'.format(output_path))
    loginfo('Game{0} to train/evaluate models on: {1}'
            .format('s' if len(games) > 1 else '',
                    ', '.join(games) if VALID_GAMES.difference(games)
                    else 'all games'))
    loginfo('Maximum number of learning rounds to conduct: {0}'
            .format(train_rounds))
    loginfo('Maximum number of training samples to use in each round: {0}'
            .format(train_samples_per_round))
    loginfo('Maximum number of grid search folds to use during the grid search'
            ' round: {0}'.format(grid_search_folds))
    loginfo('Maximum number of training samples to use in each grid search '
            'fold: {0}'.format(grid_search_samples_per_fold))
    loginfo('Prediction label: {0}'.format(prediction_label))
    loginfo('Data sampling method: {0}'.format(data_sampling))
    loginfo('Lognormal transformation: {0}'.format(lognormal))
    loginfo('Power transformation: {0}'.format(power_transform))
    loginfo('Non-NLP features to use: {0}'
            .format(', '.join(non_nlp_features) if non_nlp_features else 'none'))
    if only_non_nlp_features:
        if not non_nlp_features:
            raise ValueError('No features to train a model on since the '
                             '--only_non_nlp_features flag was used and the '
                             'set of non-NLP features is empty.')
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
    if rescale_predictions:
        loginfo('Rescaling predicted values based on the mean/standard '
                'deviation of the input values.')
    loginfo('Learners: {0}'.format(', '.join([LEARNER_ABBRS_DICT[learner]
                                              for learner in learners])))
    loginfo('Using {0} as the objective function'.format(objective))
    if n_jobs < 1:
        msg = '--n_jobs must be greater than 0.'
        logerr(msg)
        raise ValueError(msg)
    loginfo('Number of tasks to run in parallel during learner fitting (when '
            'possible to run tasks in parallel): {0}'.format(n_jobs))

    # Connect to running Mongo server
    loginfo('MongoDB host: {0}'.format(host))
    loginfo('MongoDB port: {0}'.format(port))
    try:
        db = connect_to_db(host=host, port=port)
    except ConnectionFailure as e:
        logerr('Unable to connect to MongoDB reviews collection.')
        logerr(e)
        raise e

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
        try:
            bin_ranges = get_bin_ranges_helper(db,
                                               games,
                                               prediction_label,
                                               nbins,
                                               bin_factor,
                                               lognormal=lognormal,
                                               power_transform=power_transform)
        except ValueError as e:
            msg = ('Encountered a ValueError while computing the bin ranges '
                   'given {0} and {1} as the values for the number of bins and'
                   ' the bin factor. This could be due to an unrecognized '
                   'prediction label, which would cause no values to be found,'
                   'which in turn would result in an empty array.'
                   .format(nbins, bin_factor))
            logerr(msg)
            raise e
        if lognormal or power_transform:
            transformation = ('lognormal' if lognormal
                              else 'x**{0}'.format(power_transform))
        else:
            transformation = None
        loginfo('Bin ranges (nbins = {0}, bin_factor = {1}{2}): {3}'
                .format(nbins,
                        bin_factor,
                        ', {0} transformation'.format(transformation)
                        if transformation
                        else '',
                        bin_ranges))

    # Do learning experiments
    loginfo('Starting incremental learning experiments...')
    learners = sorted(learners)
    try:
        cfg = CVConfig(
                  db=db,
                  games=games,
                  learners=learners,
                  param_grids=[find_default_param_grid(learner)
                               for learner in learners],
                  training_rounds=train_rounds,
                  training_samples_per_round=train_samples_per_round,
                  grid_search_samples_per_fold=grid_search_samples_per_fold,
                  non_nlp_features=non_nlp_features,
                  prediction_label=prediction_label,
                  output_path=output_path,
                  objective=objective,
                  data_sampling=data_sampling,
                  grid_search_folds=grid_search_folds,
                  hashed_features=0 if feature_hashing else None,
                  nlp_features=not only_non_nlp_features,
                  bin_ranges=bin_ranges,
                  lognormal=lognormal,
                  power_transform=power_transform,
                  majority_baseline=evaluate_maj_baseline,
                  rescale=rescale_predictions,
                  n_jobs=n_jobs)
    except (SchemaError, ValueError) as e:
        logerr('Encountered an exception while instantiating the CVConfig '
               'instance: {0}'.format(e))
        raise e
    try:
        experiments = RunCVExperiments(cfg)
    except ValueError as e:
        logerr('Encountered an exception while instantiating the '
               'RunCVExperiments instance: {0}'.format(e))
        raise e

    # Save the best-performing features
    if save_best_features:
        loginfo('Generating feature coefficient output files for each model '
                '(after all learning rounds)...')
        experiments.store_sorted_features()

    # Save the model files
    if save_model_files:
        loginfo('Writing out model files for each model to disk...')
        experiments.store_models()

    # Generate evaluation report for the majority baseline model, if
    # specified
    if evaluate_maj_baseline:
        loginfo('Generating report for the majority baseline model...')
        loginfo('Majority label: {0}'.format(experiments._majority_label))
        experiments.generate_majority_baseline_report()

    loginfo('Complete.')


if __name__ == '__main__':
    main()
