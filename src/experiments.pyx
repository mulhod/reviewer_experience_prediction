"""
:author: Matt Mulholland
:date: 11/19/2015

Module of functions/classes related to learning experiments.
"""
from os.path import join

from sklearn.cluster import MiniBatchKMeans
from sklearn.naive_bayes import (BernoulliNB,
                                 MultinomialNB)
from sklearn.linear_model import (Perceptron,
                                  PassiveAggressiveRegressor)

from data import APPID_DICT

SEED = 123456789

# Define default parameter grids
DEFAULT_PARAM_GRIDS = \
    {MiniBatchKMeans: {'n_clusters': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                       'init' : ['k-means++', 'random'],
                       'random_state': [SEED]},
     BernoulliNB: {'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]},
     MultinomialNB: {'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]},
     Perceptron: {'penalty': [None, 'l2', 'l1', 'elasticnet'],
                  'alpha': [0.0001, 0.001, 0.01, 0.1],
                  'n_iter': [5, 10],
                  'random_state': [SEED]},
     PassiveAggressiveRegressor:
         {'C': [0.01, 0.1, 1.0, 10.0, 100.0],
          'n_iter': [5, 10],
          'random_state': [SEED],
          'loss': ['epsilon_insensitive',
                   'squared_epsilon_insensitive']}}

# Learners
LEARNER_NAMES_DICT = {MiniBatchKMeans: 'MiniBatchKMeans',
                      BernoulliNB: 'BernoulliNB',
                      MultinomialNB: 'MultinomialNB',
                      Perceptron: 'Perceptron',
                      PassiveAggressiveRegressor:
                          'PassiveAggressiveRegressor'}
LEARNER_ABBRS_DICT = {'mbkm': 'MiniBatchKMeans',
                      'bnb': 'BernoulliNB',
                      'mnb': 'MultinomialNB',
                      'perc': 'Perceptron',
                      'pagr': 'PassiveAggressiveRegressor'}
LEARNER_DICT_KEYS = frozenset(LEARNER_ABBRS_DICT.keys())
LEARNER_DICT = {k: eval(LEARNER_ABBRS_DICT[k]) for k in LEARNER_DICT_KEYS}
LEARNER_ABBRS_STRING = ', '.join(['"{0}" ({1})'.format(abbr, learner)
                                  for abbr, learner in LEARNER_ABBRS_DICT.items()])
LEARNERS_REQUIRING_CLASSES = frozenset({'BernoulliNB', 'MultinomialNB',
                                        'Perceptron'})

# Objective functions
OBJ_FUNC_ABBRS_DICT = {'pearson_r': "Pearson's r",
                       'significance': 'significance',
                       'precision_macro': 'precision (macro)',
                       'precision_weighted': 'precision (weighted)',
                       'f1_macro': 'f1 (macro)',
                       'f1_weighted': 'f1 (weighted)',
                       'accuracy': 'accuracy',
                       'uwk': 'unweighted kappa',
                       'uwk_off_by_one': 'unweighted kappa (off by one)',
                       'qwk': 'quadratic weighted kappa',
                       'qwk_off_by_one':
                           'quadratic weighted kappa (off by one)',
                       'lwk': 'linear weighted kappa',
                       'lwk_off_by_one': 'linear weighted kappa (off by one)'}
OBJ_FUNC_ABBRS_STRING = \
    ', '.join(['"{0}"{1}'
               .format(abbr, ' ({0})'.format(obj_func) if abbr != obj_func else '')
               for abbr, obj_func in OBJ_FUNC_ABBRS_DICT.items()])

# Feature names
LABELS = frozenset({'num_guides', 'num_games_owned', 'num_friends',
                    'num_voted_helpfulness', 'num_groups',
                    'num_workshop_items', 'num_reviews', 'num_found_funny',
                    'friend_player_level', 'num_badges', 'num_found_helpful',
                    'num_screenshots', 'num_found_unhelpful',
                    'found_helpful_percentage', 'num_comments',
                    'total_game_hours', 'total_game_hours_bin',
                    'total_game_hours_last_two_weeks',
                    'num_achievements_percentage', 'num_achievements_attained',
                    'num_achievements_possible'})
LABELS_STRING = ', '.join(LABELS)
TIME_LABELS = frozenset({'total_game_hours', 'total_game_hours_bin',
                         'total_game_hours_last_two_weeks'})

# Orderings
ORDERINGS = frozenset({'objective_last_round', 'objective_best_round',
                       'objective_slope'})

# Valid games
VALID_GAMES = frozenset([game for game in list(APPID_DICT) if game != 'sample'])


def find_default_param_grid(learner, param_grids_dict=DEFAULT_PARAM_GRIDS):
    """
    Finds the default parameter grid for the specified learner.

    :param learner: abbreviated string representation of a learner
    :type learner: str
    :param param_grids_dict: dictionary of learner classes mapped to
                             parameter grids
    :type param_grids_dict: dict

    :returns: parameter grid
    :rtype: dict

    :raises: ValueError
    """

    for key_cls, grid in param_grids_dict.items():
        if issubclass(LEARNER_DICT[learner], key_cls):
            return grid
    raise ValueError('Unrecognized learner abbreviation: {0}'.format(learner))


def parse_learners_string(learners_string):
    """
    Parse command-line argument consisting of a set of learners to
    use (or the value "all" for all possible learners).

    :param learners_string: comma-separated list of learner
                            abbreviations (or "all" for all possible
                            learners)
    :type learners_string: str

    :returns: set of learner abbreviations
    :rtype: set

    :raises: ValueError
    """

    if learners_string == 'all':
        learners = set(LEARNER_DICT_KEYS)
    else:
        learners = set(learners_string.split(','))
        if not learners.issubset(LEARNER_DICT_KEYS):
            raise ValueError('Found unrecognized learner(s) in list of '
                             'passed-in learners: {0}. Available learners: {1}.'
                             .format(', '.join(learners), LEARNER_ABBRS_STRING))

    return learners


def parse_non_nlp_features_string(features_string, prediction_label):
    """
    Parse command-line argument consisting of a set of non-NLP features
    (or one of the values "all" or "none" for all or none of the
    possible non-NLP features).

    If the prediction label is one of the time features, take other
    time features out of the set of non-NLP features since the
    information could be duplicated.

    :param features_string: comma-separated list of non-NLP features
                            (or "all"/"none" for all/none of the
                            possible non-NLP features)
    :type features_string: str
    :param prediction_label: the feature that is being predicted
    :type prediction_label: str

    :returns: set of non-NLP features to use
    :rtype: set

    :raises: ValueError
    """

    if features_string == 'all':
        non_nlp_features = set(LABELS)
        if prediction_label in TIME_LABELS:
            [non_nlp_features.remove(label) for label in TIME_LABELS]
        else:
            non_nlp_features.remove(prediction_label)
    elif features_string == 'none':
        non_nlp_features = set()
    else:
        non_nlp_features = set(features_string.split(','))
        if not non_nlp_features.issubset(LABELS):
            raise ValueError('Found unrecognized feature(s) in the list of '
                             'passed-in non-NLP features: {0}. Available '
                             'features: {1}.'
                             .format(', '.join(non_nlp_features),
                                     ', '.join(LABELS)))
        if (prediction_label in TIME_LABELS
            and non_nlp_features.intersection(TIME_LABELS)):
            raise ValueError('The list of non-NLP features should not '
                             'contain any of the time-related features if '
                             'the prediction label is itself a '
                             'time-related feature.')

    return non_nlp_features


def parse_games_string(games_string):
    """
    Parse games string passed in via the command-line into a set of
    valid games (or the value "all" for all games).

    :param games_string: comma-separated list of games (or "all")
    :type games_string: str

    :returns: set of games
    :rtype: set

    :raises: ValueError
    """

    # Return empty set for empty string
    if not games_string:
        return set()

    # Return the full list of games if 'all' is used
    if games_string == 'all':
        return set(VALID_GAMES)

    # Parse string
    specified_games = games_string.split(',')

    # Raise exception if the list contains unrecognized games
    if any(game not in VALID_GAMES for game in specified_games):
        raise ValueError('Found unrecognized games in the list of specified '
                         'games: {0}. These are the valid games (in addition '
                         'to using "all" for all games): {1}.'
                         .format(', '.join(specified_games),
                                 ', '.join(VALID_GAMES)))
    return set(specified_games)


def generate_learning_reports(exps, dfs, games, output_path):
    """
    Generate experimental reports for each run represented in the lists
    of input dataframes.

    The output files will have indices in their names, which simply
    correspond to the sequence in which they occur in the list of input
    dataframes.

    :param exps: object representing a set of experimental machine
                 learning tasks
    :type exps: RunExperiments
    :param dfs: list of dataframes
    :type dfs: list
    :param games: list of games
    :type games: list
    :param output_path: path to destination directory
    :type output_path: str

    :rtype: None
    """

    cdef int i
    cdef int zero = 0
    cdef int one = 1
    for i, df in enumerate(dfs):
        learner_name = df[exps.__learner__].irow(zero)
        df.to_csv(join(output_path,
                       '{0}_{1}_learning_stats_{2}.csv'
                       .format('_'.join(games), learner_name, i + one)),
                  index=False)
