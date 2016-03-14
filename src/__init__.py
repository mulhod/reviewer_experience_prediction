import logging
from os import listdir
from os.path import (join,
                     exists,
                     dirname,
                     realpath)
from re import compile as recompile

from typing import (Any,
                    Dict,
                    List,
                    Union,
                    Tuple,
                    TypeVar,
                    Callable,
                    Sequence,
                    Optional)
from sklearn.cluster import MiniBatchKMeans
from sklearn.naive_bayes import (BernoulliNB,
                                 MultinomialNB)
from sklearn.feature_extraction import (FeatureHasher,
                                        DictVectorizer)
from sklearn.linear_model import (Perceptron,
                                  SGDClassifier,
                                  PassiveAggressiveRegressor,
                                  PassiveAggressiveClassifier)

from data import APPID_DICT

project_dir = dirname(dirname(realpath(__file__)))
data_dir = join(project_dir, 'data')
log_dir = join(project_dir, 'logs')
reports_dir = join(project_dir, 'reports')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Type aliases
BinRanges = List[Tuple[float, float]]
Learner = Union[Perceptron,
                MiniBatchKMeans,
                BernoulliNB,
                MultinomialNB,
                SGDClassifier,
                PassiveAggressiveRegressor,
                PassiveAggressiveClassifier]
ParamGrid = List[Dict[str, List[Any]]]
Vectorizer = Union[DictVectorizer, FeatureHasher]
Numeric = Union[int, float]
Generic = TypeVar('Generic')
ScoringFunction = Callable[[Sequence[Numeric],
                            Sequence[Numeric]],
                           Numeric]
Scorer = Optional[Union[str, ScoringFunction]]

# Seed for random state
SEED = 123456789

# Define default parameter grids
DEFAULT_PARAM_GRIDS = \
    {MiniBatchKMeans: [{'n_clusters': [3, 5, 10],
                        'init' : ['k-means++', 'random'],
                        'random_state': [SEED]}],
     BernoulliNB: [{'alpha': [0.1, 0.25, 0.5],
                    'fit_prior': [True, False]}],
     MultinomialNB: [{'alpha': [0.1, 0.25, 0.5],
                      'fit_prior': [True, False]}],
     Perceptron: [{'penalty': [None, 'l2', 'l1', 'elasticnet'],
                   'alpha': [0.0001, 0.001, 0.01, 0.1],
                   'n_iter': [5, 10],
                   'eta0': [0.5, 1, 2],
                   'random_state': [SEED]}],
     PassiveAggressiveRegressor:
         [{'C': [0.01, 0.1, 1.0, 10.0, 100.0],
           'n_iter': [5, 10],
           'random_state': [SEED],
           'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']}],
     PassiveAggressiveClassifier:
         [{'C': [0.01, 0.1, 1.0, 10.0, 100.0],
           'n_iter': [5, 10],
           'random_state': [SEED],
           'loss': ['hinge', 'squared_hinge']}],
     SGDClassifier: [{'loss': ['hinge', 'huber', 'log', 'epsilon_insensitive'],
                      'penalty': ['l1', 'l2', 'elasticnet'],
                      'n_iter': [5, 10],
                      'random_state': [SEED],
                      'eta0': [0.0, 0.001, 0.01, 0.1],
                      'alpha': [0.000001, 0.00001, 0.0001, 0.001],
                      'average': [True]}]}

# Learners
LEARNER_ABBRS_DICT = {'mbkm': 'MiniBatchKMeans',
                      'bnb': 'BernoulliNB',
                      'mnb': 'MultinomialNB',
                      'perc': 'Perceptron',
                      'pagr': 'PassiveAggressiveRegressor',
                      'pagc': 'PassiveAggressiveClassifier',
                      'sgdc': 'SGDClassifier'}
LEARNER_DICT_KEYS = frozenset(LEARNER_ABBRS_DICT.keys())
LEARNER_DICT = {k: eval(LEARNER_ABBRS_DICT[k]) for k in LEARNER_DICT_KEYS}
LEARNER_ABBRS_STRING = ', '.join(['"{0}" ({1})'.format(abbr, learner)
                                  for abbr, learner in LEARNER_ABBRS_DICT.items()])

# Objective functions
OBJ_FUNC_ABBRS_DICT = {'pearson_r': "Pearson product-moment correlation",
                       'spearman': 'Spearman\'s rank correlation coefficient',
                       'kendall_tau': 'Kendall-Tau rank correlation coefficient',
                       'precision_macro': 'precision (macro)',
                       'precision_weighted': 'precision (weighted)',
                       'recall': "recall",
                       'f1_macro': 'f1 (macro)',
                       'f1_weighted': 'f1 (weighted)',
                       'f1_score_least_frequent': 'f1 (least frequent class)',
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
               .format(abbr, ' ({0})'.format(func) if abbr != func else '')
               for abbr, func in OBJ_FUNC_ABBRS_DICT.items()])

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
FRIENDS_LABELS = frozenset({'num_friends', 'friend_player_level'})
HELPFUL_LABELS = frozenset({'num_voted_helpfulness', 'num_found_helpful',
                            'found_helpful_percentage', 'num_found_unhelpful'})
ACHIEVEMENTS_LABELS = frozenset({'num_achievements_percentage',
                                 'num_achievements_attained',
                                 'num_achievements_possible'})
LABELS_WITH_PCT_VALUES = frozenset({'num_achievements_percentage',
                                    'found_helpful_percentage'})

# Valid games
VALID_GAMES = frozenset([game for game in list(APPID_DICT) if game != 'sample'])

# Regular expressions
FLOAT_ONE_DEC = recompile(r'^\-?\d+\.\d$')
test_float_decimal_places = FLOAT_ONE_DEC.search

# Data-scraping regular expressions
SPACE = recompile(r'[\s]+')
space_sub = SPACE.sub
BREAKS_REGEX = recompile(r'\<br\>')
breaks_sub = BREAKS_REGEX.sub
COMMA = recompile(r',')
comma_sub = COMMA.sub
HELPFUL_OR_FUNNY = recompile('(helpful|funny)')
helpful_or_funny_search = HELPFUL_OR_FUNNY.search
DATE_END_WITH_YEAR_STRING = recompile(r', \d{4}$')
date_end_with_year_string_search = DATE_END_WITH_YEAR_STRING.search
COMMENT_RE_1 = recompile(r'<span id="commentthread[^<]+')
comment_re_1_search = COMMENT_RE_1.search
COMMENT_RE_2 = recompile(r'>(\d*)$')
comment_re_2_search = COMMENT_RE_2.search
UNDERSCORE = recompile(r'_')
underscore_sub = UNDERSCORE.sub
QUOTES = recompile(r'\'|"')
quotes_sub = QUOTES.sub
BACKSLASH = recompile(r'\\')
backslash_sub = BACKSLASH.sub


def get_game_files(games_str: str, data_dir_path: str = data_dir) -> List[str]:
    """
    Get list of game files (file-names only).

    :param games_str: comma-separated list of game files (that exist in
                      the data directory) with or without a .jsonlines
                      suffix (or "all" for all game files) (Note: if
                      "sample"/"sample.jsonlines" is included it will be
                      filtered out)
    :type games_str: str
    :param data_dir_path: path to data directory (defaults to
                          `src.data_dir`)
    :type data_dir_path: str

    :returns: list of games
    :rtype: list

    :raises ValueError: no games were included in the list of games (or
                        `games_str` only includes
                        "sample"/"sample.jsonlines") or there are no
                        .jsonlines files in the data directory passed in
                        via `data_dir_path`
    :raises FileNotFoundError: if file(s) corresponding to games in the
                               input cannot be found
    """

    game_files = []
    sample_file_inputs = ['sample', 'sample.jsonlines']
    if not games_str or all(game in sample_file_inputs for game
                            in games_str.split(',')):
        raise ValueError('Games string is empty or it includes only the sample'
                         ' file (not intended for actual use): {}.'
                         .format(', '.join(games_str.split(','))))
    elif games_str == "all":
        game_files = [f for f in listdir(data_dir_path) if f.endswith('.jsonlines')]

        # Remove the sample game file from the list if it exists
        if any(game in game_files for game in sample_file_inputs):
            for game in sample_file_inputs:
                if game in game_files:
                    del game_files[game_files.index(game)]

        if not game_files:
            raise ValueError('No non-sample file .jsonlines files found in '
                             '"data_dir_path".')
    else:
        for f in games_str.split(','):
            if f in sample_file_inputs:
                continue
            f = f if f.endswith('.jsonlines') else '{0}.jsonlines'.format(f)
            f_path = join(data_dir_path, f)
            if not exists(f_path):
                raise FileNotFoundError('{0} does not exist (input string: '
                                        '{1}).'.format(f_path, games_str))
            game_files.append(f)

        # Raise exception if the only file that was included was
        # "sample"/"sample.jsonlines"
        if not game_files:
            raise ValueError('No non-sample file .jsonlines file was '
                             'included.')

    return game_files


def find_default_param_grid(learner: str,
                            param_grids_dict: Dict[str, ParamGrid]
                                = DEFAULT_PARAM_GRIDS) -> ParamGrid:
    """
    Finds the default parameter grid for the specified learner.

    :param learner: abbreviated string representation of a learner
    :type learner: str
    :param param_grids_dict: dictionary of learner classes mapped to
                             parameter grids
    :type param_grids_dict: dict

    :returns: parameter grid
    :rtype: dict

    :raises ValueError: if an unrecognized learner abbreviation is used
    """

    if not learner:
        raise ValueError('Learner string is empty.')
    if not params_grids_dict:
        raise ValueError('Default parameter grids dictionary is empty. It '
                         'should consist of a dictionary mapping learner '
                         'classes to lists of parameter grids, each of which '
                         'is a list of dictionaries mapping parameters to '
                         'lists of parameter values.')
    for key_cls, grids in param_grids_dict.items():
        if issubclass(LEARNER_DICT[learner], key_cls):
            return grids
    raise ValueError('Unrecognized learner abbreviation: {0}'.format(learner))


def parse_learners_string(learners_string: str) -> set:
    """
    Parse command-line argument consisting of a set of learners to
    use (or the value "all" for all possible learners).

    :param learners_string: comma-separated list of learner
                            abbreviations (or "all" for all possible
                            learners)
    :type learners_string: str

    :returns: set of learner abbreviations
    :rtype: set

    :raises ValueError: if unrecognized learner(s) are included in the
                        input
    """

    if not learners_string:
        raise ValueError('Learners string is empty.')
    elif learners_string == 'all':
        learners = set(LEARNER_DICT_KEYS)
    else:
        learners = set(learners_string.split(','))
        if not learners.issubset(LEARNER_DICT_KEYS):
            raise ValueError('Found unrecognized learner(s) in list of '
                             'passed-in learners: {0}. Available learners: {1}.'
                             .format(', '.join(learners), LEARNER_ABBRS_STRING))

    return learners


def parse_non_nlp_features_string(features_string: str,
                                  prediction_label: str) -> set:
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

    :raises ValueError: if unrecognized features are found in the input
                        or there is a conflict between features
                        included in the input and the feature used as
                        the prediction label
    """

    if not features_string:
        raise ValueError('Features string is empty.')
    if not prediction_label or not prediction_label in LABELS:
        raise ValueError('Prediction label is empty or is an unrecognized '
                         'value.')
    elif features_string == 'all':
        non_nlp_features = set(LABELS)

        # Remove time-related labels if the prediction label is also
        # time-related
        if prediction_label in TIME_LABELS:
            [non_nlp_features.remove(label) for label in TIME_LABELS]

        # Remove friends-related labels if the prediction label is also
        # friends-related
        elif prediction_label in FRIENDS_LABELS:
            [non_nlp_features.remove(label) for label in FRIENDS_LABELS]

        # Remove helpful-related labels if the prediction label is also
        # helpful-related
        elif prediction_label in HELPFUL_LABELS:
            [non_nlp_features.remove(label) for label in HELPFUL_LABELS]

        # Remove achievements-related labels if the prediction label is
        # also achievements-related
        elif prediction_label in ACHIEVEMENTS_LABELS:
            [non_nlp_features.remove(label) for label in ACHIEVEMENTS_LABELS]

        # Otherwise, just remove the prediction label
        else:
            non_nlp_features.remove(prediction_label)

    elif features_string == 'none':
        non_nlp_features = set()
    else:
        non_nlp_features = set(features_string.split(','))

        # Raise an exception if unrecognized labels are found
        if not non_nlp_features.issubset(LABELS):
            raise ValueError('Found unrecognized feature(s) in the list of '
                             'passed-in non-NLP features: {0}. Available '
                             'features: {1}.'
                             .format(', '.join(non_nlp_features), LABELS_STRING))

        # Raise an exception if there are conflicts between the
        # prediction label and the set of non-NLP labels to use
        if (prediction_label in TIME_LABELS
            and non_nlp_features.intersection(TIME_LABELS)):
            raise ValueError('The list of non-NLP features should not '
                             'contain any of the time-related features if '
                             'the prediction label is itself a '
                             'time-related feature.')
        elif (prediction_label in FRIENDS_LABELS
              and non_nlp_features.intersection(FRIENDS_LABELS)):
            raise ValueError('The list of non-NLP features should not '
                             'contain any of the friends-related features if '
                             'the prediction label is itself a '
                             'friends-related feature.')
        elif (prediction_label in HELPFUL_LABELS
              and non_nlp_features.intersection(HELPFUL_LABELS)):
            raise ValueError('The list of non-NLP features should not '
                             'contain any of the helpful-related features if '
                             'the prediction label is itself a '
                             'helpful-related feature.')
        elif (prediction_label in ACHIEVEMENTS_LABELS
              and non_nlp_features.intersection(ACHIEVEMENTS_LABELS)):
            raise ValueError('The list of non-NLP features should not '
                             'contain any of the achievements-related '
                             'features if the prediction label is itself an '
                             'achievements-related feature.')

    return non_nlp_features


def parse_games_string(games_string: str) -> set:
    """
    Parse games string passed in via the command-line into a set of
    valid games (or the value "all" for all games).

    :param games_string: comma-separated list of games (or "all")
    :type games_string: str

    :returns: set of games
    :rtype: set

    :raises ValueError: if unrecognized game IDs are found in the input
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
