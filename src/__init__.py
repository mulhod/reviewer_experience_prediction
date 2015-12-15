import re
from sklearn.cluster import MiniBatchKMeans
from sklearn.naive_bayes import (BernoulliNB,
                                 MultinomialNB)
from sklearn.linear_model import (Perceptron,
                                  PassiveAggressiveRegressor)

from data import APPID_DICT

log_format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Seed for random state
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
LEARNER_ABBRS_DICT = {'mbkm': 'MiniBatchKMeans',
                      'bnb': 'BernoulliNB',
                      'mnb': 'MultinomialNB',
                      'perc': 'Perceptron',
                      'pagr': 'PassiveAggressiveRegressor'}
LEARNER_DICT_KEYS = frozenset(LEARNER_ABBRS_DICT.keys())
LEARNER_DICT = {k: eval(LEARNER_ABBRS_DICT[k]) for k in LEARNER_DICT_KEYS}
LEARNER_ABBRS_STRING = ', '.join(['"{0}" ({1})'.format(abbr, learner)
                                  for abbr, learner in LEARNER_ABBRS_DICT.items()])

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
FLOAT_ONE_DEC = re.compile(r'^\-?\d+\.\d$')
test_float_decimal_places = FLOAT_ONE_DEC.search

# Data-scraping regular expressions
SPACE = re.compile(r'[\s]+')
space_sub = SPACE.sub
BREAKS_REGEX = re.compile(r'\<br\>')
breaks_sub = BREAKS_REGEX.sub
COMMA = re.compile(r',')
comma_sub = COMMA.sub
HELPFUL_OR_FUNNY = re.compile('(helpful|funny)')
helpful_or_funny_search = HELPFUL_OR_FUNNY.search
DATE_END_WITH_YEAR_STRING = re.compile(r', \d{4}$')
date_end_with_year_string_search = DATE_END_WITH_YEAR_STRING.search
COMMENT_RE_1 = re.compile(r'<span id="commentthread[^<]+')
comment_re_1_search = COMMENT_RE_1.search
COMMENT_RE_2 = re.compile(r'>(\d*)$')
comment_re_2_search = COMMENT_RE_2.search
UNDERSCORE = re.compile(r'_')
underscore_sub = UNDERSCORE.sub
QUOTES = re.compile(r'\'|"')
quotes_sub = QUOTES.sub
BACKSLASH = re.compile(r'\\')
backslash_sub = BACKSLASH.sub
