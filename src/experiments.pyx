"""
:author: Matt Mulholland
:date: 11/19/2015

Module of functions/classes related to learning experiments.
"""
from os.path import join
from collections import Counter

from pymongo import collection

from data import APPID_DICT
from src import (LABELS,
                 TIME_LABELS,
                 VALID_GAMES,
                 LEARNER_DICT,
                 FRIENDS_LABELS,
                 HELPFUL_LABELS,
                 LEARNER_DICT_KEYS,
                 ACHIEVEMENTS_LABELS,
                 DEFAULT_PARAM_GRIDS,
                 LEARNER_ABBRS_STRING,
                 LABELS_WITH_PCT_VALUES)
from src.datasets import (get_bin,
                          validate_bin_ranges,
                          compute_label_value)


def find_default_param_grid(learner: str,
                            param_grids_dict: dict = DEFAULT_PARAM_GRIDS) -> dict:
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

    for key_cls, grid in param_grids_dict.items():
        if issubclass(LEARNER_DICT[learner], key_cls):
            return grid
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

    if learners_string == 'all':
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

    if features_string == 'all':
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
                             .format(', '.join(non_nlp_features),
                                     ', '.join(LABELS)))

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


def distributional_info(db: collection, label: str, games: list,
                        partition: str = 'test', bin_ranges: list = None,
                        lognormal: bool = False, limit: int = 0) -> dict:
    """
    Generate some distributional information regarding the given label
    (or for the implicit/transformed labels given a list of label bin
    ranges and/or setting `lognormal` to True) for the for the given
    list of games.

    By default, the 'test' partition is used, but the 'train' partition
    can be specified via the `partition` parameter. If 'all' is
    specified for the `partition` parameter, the partition is left
    unspecified (i.e., all of the data is used). Also, a limit
    (concerning the cursor that is created) can be specified via the
    `limit` parameter.

    Returns a dictionary containing keys whose values are: a mapping
    between ID strings and label values and a frequency distribution of
    label values. If `bin_ranges` is specified, all labels will be
    converted using that information.

    :param db: MongoDB collection
    :type db: collection
    :param label: label used for prediction
    :type label: str
    :param games: list of game IDs
    :type games: list
    :param partition: data partition, i.e., 'train', 'test', or 'all'
                      to use all of the data (default: 'test')
    :type partition: str
    :param bin_ranges: list of ranges
    :type bin_ranges: list or None
    :param lognormal: transform raw label values using `ln` (default:
                      False)
    :type lognormal: bool
    :param limit: cursor limit (defaults to 0, which signifies no
                                limit)
    :type limit: int

    :returns: dictionary containing `id_strings_labels_dict` and
              `labels_counter` keys, which are mapped to a dictionary
              of ID strings mapped to labels and a Counter object
              representing the frequency distribution of the label
              values, respectively
    :rtype: dict

    :raises ValueError: if unrecognized games were found in the input
                        or no reviews were found for the combination of
                        game, partition, etc., or the `bin_ranges`
                        value is invalid
    """

    # Check `partition`, `label`, and `limit` parameter values
    if partition != 'test' and not partition in ['train', 'all']:
        raise ValueError('The only values recognized for the "partition" '
                         'parameter are "test", "train", and "all" (for no '
                         'partition, i.e., all of the data).')
    if not label in LABELS:
        raise ValueError('Unrecognized label: {0}'.format(label))
    if limit != 0 and (type(limit) != int or limit < 0):
        raise ValueError('"limit" must be a positive integer.')

    # Make sure the games are in the list of valid games
    if any(not game in APPID_DICT for game in games):
        raise ValueError('All or some of the games in the given list of '
                         'games, {0}, are not in list of available games'
                         .format(', '.join(games)))

    # Generate a query
    if len(games) == 1:
        query = {'game': games[0]}
    else:
        query = {'game': {'$in': games}}
    if partition != 'all':
        query['partition'] = partition

    # Create a MongoDB cursor on the collection
    kwargs = {'limit': limit}
    proj = {'nlp_features': 0}
    cursor = db.find(query, proj, **kwargs)

    # Get review documents (only including label + ID string)
    samples = []
    for doc in cursor:
        # Apply lognormal transformation and/or multiplication by 100
        # if this is a percentage value
        label_value = compute_label_value(get_label_in_doc(doc, label),
                                          label, lognormal=lognormal)

        # Skip label values that are equal to None
        if label_value == None:
            continue

        if bin_ranges:
            # Validate `bin_ranges`
            validate_bin_ranges(bin_ranges)
            label_value = get_bin(bin_ranges, label_value)

        samples.append({'id_string': doc['id_string'], label: label_value})

    # Raise exception if no review documents were found
    if not samples:
        raise ValueError('No review documents were found!')

    # Return dictionary containing a key 'id_strings_labels_dict'
    # mapped to a dictionary mapping ID strings to label values and a
    # key 'labels_counter' mapped to a Counter object of the label
    # values
    id_strings_labels_dict = {doc['id_string']: doc[label] for doc in samples}
    labels_counter = Counter([doc[label] for doc in samples])
    return dict(id_strings_labels_dict=id_strings_labels_dict,
                labels_counter=labels_counter)


def get_label_in_doc(doc: dict, label: str):
    """
    Return the value for a label in a sample document and return None
    if not in the document.

    :param doc: sample document dictionary
    :type doc: dict
    :param label: document key that functions as a label
    :type label: str

    :returns: label value
    :rtype: int/float/str/None

    :raises ValueError: if the label is not in `LABELS`
    """

    if not label in LABELS:
        raise ValueError('Unrecognized label: {0}'.format(label))

    if label in doc:
        return doc[label]

    achievement_progress_string = 'achievement_progress'
    if achievement_progress_string in doc:
        achievement_progress = doc[achievement_progress_string]
        if label in achievement_progress:
            return achievement_progress[label]


def evenly_distribute_samples(db: collection, label: str, games: list,
                              partition: str = 'test',
                              bin_ranges: list = None,
                              lognormal: bool = False) -> str:
    """
    Generate ID strings from data samples that, altogether, form a
    maximally evenly-distributed set of samples with respect to the
    the prediction label or to the "binned" prediction label,
    specifically for cases when a small subset of the total data is
    being used. If `bin_ranges` is specified, all labels will be
    converted using it.

    By default, the 'test' partition is used, but the 'train' partition
    can be specified via the `partition` parameter. If 'all' is
    specified for the `partition` parameter, the partition is left
    unspecified (i.e., all of the data is used).

    :param db: MongoDB collection
    :type db: collection
    :param label: label used for prediction
    :type label: str
    :param games: list of game IDs
    :type games: list
    :param partition: data partition, i.e., 'train', 'test', or 'all'
                      to use all of the data (default: 'test')
    :type partition: str
    :param bin_ranges: list of ranges
    :type bin_ranges: list or None
    :param lognormal: transform raw label values using `ln` (default:
                      False)
    :type lognormal: bool

    :yields: ID string
    :ytype: str
    """

    # Check `partition` parameter value
    if partition != 'test' and not partition in ['train', 'all']:
        raise ValueError('The only values recognized for the "partition" '
                         'parameter are "test", "train", and "all" (for no '
                         'partition, i.e., all of the data).')

    # Get dictionary of ID strings mapped to labels and a frequency
    # distribution of the labels
    distribution_dict = distributional_info(db, label, games,
                                            partition=partition,
                                            bin_ranges=bin_ranges,
                                            lognormal=lognormal)

    # Create a maximally evenly-distributed list of samples with
    # respect to label
    labels_id_strings_lists_dict = dict()
    for label_value in distribution_dict['labels_counter']:
        labels_id_strings_lists_dict[label_value] = \
            [_id for _id, _label
             in distribution_dict['id_strings_labels_dict'].items()
             if _label == label_value]
    i = 0
    while i < len(distribution_dict['id_strings_labels_dict']):

        # For each label value, pop off an ID string, if available
        for label_value in distribution_dict['labels_counter']:
            if labels_id_strings_lists_dict[label_value]:
                yield labels_id_strings_lists_dict[label_value].pop()
                i += 1
