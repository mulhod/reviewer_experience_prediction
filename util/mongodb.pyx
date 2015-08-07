'''
:author: Matt Mulholland
:date: May 5, 2015

Module of code related to the MongoDB database that holds all of the review
data.

The insert_train_test_reviews function gets all suitable, English-language
reviews for a given data-set (at the provided file-path) and inserts them into
the the MongoDB database ('reviews_project') under the 'reviews' collection.
'''
import logging
logger = logging.getLogger()
loginfo = logger.info
logdebug = logger.debug
logwarn = logger.warning
logerr = logger.error
from sys import exit
from math import ceil
from data import APPID_DICT
from random import (randint,
                    shuffle,
                    seed)
from json import JSONDecoder
from os.path import (basename,
                     splitext)
# Make local binding to JSONEncoder method attribute
json_decoder = JSONDecoder()
json_decode = json_decoder.decode
from pymongo.errors import (DuplicateKeyError,
                            BulkWriteError)
from util.datasets import (get_and_describe_dataset,
                           get_bin_ranges,
                           get_bin)


def insert_train_test_reviews(reviewdb, file_path, int max_size,
                              float percent_train, bins=0, bin_factor=1.0,
                              describe=False, just_describe=False):
    '''
    Insert training/test set reviews into the MongoDB database and optionally
    generate a report and graphs describing the filtering mechanisms.

    :param reviewdb: MongoDB reviews collection
    :type reviewdb: pymongo.MongoClient object
    :param file_path: path to game reviews file
    :type file_path: str
    :param max_size: maximum size of training/test set combination (in number
                     of reviews)
    :type max_size: int
    :param percent_train: percent of training/test combination that should be
                          reserved for the training set
    :type percent_train: float/int
    :param bins: number of bins in which to sub-divide the hours played
                 values (defaults to 0, in which case the values will be left
                 as they are)
    :type bins: int
    :param bin_factor: if the bins parameter is set to something other than
                       the default, the size of the bins relative to each
                       other will be governed by bin_factor, i.e., the size
                       of the bins in terms of the ranges of values will
                       be smaller for the bins that have a lot of instances
                       and will increase in size for the more
                       sparsely-populated bins
    :type bin_factor: float
    :param describe: describe data-set, outputting a report with some
                     descriptive statistics and histograms representing review
                     length and hours played distributions
    :type describe: boolean
    :param just_describe: only get the reviews and generate the statistical
                          report
    :type just_describe: boolean
    :returns: None
    '''

    # Seed the random number generator (hopefully ensuring that repeated
    # iterations will result in the same behavior from random.randint and
    # random.shuffle)
    seed(1)

    game = splitext(basename(file_path))[0]
    appid = APPID_DICT[game]

    loginfo('Inserting reviews from {}...'.format(game))
    if bins:
        loginfo('Dividing the hours played values into {} bins with a bin '
                'factor of {}...'.format(bins,
                                         bin_factor))

    # Make sense of arguments
    if (describe
        and just_describe):
        logwarn('If the just_describe and describe keyword arguments are set '
                'to True, just_describe wins out, i.e., the report will be '
                'generated, but no reviews will be inserted into the '
                'database.')

    # Get list of all reviews represented as dictionaries with 'review' and
    # 'hours' keys and get the filter values
    dataset = get_and_describe_dataset(file_path,
                                       report=(describe
                                               or just_describe))
    reviews = dataset['reviews']
    logdebug('Number of original, English language reviews collected: {}'
             .format(dataset['orig_total_reviews']))
    cdef float maxl = dataset['maxl']
    cdef float minl = dataset['minl']
    cdef float maxh = dataset['maxh']
    cdef float minh = dataset['minh']
    logdebug('Maximum length = {}'.format(dataset['maxl']))
    logdebug('Minimum length = {}'.format(dataset['minl']))
    logdebug('Maximum amount of hours played = {}'.format(dataset['maxh']))
    logdebug('Minimum amount of hours played = {}'.format(dataset['minh']))

    # If the hours played values are to be divided into bins, get the range
    # that each bin maps to
    if bins:
        bin_ranges = get_bin_ranges(minh,
                                    maxh,
                                    bins,
                                    bin_factor)
    else:
        bin_ranges = False

    # Shuffle the list of reviews so that we randomize it
    shuffle(reviews)

    # Get the training and test sets and the set of extra reviews (which
    # might get pulled in later if necessary)
    cdef int num_reviews = len(reviews)
    if num_reviews > max_size:
        train_test_reviews = reviews[:max_size]
    else:
        train_test_reviews = reviews[:num_reviews]
        max_size = num_reviews
    remaining_reviews = reviews[max_size:]

    # Divide the selected reviews into training/test sets
    cdef int training_set_size = \
        <int>ceil(len(train_test_reviews)*(percent_train/100.0))
    training_reviews = train_test_reviews[:training_set_size + 1]
    test_reviews = train_test_reviews[training_set_size + 1:]
    logdebug('Number of training set reviews: {}'
             .format(len(training_reviews)))
    logdebug('Number of test set reviews: {}'.format(len(test_reviews)))
    logdebug('Number of extra reviews: {}'.format(len(remaining_reviews)))
    logdebug('NOTE: It is possible that fewer reviews get inserted into the '
             'DB for the training set or test set if there are errors during '
             'insertion and there are no replacement reviews to substitute in'
             '.')

    if not just_describe:

        # Initialize a bulk writer and add insertion operations for training,
        # test, and extra reviews and then execture the operations and print
        # out some information about how many entries were inserted, etc.
        bulk = reviewdb.initialize_unordered_bulk_op()

        # Training set reviews
        add_bulk_inserts_for_partition(bulk,
                                       training_reviews,
                                       game,
                                       appid,
                                       'training',
                                       bins=bin_ranges)

        # Test set reviews
        add_bulk_inserts_for_partition(bulk,
                                       test_reviews,
                                       game, appid,
                                       'test',
                                       bins=bin_ranges)

        # Extra reviews
        add_bulk_inserts_for_partition(bulk,
                                       remaining_reviews,
                                       game,
                                       appid,
                                       'extra',
                                       bins=bin_ranges)

        # Execute bulk insert operations
        try:
            result = bulk.execute()
        except BulkWriteError as bwe:
            logger.debug(bwe.details)
            exit(1)
        logdebug(repr(result))

        # Print out some information about how many reviews were added
        train_inserts = reviewdb.find({'appid': appid,
                                       'partition': 'training'}).count()
        test_inserts = reviewdb.find({'appid': appid,
                                      'partition': 'test'}).count()
        extra_inserts = reviewdb.find({'appid': appid,
                                       'partition': 'extra'}).count()
        logdebug('Inserted {} training set reviews, {} test set reviews, and '
                 '{} extra reviews...'.format(train_inserts,
                                              test_inserts,
                                              extra_inserts))


cdef add_bulk_inserts_for_partition(bulk_writer, rdicts, game, appid,
                                    partition_id, bins=False):
    '''
    Add insert operations to a bulk writer.

    :param bulk_writer: a bulk writer instance, to which we can add insertion
                        operations that will be executed later on
    :type bulk_writer: pymongo.bulk.BulkOperationBuilder instance
    :param rdicts: list of review dictionaries
    :type rdicts: list of dict
    :param game: name of game
    :type game: str
    :param appid: appid string, ID number of game
    :type appid: str
    :param partition_id: name/ID of partition, i.e., 'test', 'training',
                         'extra'
    :type partition_id: str
    :param bins: False (i.e., if a converted hours value should not also be
                 inserted) or a list of 2-tuples containing floating point
                 numbers representing the beginning of a range (actually, the
                 lower, non-inclusive bound of the range) and the end (the
                 upper, inclusive bound of the range) (default: False)
    :type bins: False or list of 2-tuples of floats
    :returns: None
    '''

    for rd in rdicts:
        # Add keys for the partition (i.e., "extra"), the game's name, and the
        # appid
        rd['game'] = game
        rd['appid'] = appid
        rd['partition'] = partition_id

        if bins:
            _bin = get_bin(bins,
                           rd['total_game_hours'])
            if _bin > -1:
                rd['total_game_hours_bin'] = _bin
            else:
                logerr('The hours played value ({}) did not seem to fall '
                       'within any of the bin ranges.\n\nBin ranges\n{}\n'
                       'Exiting.'.format(rd['total_game_hours_bin'],
                                         repr(bins)))
                exit(1)

        try:
            bulk_writer.insert(rd)
        except DuplicateKeyError as e:
            logwarn('Encountered DuplicateKeyError. Throwing out the '
                    'following review:\n{}'.format(rd))


def get_review_features_from_db(db, _id):
    '''
    Collect the features from the database for a given review and return the
    decoded value.

    :param db: database
    :type db: MongoClient instance
    :param _id: ID string for review
    :type _id: pymong.bson.objectid.ObjectId
    :returns: dict if features were found; None otherwise
    '''

    features_doc = db.find_one({'_id': _id},
                               {'_id': 0,
                                'features': 1})
    return json_decode(features_doc.get('features')) if features_doc else None
