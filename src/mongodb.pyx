"""
:author: Matt Mulholland
:date: May 5, 2015

Module of code related to the MongoDB database that holds all of the
review data.

The `insert_train_test_reviews` function gets all suitable,
English-language reviews for a given data-set (at the provided
file-path) and inserts them into the the MongoDB database
('reviews_project') under the 'reviews' collection.
"""
import logging
from sys import exit
from math import ceil
from time import sleep
from random import (seed,
                    randint,
                    shuffle)
from json import dumps
from os.path import (basename,
                     splitext)

import numpy as np
from bson import BSON
from pymongo import (cursor,
                     collection,
                     MongoClient)
from bson.objectid import ObjectId
from pymongo.bulk import BulkOperationBuilder
from pymongo.errors import (AutoReconnect,
                            BulkWriteError,
                            ConnectionFailure,
                            DuplicateKeyError)

from data import APPID_DICT
from src.features import bulk_extract_features
from src.datasets import (get_bin,
                          get_bin_ranges,
                          validate_bin_ranges,
                          get_and_describe_dataset)

# Logging-related
logger = logging.getLogger()
loginfo = logger.info
logdebug = logger.debug
logwarn = logger.warning
logerr = logger.error

# BSON encoding
bson_encode = BSON.encode

def connect_to_db(host: str = 'localhost', port: int = 27017,
                  tries: int = 10) -> collection:
    """
    Connect to database and return a collection object.

    :param host: host-name of MongoDB server
    :type host: str
    :param port: Mongo database port
    :type port: int
    :param tries: number of times to try to connect client (default:
                  10)
    :type tries: int

    :rtype: MongoDB collection
    :returns: collection

    :raises ConnectionFailure: if ConnectionFailure is encountered more
                               than `tries` times
    """

    mongo_url = 'mongodb://{0}:{1}'.format(host, port)
    while tries > 0:
        tries -= 1
        try:
            connection = MongoClient(mongo_url, max_pool_size=None,
                                     connectTimeoutMS=100000,
                                     socketKeepAlive=True)
        except ConnectionFailure as e:
            if tries == 0:
                logerr('Unable to connect client to Mongo server at {0}.'
                       .format(mongo_url))
                raise e
            else:
                logwarn('Unable to connect client to Mongo server at {0}. '
                        'Will try {1} more time{2}...'
                        .format(mongo_url, tries, 's' if tries > 1 else ''))

    db = connection['reviews_project']
    return db['reviews']


def create_game_cursor(db: collection,
                       game_id: str,
                       data_partition: str,
                       int batch_size) -> cursor:
    """
    Create Cursor object with given game and partition to iterate
    through game documents.

    :param db: Mongo reviews collection
    :type db: collection
    :param game_id: game ID
    :type game_id: str
    :param data_partition: data partition, i.e., 'training', 'test',
                           etc.; can alternatively be the value "all"
                           for all partitions
    :type data_partition: str
    :param batch_size: size of each batch that the cursor returns
    :type batch_size: int

    :returns: cursor on a MongoDB collection
    :rtype: cursor

    :raises ValueError: if no matching documents were found
    """

    if data_partition == 'all':
        game_cursor = db.find({'game': game_id},
                              {'nlp_features': 0, 'game': 0, 'partition': 0},
                              timeout=False)
    else:
        game_cursor = db.find({'game': game_id, 'partition': data_partition},
                              {'nlp_features': 0, 'game': 0, 'partition': 0},
                              timeout=False)
    game_cursor.batch_size = batch_size

    if game_cursor.count() == 0:
        error_msg = None
        if data_partition == 'all':
            error_msg = ('No matching documents were found in the MongoDB '
                         'collection for game {0}.'.format(game_id))
            logerr(error_msg)
        else:
            error_msg = ('No matching documents were found in the MongoDB '
                         'collection in the {0} partition for game {1}.'
                         .format(data_partition, game_id))
            logerr(error_msg)
        raise ValueError(error_msg)

    return game_cursor


def insert_train_test_reviews(db: collection,
                              file_path: str,
                              int max_size,
                              float percent_train,
                              bins: int = 0,
                              bin_factor: float = 1.0,
                              describe: bool = False,
                              just_describe: bool = False,
                              reports_dir: str = None) -> None:
    """
    Insert training/test set reviews into the MongoDB database and
    optionally generate a report and graphs describing the filtering
    mechanisms.

    :param db: MongoDB collection
    :type db: collection
    :param file_path: path to game reviews file
    :type file_path: str
    :param max_size: maximum size of training/test set combination (in
                     number of reviews)
    :type max_size: int
    :param percent_train: percent of training/test combination that
                          should be reserved for the training set
    :type percent_train: float
    :param bins: number of bins in which to sub-divide the hours played
                 values (defaults to 0, in which case the values will
                 be left as they are)
    :type bins: int
    :param bin_factor: if the bins parameter is set to something other
                       than the default, the size of the bins relative
                       to each other will be governed by bin_factor,
                       i.e., the size of the bins in terms of the
                       ranges of values will be smaller for the bins
                       that have a lot of instances and will increase
                       in size for the more sparsely-populated bins
    :type bin_factor: float
    :param describe: describe data-set, outputting a report with some
                     descriptive statistics and histograms representing
                     review length and hours played distributions
    :type describe: bool
    :param just_describe: only get the reviews and generate the
                          statistical report
    :type just_describe: bool
    :param reports_dir: path to directory to which report files should
                        be written
    :type reports_dir: str

    :returns: None
    :rtype: None
    """

    # Seed the random number generator (hopefully ensuring that
    # repeated iterations will result in the same behavior from
    # random.randint and random.shuffle)
    seed(1)

    game = splitext(basename(file_path))[0]
    appid = APPID_DICT[game]

    loginfo('Inserting reviews from {0}...'.format(game))
    if bins:
        loginfo('Dividing the hours played values into {0} bins with a bin '
                'factor of {1}...'.format(bins, bin_factor))

    # Make sense of arguments
    if describe and just_describe:
        logwarn('If the just_describe and describe keyword arguments are set '
                'to True, just_describe wins out, i.e., the report will be '
                'generated, but no reviews will be inserted into the '
                'database.')

    # Get list of all reviews represented as dictionaries
    reviews = get_and_describe_dataset(file_path,
                                       report=describe or just_describe,
                                       reports_dir=reports_dir)

    # If the hours played values are to be divided into bins, get the
    # range that each bin maps to and add values for the number of
    # bins, the bin ranges, and the bin factor to the review
    # dictionaries
    if bins:
        bin_ranges = get_bin_ranges(min([r['total_game_hours'] for r in reviews]),
                                    max([r['total_game_hours'] for r in reviews]),
                                    bins,
                                    bin_factor)
        bin_dict = dict(nbins=bins, bin_factor=bin_factor, bin_ranges=bin_ranges)
        [review.update(bin_dict) for review in reviews]
    else:
        bin_ranges = False

    # Shuffle the list of reviews so that we randomize it
    shuffle(reviews)

    # Get the training and test sets and the set of extra reviews
    # (which might get pulled in later if necessary)
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
    logdebug('Number of training set reviews: {0}'.format(len(training_reviews)))
    logdebug('Number of test set reviews: {0}'.format(len(test_reviews)))
    logdebug('Number of extra reviews: {0}'.format(len(remaining_reviews)))
    logdebug('NOTE: It is possible that fewer reviews get inserted into the '
             'DB for the training set or test set if there are errors during '
             'insertion and there are no replacement reviews to substitute in'
             '.')

    if not just_describe:
        # Initialize a bulk writer and add insertion operations for
        # training, test, and extra reviews and then execture the
        # operations and print out some information about how many
        # entries were inserted, etc.
        bulk = db.initialize_unordered_bulk_op()

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
                                       game,
                                       appid,
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
            logdebug(bwe.details)
            exit(1)
        logdebug(repr(result))

        # Print out some information about how many reviews were added
        train_inserts = db.find({'appid': appid, 'partition': 'training'}).count()
        test_inserts = db.find({'appid': appid, 'partition': 'test'}).count()
        extra_inserts = db.find({'appid': appid, 'partition': 'extra'}).count()
        logdebug('Inserted {0} training set reviews, {1} test set reviews, '
                 'and {2} extra reviews...'.format(train_inserts, test_inserts,
                                                   extra_inserts))


cdef add_bulk_inserts_for_partition(bulk_writer: BulkOperationBuilder,
                                    rdicts: list,
                                    game: str,
                                    appid: str,
                                    partition_id: str,
                                    bins: list = False):
    """
    Add insert operations to a bulk writer.

    :param bulk_writer: a bulk writer instance, to which we can add
                        insertion operations that will be executed
                        later on
    :type bulk_writer: BulkOperationBuilder
    :param rdicts: list of review dictionaries
    :type rdicts: list
    :param game: name of game
    :type game: str
    :param appid: appid string, ID number of game
    :type appid: str
    :param partition_id: name/ID of partition, i.e., 'test',
                         'training', 'extra'
    :type partition_id: str
    :param bins: False (i.e., if a converted hours value should not
                 also be inserted) or a list of 2-tuples containing
                 floating point numbers representing the beginning of a
                 range and the end (default: False)
    :type bins: False or list

    :returns: None
    :rtype: None

    :raises ValueError: if `bins` is invalid (only if it is specified)
                        or the hours played value seems to be invalid
    """

    for rd in rdicts:

        # Add keys for the partition (i.e., "extra"), the game's name,
        # and the appid
        rd['game'] = game
        rd['appid'] = appid
        rd['partition'] = partition_id

        if bins:

            # Validate `bins`
            try:
                validate_bin_ranges(bins)
            except ValueError as e:
                error_msg = '"bins" could not be validated.'
                logerr(error_msg)
                logerr(e)
                raise ValueError(error_msg)
            _bin = get_bin(bins, rd['total_game_hours'])

            if _bin > -1:
                rd['total_game_hours_bin'] = _bin
            else:
                error_msg = ('The hours played value ({0}) did not seem to '
                             'fall within any of the bin ranges.\n\nBin '
                             'ranges\n{1}'
                             .format(rd['total_game_hours'], repr(bins)))
                logerr(error_msg)
                raise ValueError(error_msg)

        try:
            bulk_writer.insert(rd)
        except DuplicateKeyError as e:
            logwarn('Encountered DuplicateKeyError. Throwing out the '
                    'following review:\n{0}'.format(rd))


def generate_update_query(update_dict: dict, binarized_features: bool = True) -> dict:
    """
    Generate an update query in the form needed for the MongoDB
    updates.

    :param update_dict: dictionary containing an `_id` field and a
                        `features` field
    :type update_dict: dict
    :param binarized_features: value representing whether or not the
                               features were binarized
    :type binarized_features: bool

    :returns: update query dictionary
    :rtype: dict
    """

    return {'$set': {'nlp_features': bson_encode(update_dict['features']),
                     'binarized': binarized_features,
                     'id_string': str(update_dict['_id'])}}


def bulk_extract_features_and_update_db(db: collection,
                                        game: str,
                                        partition: str = 'all',
                                        reuse_nlp_feats: bool = True,
                                        use_binarized_nlp_feats: bool = True,
                                        lowercase_text: bool = True,
                                        lowercase_cngrams: bool = False,
                                        update_batch_size: int = 100) -> int:
    """
    Extract NLP features from the review texts for a given game/data
    partition and update the database.

    :param db: MongoDB collection
    :type db: collection
    :param game: game ID
    :type game: str
    :param partition: partition of the data (leave unspecified or use
                      'all' to leave the partition unspecified)
    :type partition: str
    :param reuse_nlp_feats: reuse NLP features from database instead of
                            extracting them all over again (default:
                            True)
    :type reuse_nlp_feats: bool
    :param use_binarized_nlp_feats: use binarized NLP features
                                    (default: True)
    :type use_binarized_nlp_feats: bool
    :param lowercase_text: whether or not to lower-case the review
                           text (default: True)
    :type lowercase_text: bool
    :param lowercase_cngrams: whether or not to lower-case the
                              character n-grams (default: False)
    :type lowercase_cngrams: bool
    :param update_batch_size: size of update batches (default: 100)
    :type update_batch_size: int

    :returns: total number of successful updates
    :rtype: int
    """

    if not partition in ['all', 'test', 'training', 'extra']:
        raise ValueError('"partition" must be in the following list of '
                         'values: "all", "test", "training", "extra".')

    bulk = db.initialize_unordered_bulk_op()
    cdef int batch_size = 1000
    game_cursor = create_game_cursor(db,
                                     game,
                                     partition,
                                     batch_size)
    updates = bulk_extract_features(db,
                                    game_cursor,
                                    reuse_nlp_feats=reuse_nlp_feats,
                                    use_binarized_nlp_feats=use_binarized_nlp_feats,
                                    lowercase_text=lowercase_text,
                                    lowercase_cngrams=lowercase_cngrams)
    NO_MORE_UPDATES = False
    cdef int TOTAL_UPDATES = 0
    cdef int i
    while not NO_MORE_UPDATES:

        # Add updates to the bulk update builder up until reaching the
        # the batch size limit (or until we run out of data)
        i = 0
        while i < update_batch_size:
            try:
                update = next(updates)
            except StopIteration:
                NO_MORE_UPDATES = True
                break
            (bulk
             .find({'_id': update['_id']})
             .update(generate_update_query(update,
                                           binarized_features=use_binarized_nlp_feats)))
            i += 1
        TOTAL_UPDATES += i

        # Execute bulk update operations
        try:
            result = bulk.execute()
        except BulkWriteError as bwe:
            logerr(bwe.details)
            raise bwe
        logdebug(repr(result))

    return TOTAL_UPDATES
