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
from time import sleep
from random import (seed,
                    randint,
                    shuffle)
from data import APPID_DICT
from json import (loads,
                  dumps)
from os.path import (basename,
                     splitext)
from pymongo import MongoClient
from util.datasets import (get_bin,
                           get_bin_ranges,
                           get_and_describe_dataset)
from pymongo.errors import (AutoReconnect,
                            BulkWriteError,
                            ConnectionFailure,
                            DuplicateKeyError)

def connect_to_db(port, tries=10):
    '''
    Connect to database and return a collection object.

    :param port: Mongo database port
    :type port: int (or str)
    :param tries: number of times to try to connect client (default: 10)
    :type tries: int
    :returns: pymongo.collection.Collection object
    '''

    connection_string = 'mongodb://localhost:{}'.format(port)
    while tries > 0:
        tries -= 1
        try:
            connection = MongoClient(connection_string,
                                     max_pool_size=None,
                                     connectTimeoutMS=100000,
                                     socketKeepAlive=True)
        except ConnectionFailure as e:
            if tries == 0:
                logerr('Unable to connect client to Mongo server at {}. '
                       'Exiting.'.format(connection_string))
                exit(1)
            else:
                logwarn('Unable to connect client to Mongo server at {}. Will'
                        ' try {} more time{}...'.format(connection_string,
                                                        tries,
                                                        's' if tries > 1
                                                            else ''))

    db = connection['reviews_project']
    return db['reviews']


def create_game_cursor(db, game_id, data_partition, int batch_size):
    '''
    Create Cursor object with given game and partition to iterate through game
    documents.

    :param db: Mongo reviews collection
    :type db: pymongo.collection.Collection
    :param game_id: game ID
    :type game_id: str
    :param data_partition: data partition, i.e., 'training', 'test', etc.
    :type data_partition: str
    :param batch_size: size of each batch that the cursor returns
    :type batch_size: int
    :returns: pymongo.cursor.Cursor object
    '''

    game_cursor = db.find({'game': game_id,
                           'partition': data_partition},
                          {'features': 0,
                           'game': 0,
                           'partition': 0},
                          timeout=False)
    game_cursor.batch_size = batch_size

    if game_cursor.count() == 0:
        logerr('No matching documents were found in the MongoDB collection in'
               ' the {} partition for game {}. Exiting.'
               .format(data_partition,
                       game_id))
        exit(1)

    return game_cursor


def insert_train_test_reviews(reviewdb, file_path, int max_size,
                              float percent_train, bins=0, bin_factor=1.0,
                              describe=False, just_describe=False):
    '''
    Insert training/test set reviews into the MongoDB database and optionally
    generate a report and graphs describing the filtering mechanisms.

    :param reviewdb: Mongo reviews collection
    :type reviewdb: pymongo.collection.Collection object
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
    # 'total_game_hours' keys and get the filter values
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
    # that each bin maps to and add values for the number of bins, the bin
    # ranges, and the bin factor to the review dictionaries
    if bins:
        bin_ranges = get_bin_ranges(minh,
                                    maxh,
                                    bins,
                                    bin_factor)
        bin_dict = dict(nbins=bins,
                        bin_factor=bin_factor,
                        bin_ranges=bin_ranges)
        [review.update(bin_dict) for review in reviews]
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
                       'Exiting.'.format(rd['total_game_hours'],
                                         repr(bins)))
                exit(1)

        try:
            bulk_writer.insert(rd)
        except DuplicateKeyError as e:
            logwarn('Encountered DuplicateKeyError. Throwing out the '
                    'following review:\n{}'.format(rd))


def get_review_features_from_db(db, _id):
    '''
    Collect the features from the database collection for a given review and
    return the decoded value.

    :param db: Mongo reviews collection
    :type db: pymongo.collection.Collection object
    :param _id: ID string for review
    :type _id: pymong.bson.objectid.ObjectId
    :returns: dict if features were found; None otherwise
    '''

    features_doc = db.find_one({'_id': _id},
                               {'_id': 0,
                                'features': 1})
    return loads(features_doc.get('features')) if features_doc else None


def update_db(db_update, _id, feats, _binarize):
    '''
    Update Mongo database document with extracted features.

    :param db_update: bound method Collection.update of Mongo collection
    :type db_update: method
    :param _id: database document's Object ID
    :type _id: bson.objectid.ObjectId object
    :param feats: dictionary of features
    :type feats: dict
    :param _binarize: whether or not the features are binarized
    :type _binarize: boolean
    :returns: None
    '''

    cdef int tries = 0
    while tries < 5:
        try:
            db_update({'_id': _id},
                      {'$set': {'features': dumps(feats),
                                'binarized': _binarize}})
            break
        except AutoReconnect:
            logwarn('Encountered AutoReconnect failure, attempting to '
                    'reconnect automatically after 20 seconds...')
            tries += 1
            if tries >= 5:
                logerr('Unable to update database even after 5 tries. '
                       'Exiting.')
                exit(1)
            sleep(20)
