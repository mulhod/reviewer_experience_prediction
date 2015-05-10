'''
@author Matt Mulholland
@date 05/05/2015

Module of code related to the MongoDB database that holds all of the review data.

The insert_train_test_reviews function gets all suitable, English-language reviews for a given data-set (at the provided file-path) and inserts them into the the MongoDB database ('reviews_project') under the 'reviews' collection.
'''
import sys
from math import ceil
from pprint import pprint
from data import APPID_DICT
from os.path import basename
from random import randint, shuffle, seed
from util.datasets import get_and_describe_dataset, get_bin_ranges, get_bin
from pymongo.errors import DuplicateKeyError, BulkWriteError


def insert_train_test_reviews(reviewdb, file_path, int max_size,
                              float percent_train, bins=0, describe=False,
                              just_describe=False):
    '''
    Insert training/test set reviews into the MongoDB database and optionally generate a report and graphs describing the filtering mechanisms.

    :param reviewdb: MongoDB reviews collection
    :type reviewdb: pymongo.MongoClient object
    :param file_path: path to game reviews file
    :type file_path: str
    :param max_size: maximum size of training/test set combination (in number of reviews)
    :type max_size: int
    :param percent_train: percent of training/test combination that should be reserved for the training set
    :type percent_train: float/int
    :param bins: number of bins in which to sub-divide the hours played values (defaults to 0, in which case the values will be left as they are)
    :type bins: int
    :param describe: describe data-set, outputting a report with some descriptive statistics and histograms representing review length and hours played distributions
    :type describe: boolean
    :param just_describe: only get the reviews and generate the statistical report
    :type just_describe: boolean
    :returns: None
    '''

    # Seed the random number generator (hopefully ensuring that repeated
    # iterations will result in the same behavior from random.randint and
    # random.shuffle)
    seed(1)

    game = basename(file_path)[:-4]
    appid = APPID_DICT[game]

    sys.stderr.write('Inserting reviews from {}...\n'.format(game))
    if bins:
        sys.stderr.write('Dividing the hours played values into {} bins...' \
                         '\n'.format(bins))

    # Make sense of arguments
    if describe and just_describe:
        sys.stderr.write('WARNING: If the just_describe and describe ' \
                         'keyword arguments are set to True, just_describe ' \
                         'wins out, i.e., the report will be generated, but' \
                         ' no reviews will be inserted.\n')

    # Get list of all reviews represented as dictionaries with 'review' and
    # 'hours' keys and get the filter values
    dataset = get_and_describe_dataset(file_path,
                                       report=(describe or just_describe))
    reviews = dataset['reviews']
    sys.stderr.write('Number of original, English language reviews ' \
                     'collected: {}\n'.format(
                                         dataset['orig_total_reviews']))
    cdef float maxl = dataset['maxl']
    cdef float minl = dataset['minl']
    cdef float maxh = dataset['maxh']
    cdef float minh = dataset['minh']
    sys.stderr.write('Maximum length = {}\n' \
                     'Minimum length = {}\n' \
                     'Maximum amount of hours played = {}\n' \
                     'Minimum amount of hours played = {}\n' \
                     '\n'.format(dataset['maxl'],
                                 dataset['minl'], 
                                 dataset['maxh'],
                                 dataset['minh']))

    # If the hours played values are to be divided into bins, get the range
    # that each bin maps to
    if bins:
        bin_ranges = get_bin_ranges(minh,
                                    maxh,
                                    bins)
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
    sys.stderr.write('Number of training set reviews: ' \
                     '{}\n'.format(len(training_reviews)))
    sys.stderr.write('Number of test set reviews:' \
                     ' {}\n'.format(len(test_reviews)))
    sys.stderr.write('Number of extra reviews:' \
                     ' {}\n'.format(len(remaining_reviews)))
    sys.stderr.write('NOTE: It is possible that fewer reviews get ' \
                     'inserted into the DB for the training set or test ' \
                     'set if there are errors during insertion and there' \
                     ' are no replacement reviews to substitute in.\n\n')

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
            pprint(bwe.details,
                   stream=sys.stderr)
            sys.exit(1)
        pprint(result,
               stream=sys.stderr)

        # Print out some information about how many reviews were added
        train_inserts = reviewdb.find({'appid': appid,
                                       'partition': 'training'}).count()
        test_inserts = reviewdb.find({'appid': appid,
                                      'partition': 'test'}).count()
        extra_inserts = reviewdb.find({'appid': appid,
                                       'partition': 'extra'}).count()
        sys.stderr.write('Inserted {} training set reviews, {} test set ' \
                         'reviews, and {} extra reviews...\n\n' \
                         '\n'.format(train_inserts,
                                     test_inserts,
                                     extra_inserts))


cdef add_bulk_inserts_for_partition(bulk_writer, rdicts, game, appid,
                                    partition_id, bins=False):
    '''
    Add insert operations to a bulk writer.

    :param bulk_writer: a bulk writer instance, to which we can add insertion operations that will be executed later on
    :type bulk_writer: pymongo.bulk.BulkOperationBuilder instance
    :param rdicts: list of review dictionaries
    :type rdicts: list of dict
    :param game: name of game
    :type game: str
    :param appid: appid string, ID number of game
    :type appid: str
    :param partition_id: name/ID of partition, i.e., 'test', 'training', 'extra'
    :type partition_id: str
    :param bins: False (i.e., if a converted hours value should not also be inserted) or a list of 2-tuples containing floating point numbers representing the beginning of a range (actually, the lower, non-inclusive bound of the range) and the end (the upper, inclusive bound of the range) (default: False)
    :type bins: False or list of 2-tuples of floats
    :returns: None
    '''

    for rd in rdicts:

        # Add keys for the partition ("extra"), the game's name, and the
        # appid
        rd['game'] = game
        rd['appid'] = appid
        rd['partition'] = partition_id

        if bins:
            _bin = get_bin(bins,
                           rd['hours'])
            if _bin > -1:
                rd['hours_bin'] = _bin
            else:
                sys.exit('WARNING: The hours played value ({}) did not seem' \
                         ' to fall within any of the bin ranges.\n\nBin ' \
                         'ranges:\n\n{}\n\nExiting.\n'.format(rd['hours_bin'],
                                                              repr(bins)))

        try:
            bulk_writer.insert(rd)
        except DuplicateKeyError as e:
            sys.stderr.write('WARNING: Encountered DuplicateKeyError. ' \
                             'Throwing out following ' \
                             'review:\n\n{}\n\n'.format(rd))