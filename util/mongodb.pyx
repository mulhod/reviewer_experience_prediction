'''
@author Matt Mulholland
@date 05/05/2015

Module of code related to the MongoDB database that holds all of the review data.

The insert_train_test_reviews function gets all suitable, English-language reviews for a given data-set (at the provided file-path) and inserts them into the the MongoDB database ('reviews_project') under the 'reviews' collection.
'''
import sys
import numpy as np
from math import ceil
from data import APPID_DICT
from os.path import basename
from random import randint, shuffle, seed
from pymongo.errors import DuplicateKeyError
from util.datasets import get_and_describe_dataset


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
    cdef float MAXLEN = dataset['MAXLEN']
    cdef float MINLEN = dataset['MINLEN']
    cdef float MAXHOURS = dataset['MAXHOURS']
    cdef float MINHOURS = dataset['MINHOURS']
    sys.stderr.write('Max. length: {}\nMin. length: {}\nMax. # of ' \
                     'hours: {}\nMin. # of hours: {}\n' \
                     '\n'.format(dataset['MAXLEN'],
                                 dataset['MINLEN'], 
                                 dataset['MAXHOURS'],
                                 dataset['MINHOURS']))

    # If the hours played values are to be divided into bins, get the range
    # that each bin maps to
    if bins:
        bin_ranges = get_bin_ranges(MINHOURS,
                                    MAXHOURS,
                                    bins)

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
    cdef int training_set_size = ceil(len(train_test_reviews)
                                      *(percent_train/100))
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

    # Initialize a bulk writer
    bulk = reviewdb.initialize_unordered_bulk_op()

    # Insert training set reviews into MongoDB collection
    for r in training_reviews:
        # First, let's add some keys for the training/test partition, the
        # game's name, and the appid
        r['game'] = game
        r['appid'] = appid
        r['partition'] = 'training'

        if bins:
            _bin = get_bin(bin_ranges,
                           r['hours'])
            if _bin:
                r['hours'] = _bin
            else:
                sys.exit('WARNING: The hours played value ({}) did not seem' \
                         ' to fall within any of the bin ranges.\n\nBin ' \
                         'ranges:\n\n{}\n\nExiting.\n'.format(r['hours'],
                                                              repr(bin_ranges)))

        try:
            # Actually, to really mimic the real situation, we'd have to
            # insert and then remove...
            if not just_describe:
                bulk.insert(r)
            pass
        except DuplicateKeyError as e:
            if remaining_reviews:
                sys.stderr.write('WARNING: Encountered ' \
                                 'DuplicateKeyError. Throwing out ' \
                                 'following review:\n\n{}\n\nTaking ' \
                                 'review from list of remaining ' \
                                 'reviews.\n'.format(r))
                training_reviews.append(remaining_reviews.pop())
            else:
                sys.stderr.write('WARNING: Encountered ' \
                                 'DuplicateKeyError. Throwing out ' \
                                 'following review:\n\n{}\n\nNo reviews ' \
                                 'left to substitute in.\n'.format(r))

    # Insert test set reviews into MongoDB collection
    for r in test_reviews:
        # First, let's add some keys for the training/test partition, the
        # game's name, and the appid
        r['game'] = game
        r['appid'] = appid
        r['partition'] = 'test'

        if bins:
            _bin = get_bin(bin_ranges,
                           r['hours'])
            if _bin:
                r['hours'] = _bin
            else:
                sys.exit('WARNING: The hours played value ({}) did not seem' \
                         ' to fall within any of the bin ranges.\n\nBin ' \
                         'ranges:\n\n{}\n\nExiting.\n'.format(r['hours'],
                                                              repr(bin_ranges)))

        try:
            if not just_describe:
                bulk.insert(r)
            pass
        except DuplicateKeyError as e:
            if remaining_reviews:
                sys.stderr.write('WARNING: Encountered ' \
                                 'DuplicateKeyError. Throwing out ' \
                                 'following review:\n\n{}\n\nTaking ' \
                                 'review from list of remaining ' \
                                 'reviews.\n'.format(r))
                training_reviews.append(remaining_reviews.pop())
            else:
                sys.stderr.write('WARNING: Encountered ' \
                                 'DuplicateKeyError. Throwing out ' \
                                 'following review:\n\n{}\n\nNo reviews ' \
                                 'left to substitute in.\n'.format(r))

    # Insert extra reviews into the MongoDB collection, using 'extra' as
    # the value of the 'partition' key
    for r in remaining_reviews:
        # Add keys for the partition ("extra"), the game's name, and the
        # appid
        r['game'] = game
        r['appid'] = appid
        r['partition'] = 'extra'

        if bins:
            _bin = get_bin(bin_ranges,
                           r['hours'])
            if _bin:
                r['hours'] = _bin
            else:
                sys.exit('WARNING: The hours played value ({}) did not seem' \
                         ' to fall within any of the bin ranges.\n\nBin ' \
                         'ranges:\n\n{}\n\nExiting.\n'.format(r['hours'],
                                                              repr(bin_ranges)))

        try:
            if not just_describe:
                bulk.insert(r)
            pass
        except DuplicateKeyError as e:
            sys.stderr.write('WARNING: Encountered DuplicateKeyError. ' \
                             'Throwing out following ' \
                             'review:\n\n{}\n\n'.format(r))

    # Do a bulk write operation
    bulk.execute()

    # Print out some information about how many reviews were added
    if not just_describe:
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


def get_bin_ranges(_min, _max, nbins):
    '''
    Return list of floating point number ranges (in increments of 0.1) that correspond to each bin in the distribution.

    :param _min: minimum value of the distribution
    :type _min: float
    :param _max: maximum value of the distribution
    :type _max: float
    :param nbins: number of bins into which the distribution is being sub-divided
    :type nbins: int
    :returns: list of tuples representing the minimum and maximum values of a bin
    '''

    bin_size = round(float(_max - _min)/nbins,
                     1)
    bin_ranges = []
    _bin_start = _min - 0.1
    _bin_end = _min + bin_size
    for b in range(1, nbins + 1):
        if not b == 1:
            _bin_start = _bin_end
        if b == nbins:
            _bin_end = _bin_start + bin_size + 1.0
        else:
            _bin_end = _bin_start + bin_size
        bin_ranges.append((_bin_start,
                           _bin_end))
    return bin_ranges


def get_bin(bin_ranges, val):
    '''
    Return the index of the bin range in which the value falls.

    :param bin_ranges: list of ranges that define each bin
    :type bin_ranges: list of tuples representing the minimum and maximum values of a range of values
    :param val: value
    :type val: float
    :returns int (None if val not in any of the bin ranges)
    '''

    for i, bin_range in enumerate(bin_ranges):
        if val > bin_range[0] and val <= bin_range[1]:
            return i + 1
    return None