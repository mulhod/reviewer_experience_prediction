'''
@author Matt Mulholland
@date 05/05/2015

Module of code related to the MongoDB database that holds all of the review data.

The insert_train_test_reviews function gets all suitable, English-language reviews for a given data-set (at the provided file-path) and inserts them into the the MongoDB database ('reviews_project') under the 'reviews' collection.
'''
import sys
from math import ceil
from data import APPID_DICT
from os.path import basename
from random import randint, shuffle, seed
from pymongo.errors import DuplicateKeyError
from util.datasets import get_and_describe_dataset


def insert_train_test_reviews(reviewdb, file_path, max_size, percent_train,
                              describe=False, just_describe=False):
    '''
    Insert training/test set reviews into the MongoDB database.

    :param reviewdb: MongoDB reviews collection
    :type reviewdb: pymongo.MongoClient object
    :param file_path: path to game reviews file
    :type file_path: str
    :param max_size: maximum size of training/test set combination (in number of reviews)
    :type max_size: int
    :param percent_train: percent of training/test combination that should be reserved for the training set
    :type percent_train: float/int
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
    MAXLEN = dataset['MAXLEN']
    MINLEN = dataset['MINLEN']
    MAXHOURS = dataset['MAXHOURS']
    MINHOURS = dataset['MINHOURS']
    sys.stderr.write('Max. length: {}\nMin. length: {}\nMax. # of ' \
                     'hours: {}\nMin. # of hours: {}\n' \
                     '\n'.format(dataset['MAXLEN'],
                                 dataset['MINLEN'], 
                                 dataset['MAXHOURS'],
                                 dataset['MINHOURS']))

    # Shuffle the list of reviews so that we randomize it
    shuffle(reviews)

    # Get the training and test sets and the set of extra reviews (which
    # might get pulled in later if necessary)
    num_reviews = len(reviews)
    if num_reviews > max_size:
        train_test_reviews = reviews[:max_size]
    else:
        train_test_reviews = reviews[:num_reviews]
        max_size = num_reviews
    remaining_reviews = reviews[max_size:]

    # Divide the selected reviews into training/test sets
    training_set_size = ceil(len(train_test_reviews)*(percent_train/100))
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
                     ' are no replacement reviews to substitute in.\n')

    # Insert training set reviews into MongoDB collection
    for r in training_reviews:
        # First, let's add some keys for the training/test partition, the
        # game's name, and the appid
        r['game'] = game
        r['appid'] = appid
        r['partition'] = 'training'
        try:
            # Actually, to really mimic the real situation, we'd have to
            # insert and then remove...
            if not just_describe:
                reviewdb.insert(r)
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
        try:
            if not just_describe:
                reviewdb.insert(r)
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
        try:
            if not just_describe:
                reviewdb.insert(r)
        except DuplicateKeyError as e:
            sys.stderr.write('WARNING: Encountered DuplicateKeyError. ' \
                             'Throwing out following ' \
                             'review:\n\n{}\n\n'.format(r))

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