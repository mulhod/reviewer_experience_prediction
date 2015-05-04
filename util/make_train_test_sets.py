#!/usr/env python3.4
import sys
import pymongo
import argparse
from re import sub
import numpy as np
import pandas as pd
import seaborn as sns
from math import ceil
from os import listdir
from random import randint
from random import shuffle
import matplotlib.pyplot as plt
from data import APPID_DICT
from util.read_data_files import get_reviews_for_game
from os.path import join, basename, abspath, dirname, realpath

# Establish connection to MongoDB database
connection = pymongo.MongoClient('mongodb://localhost:27017')
db = connection['reviews_project']
reviewdb = db['reviews']

# Seaborn-related configuration
sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (8, 4)})


def get_and_describe_dataset(file_path):
    '''
    Return suggested values for the maximum/minimum review length and hours played values for a given game; also produce a report with some descriptive statistics and graphs.

    :param file_path: path to game reviews file
    :type file_path: str
    :returns: dict containing a 'reviews' key mapped to the list of read-in review dictionaries and int values mapped to keys for MAXLEN, MINLEN, MAXHOURS, and MINHOURS
    '''

    # Get path to reports directory and open report file
    reports_dir = join(dirname(realpath(__file__)),
                       'reports')
    game = basename(file_path)[:-4]
    output_path = join(reports_dir,
                       '{}_report.txt'.format(game))
    output = open(output_path,
                  'w')

    # Get list of review dictionaries
    reviews = get_reviews_for_game(file_path)

    output.write('Descriptive Report for {}\n==========================' \
                 '=====================================================' \
                 '\n\n'.format(re.sub(r'_',
                                      ' ',
                                      game)))
    output.write('Number of English-language reviews: {}\n' \
                 '\n'.format(len(reviews)))

    # Look at review lengths to figure out what should be filtered out
    output.write('Review Lengths Distribution\n\n')
    lengths = np.array([len(review['review']) for review in reviews])
    mean = lengths.mean()
    output.write('Average review length: {}\n'.format(avg_len))
    output.write('Minimum review length = {}\n'.format(min(lengths)))
    output.write('Maximum review length = {}\n'.format(max(lengths)))
    std = lengths.std()
    output.write('Standard deviation = {}\n\n'.format(std))
    
    # Use the standard deviation to define the range of acceptable reviews
    # (in terms of the length only) as within 2 standard deviations of the
    # mean (but with the added caveat that the reviews be at least 50
    # characters
    MINLEN = 50 if (mean - 2.0*std) < 50 else (mean - 2.0*std)
    MAXLEN = mean + 2.0*std
    
    # Generate length histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(pd.Series(lengths), label=game)
    ax.set_xlabel('Review length (in characters)')
    ax.set_ylabel('Total reviews')
    fig.savefig(join(reports_dir,
                     '{}_length_histogram'.format(game)))
    
    # Look at hours played values in the same way as above for length
    output.write('Review Experience Distribution\n\n')
    hours = np.array([review['hours'] for review in reviews])
    mean = hours.mean()
    output.write('Average game experience (in hours played): {}' \
                 '\n'.format(mean))
    output.write('Minimum experience = {}\n'.format(min(hours)))
    output.write('Maximum expeience = {}\n'.format(max(hours)))
    std = lengths.std()
    output.write('Standard deviation = {}\n\n'.format(std))

    # Use the standard deviation to define the range of acceptable reviews
    # (in terms of experience) as within 2 standard deviations of the mean
    # (starting from zero, actually)
    MINHOURS = 0
    MAXHOURS = mean + 2.0*std

    # Generate experience histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(pd.Series(hours),
            label=game)
    ax.set_xlabel('Game experience (in hours played)')
    ax.set_ylabel('Total reviews')
    fig.savefig(join(reports_dir,
                     '{}_experience_histogram'.format(game)))

    return dict(reviews=reviews,
                MINLEN=MINLEN
                MAXLEN=MAXLEN,
                MINHOURS=MINHOURS,
                MAXHOURS=MAXHOURS)
    


def insert_reviews(file_path, max_size, percent_train):
    '''
    Insert training/test set reviews into the MongoDB database.

    :param file_path: path to game reviews file
    :type file_path: str
    :param max_size: maximum size of training/test set combination (in number of reviews)
    :type max_size: int
    :param percent_train: percent of training/test combination that should be reserved for the training set
    :type percent_train: float/int
    :returns: None
    '''

    global reviewdb
    game = basename(file_path)[:-4]
    appid = APPID_DICT[game]
    
    sys.stderr.write('Inserting reviews from {}...\n'.format(game))

    # Get list of all reviews represented as dictionaries with 'review' and
    # 'hours' keys and get the filter values
    dataset = get_and_describe_dataset(file_path)
    reviews = dataset['reviews']
    sys.stderr.write('Number of original, English language reviews ' \
                     'collected: {}\n'.format(len(reviews)))
    MAXLEN = dataset['MAXLEN']
    MINLEN = dataset['MINLEN']
    MAXHOURS = dataset['MAXHOURS']
    MINHOURS = dataset['MINHOURS']
    sys.stderr.write('Max. length: {}\nMin. length: {}\nMax. # of ' \
                     'hours: {}\nMin. # of hours: {}\n\n'.format(MAXLEN,
                                                                 MINLEN, 
                                                                 MAXHOURS,
                                                                 MINHOURS))

    # Here we refer to the game-specific values for filtering out outliers
    reviews = [r for r in reviews if len(r['review']) <= MAXLEN
                                  and len(r['review']) >= MINLEN
                                  and r['hours'] <= MAXHOURS
                                  and r['hours'] >= MINHOURS]

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
            reviewdb.insert(r)
        except pymongo.errors.DuplicateKeyError as e:
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
            reviewdb.insert(r)
        except pymongo.errors.DuplicateKeyError as e:
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
            reviewdb.insert(r)
        except pymongo.errors.DuplicateKeyError as e:
            sys.stderr.write('WARNING: Encountered DuplicateKeyError. ' \
                             'Throwing out following ' \
                             'review:\n\n{}\n\n'.format(r))

    # Print out some information about how many reviews were added
    train_inserts = reviewdb.find({'appid': appid,
                                   'partition': 'train'}).count()
    test_inserts = reviewdb.find({'appid': appid,
                                  'partition': 'test'}).count()
    extra_inserts = reviewdb.find({'appid': appid,
                                   'partition': 'extra'}).count()
    sys.stderr.write('Inserted {} training set reviews, {} test set ' \
                     'reviews, and {} extra reviews...\n\n' \
                     '\n'.format(train_inserts,
                                 test_inserts,
                                 extra_inserts))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage='python make_train_test_sets.py',
        description='Build train/test sets for each game. Take up to ' +
                    '21k reviews and split it 66.67/33.33 training/ ' +
                    'test, respectively, by default. Both the maximum ' +
                    'size and the percentage split can be altered via ' +
                    'command-line flags. All selected reviews will be ' +
                    'put into the MongoDB "reviews_project" database\'s ' +
                    ' "reviews" collection (which is being hosted on the' +
                    ' Montclair University server on port 27017).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game_files',
        help='comma-separated list of file-names or "all" for all of the ' \
             'files (the game files should reside in the "data" directory)',
        type=str,
        required=True)
    parser.add_argument('--max_size', '-m',
        help='maximum number of reviews to get for training/testing (if' +
             ' possible)',
        type=int,
        default=21000)
    parser.add_argument('--percent_train', '-%',
        help='percent of selected reviews for which to use for the ' +
             'training set, the rest going to the test set',
        type=float,
        default=(2.0/3.0)*100.0)
    args = parser.parse_args()

    # Get paths to the project and data directories
    project_dir = dirname(dirname(abspath(realpath(__file__))))
    data_dir = join(project_dir,
                    'data')

    # Make sure args make sense
    if args.max_size < 50:
        sys.exit('ERROR: You can\'t be serious, right? You passed in a ' +
                 'value of 50 for the MAXIMUM size of the combination ' +
                 'of training/test sets? Exiting.\n')
    if args.percent_train < 1.0:
        sys.exit('ERROR: You can\'t be serious, right? You passed in a ' +
                 'value of 1.0% for the percentage of the selected ' +
                 'reviews that will be devoted to the training set? That' +
                 'is not going to be enough training samples... ' +
                 'Exiting.\n')

    # Get list of games
    game_files = []
    if args.game_files == "all":
        game_files = [f for f in listdir(data_dir) if f.endswith('.txt')]
    else:
        game_files = args.game_files.split(',')

    sys.stderr.write('Adding training/test partitions to Mongo DB for ' +
                     'the following games: ' +
                     '{}\n'.format(', '.join([g[:-4] for g in game_files])))
    sys.stderr.write('\nMaximum size for the combined training/test ' \
                     'sets: {0}\nPercentage split between training and ' \
                     'test sets: {1:.2f}/{2:.2f}' \
                     '\n'.format(args.max_size,
                                 args.percent_train,
                                 100.0 - args.percent_train))

    # For each game in our list of games, we will read in the reviews from
    # the data file and then put entries in our MongoDB collection with a
    # key that identifies each review as either training or test
    for game_file in game_files:
        insert_reviews(abspath(join(data_dir,
                                    game_file)),
                       args.max_size,
                       args.percent_train)

    sys.stderr.write('\nComplete.\n')