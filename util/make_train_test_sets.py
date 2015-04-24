#!/usr/env python3.4
import sys
import argparse
import pymongo
from os import listdir
from os.path import join, basename, abspath
from math import ceil
from random import shuffle
from util.read_data_files import get_reviews_for_game

# Establish connection to MongoDB database
connection = pymongo.MongoClient('mongodb://localhost:27017')
db = connection['reviews_project']
reviewdb = db['reviews']

def insert_reviews(file_path):
    '''
    Insert training/test set reviews into the MongoDB database.

    :param file_path: path to game reviews file
    :type file_path: str
    :returns: NoneType
    '''

    global reviewdb
    from data import APPID_DICT, FILTER_DICT
    game = basename(file_path)[:-4]
    appid = APPID_DICT[game]
    MAXLEN = FILTER_DICT[game]['MAXLEN']
    MINLEN = FILTER_DICT[game]['MINLEN']
    MAXHOURS = FILTER_DICT[game]['MAXHOURS']
    MINHOURS = FILTER_DICT[game]['MINHOURS']
    
    sys.stderr.write('Inserting reviews from {}...\n'.format(game))

    # Get list of all reviews represented as dictionaries with 'review' and
    # 'hours' keys
    reviews = get_reviews_for_game(file_path)

    # Here we refer to the game-specific values for filtering out outliers
    reviews = [r for r in reviews if len(r) <= MAXLEN
                                  and if len(r) >= MINLEN
                                  and if r['hours'] <= MAXHOURS
                                  and if r['hours'] >= MINHOURS]

    # Shuffle the list of reviews so that we randomize it
    shuffle(reviews)

    # Identify the size of the training and test sets
    num_reviews = len(reviews)
    if num_reviews >= 12000:
        reviews = reviews[:12001]
    else:
        reviews = reviews[:num_reviews]
    # The training set size will be 2/3rds of the reviews we're using for
    # making the training/test sets
    training_set_size = ceil((len(reviews)/3)*2)
    training_reviews = reviews[:training_set_size + 1]
    test_reviews = reviews[training_set_size:]
    sys.stderr.write('Number of training set reviews: {}\n'.format(len(
        training_reviews))
    sys.stderr.write('Number of test set reviews: {}\n'.format(len(
        test_reviews))

    # Insert training set reviews into MongoDB collection
    for r in training_reviews:
        # First, let's add some keys for the training/test partition, the
        # game's name, and the appid
        r['game'] = game
        r['appid'] = appid
        r['partition'] = 'training'
        reviewdb.insert(r)
    # Insert test set reviews into MongoDB collection
    for r in test_reviews:
        # First, let's add some keys for the training/test partition, the
        # game's name, and the appid
        r['game'] = game
        r['appid'] = appid
        r['partition'] = 'test'
        reviewdb.insert(r)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage='python make_train_test_sets.py',
        description='Build train/test sets for each game. We will use a ' +
                    'minimum of 600 reviews for each training set and a ' +
                    'maximum of 6500 reviews for each training set. For ' +
                    'the test sets, we will use a minimum of 300 reviews' +
                    ' and a maximum of 3500 reviews.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game_files',
        help='comma-separated list of file-names or "all" for all of the ' \
             'files (the game files should reside in the "data" directory)',
        type=str,
        required=True)
    args = parser.parse_args()

    # Get paths to the data and arff_files directories
    project_dir = dirname(dirname(abspath(realpath(__file__))))
    data_dir = join(project_dir,
                    'data')

    # Get list of games
    game_files = []
    if args.game_files == "all":
        game_files = [f for f in listdir(data_dir) if f.endswith('.txt')]
    else:
        game_files = args.game_files.split(',')

    # For each game in our list of games, we will read in the reviews from
    # the data file and then put entries in our MongoDB collection with a
    # key that identifies each review as either training or test
    for game in games:
        insert_reviews(abspath(join(data_dir,
                                    game)))