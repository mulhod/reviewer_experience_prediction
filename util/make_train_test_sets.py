#!/usr/env python3.4
'''
@author Matt Mulholland
@date May, 2015

Script used to create training/test sets in a MongoDB database from review data extracted from flat files.
'''
import sys
import logging
import pymongo
import argparse
from os import listdir
from util.mongodb import insert_train_test_reviews
from util.datasets import get_and_describe_dataset
from os.path import join, basename, abspath, dirname, realpath

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage='python make_train_test_sets.py',
        description='Build train/test sets for each game. Take up to ' \
                    '21k reviews and split it 66.67/33.33 training/test, ' \
                    'respectively, by default. Both the maximum size and ' \
                    'the percentage split can be altered via command-line ' \
                    'flags. All selected reviews will be put into the ' \
                    'MongoDB "reviews_project" database\'s  "reviews" ' \
                    'collection (which is being hosted on the Montclair ' \
                    'University server on port 27017).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game_files',
        help='comma-separated list of file-names or "all" for all of the ' \
             'files (the game files should reside in the "data" directory)',
        type=str,
        required=True)
    parser.add_argument('--max_size', '-m',
        help='maximum number of reviews to get for training/testing (if' \
             ' possible)',
        type=int,
        default=10000)
    parser.add_argument('--percent_train', '-%',
        help='percent of selected reviews for which to use for the ' \
             'training set, the rest going to the test set',
        type=float,
        default=(2.0/3.0)*100.0)
    parser.add_argument('--convert_to_bins', '-bins',
        help='number of equal sub-divisions of the hours-played values, ' \
             'e.g. if 10 and the hours values range from 0 up to 1000, ' \
             'then hours values 0-99 will become 1, 100-199 will become 2, ' \
             'etc. (will probably be necessay to train a model that ' \
             'actually is predictive to an acceptable degree); note that ' \
             'both hours values will be retained, the original under the ' \
             'name "hours" and the converted value under the name ' \
             '"hours_bin"',
        type=int,
        required=False)
    parser.add_argument('-describe', '--make_reports',
        help='generate reports and histograms describing the data ' \
             'filtering procedure',
        action='store_true',
        default=False)
    parser.add_argument('--just_describe',
        help='generate reports and histograms describing the data ' \
             'filtering procedure, but then do NOT insert the reviews into ' \
             'the DB',
        action='store_true',
        default=False)
    parser.add_argument('--mongodb_port', '-dbport',
        help='port that the MongoDB server is running',
        type=int,
        default=27017)
    args = parser.parse_args()

    # Initialize logging system
    logger = logging.getLogger('rep.make_train_test_sets')
    logger.setLevel(logging.DEBUG)

    # Create console handler with a high logging level specificity
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)

    # Add nicer formatting
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -'
                                  ' %(message)s')
    #fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    #logger.addHandler(fh)
    logger.addHandler(sh)

    # Make sure value passed in via the --convert_to_bins/-bins option flag
    # makes sense and, if so, assign value to variable bins (if not, set bins
    # equal to 0)
    if args.convert_to_bins and args.convert_to_bins < 2:
        logger.info('ERROR: The value passed in via --convert_to_bins/-bins' \
                    'must be greater than 1 since there must be multiple ' \
                    'bins to divide the hours played values. Exiting.')
        sys.exit(1)
    elif args.convert_to_bins:
        bins = args.convert_to_bins
    else:
        bins = 0

    # Establish connection to MongoDB database
    connection = pymongo.MongoClient('mongodb://localhost:' \
                                     '{}'.format(args.mongodb_port))
    db = connection['reviews_project']
    reviewdb = db['reviews']

    # Get paths to the project and data directories
    project_dir = dirname(dirname(abspath(realpath(__file__))))
    data_dir = join(project_dir,
                    'data')

    # Make sure args make sense
    if args.max_size < 50:
        logger.info('ERROR: You can\'t be serious, right? You passed in a ' \
                    'value of 50 for the MAXIMUM size of the combination of' \
                    ' training/test sets? Exiting.')
        sys.exit(1)
    if args.percent_train < 1.0:
        logger.info('ERROR: You can\'t be serious, right? You passed in a ' \
                    'value of 1.0% for the percentage of the selected ' \
                    'reviews that will be devoted to the training set? That' \
                    ' is not going to be enough training samples... Exiting.')
        sys.exit(1)

    # Make sense of arguments
    if args.make_reports and args.just_describe:
        logger.info('WARNING: If the --just_describe and -describe/' \
                    '--make_reports option flags are used, --just_describe ' \
                    'wins out, i.e., reports will be generated, but no ' \
                    'reviews will be inserted into the DB.')

    # Get list of games
    game_files = []
    if args.game_files == "all":
        game_files = [f for f in listdir(data_dir) if f.endswith('.txt')]
        del game_files[game_files.index('sample.txt')]
    else:
        game_files = args.game_files.split(',')

    logger.info('Adding training/test partitions to Mongo DB for the ' +
                'following games: {}'.format(', '.join([g[:-4] for g in
                                                        game_files])))
    logger.info('Maximum size for the combined training/test sets: ' \
                '{0}'.format(args.max_size))
    logger.info('Percentage split between training and test sets: ' \
                '{2:.2f}/{3:.2f}'.format(args.percent_train,
                                         100.0 - args.percent_train))
    if bins:
        logger.info('Converting hours played values to {} bins.'.format(bins))

    # For each game in our list of games, we will read in the reviews from
    # the data file and then put entries in our MongoDB collection with a
    # key that identifies each review as either training or test
    for game_file in game_files:
        logger.info('Getting/inserting reviews for {}' \
                    '...'.format(basename(game_file)[:-4]))
        insert_train_test_reviews(reviewdb,
                                  abspath(join(data_dir,
                                               game_file)),
                                  args.max_size,
                                  args.percent_train,
                                  bins=bins,
                                  describe=args.make_reports,
                                  just_describe=args.just_describe)

    logger.info('Complete.')