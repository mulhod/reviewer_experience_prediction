#!/usr/env python3.4
import sys
import pymongo
import argparse
from os import listdir
from util.mongodb import insert_train_test_reviews
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
        sys.exit('ERROR: You can\'t be serious, right? You passed in a ' +
                 'value of 50 for the MAXIMUM size of the combination ' +
                 'of training/test sets? Exiting.\n')
    if args.percent_train < 1.0:
        sys.exit('ERROR: You can\'t be serious, right? You passed in a ' +
                 'value of 1.0% for the percentage of the selected ' +
                 'reviews that will be devoted to the training set? That' +
                 'is not going to be enough training samples... ' +
                 'Exiting.\n')

    # Make sense of arguments
    if args.make_reports and args.just_describe:
        sys.stderr.write('WARNING: If the --just_describe and -describe/' \
                         '--make_reports option flags are used, ' \
                         '--just_describe wins out, i.e., reports will be ' \
                         'generated, but no reviews will be inserted into ' \
                         'the DB.\n')

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
        sys.stderr.write('Getting/inserting reviews for {}...\n' \
                         '\n'.format(basename(game_file)[:-4]))
        insert_train_test_reviews(reviewdb,
                                  abspath(join(data_dir,
                                               game_file)),
                                  args.max_size,
                                  args.percent_train,
                                  describe=args.make_reports,
                                  just_describe=args.just_describe)

    sys.stderr.write('\nComplete.\n')