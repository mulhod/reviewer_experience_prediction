#!/usr/env python3.4
'''
@author: Matt Mulholland
@date: 5/13/15

Script used to make predictions for datasets (or multiple datasets combined) and generate evaluation metrics.
'''
import sys
import pymongo
import logging
import argparse
from os import listdir
from data import APPID_DICT
from spacy.en import English
from collections import Counter
from json import dumps, JSONEncoder, JSONDecoder
from os.path import realpath, dirname, abspath, join, exists
from src.feature_extraction import Review, extract_features_from_review

project_dir = dirname(dirname(realpath(__file__)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage='python evaluate.py --game_files' \
        ' GAME_FILE1,GAME_FILE2[ --resuls_path PATH|--predictions_path PATH' \
        '|--just_extract_features][ OPTIONS]',
        description='generate predictions for a data-set\'s test set ' \
                    'reviews and output evaluation metrics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game_files',
        help='comma-separated list of file-names or "all" for all of the ' \
             'files (the game files should reside in the "data" directory)',
        type=str,
        required=True)
    parser.add_argument('--model', '-m',
        help='model prefix (note: take care to ensure that you are ' \
             'extracting features from test reviews in the same way that ' \
             'features were extracted for the training reviews)',
        type=str,
        required=False)
    parser.add_argument('--results_path', '-r',
        help='destination path for results output file',
        type=str,
        required=False)
    parser.add_argument('--predictions_path', '-p',
        help='destination path for predictions file',
        type=str,
        required=False)
        parser.add_argument('--do_not_lowercase_text',
        help='do not make lower-casing part of the review text ' \
             'normalization step, which affects word n-gram-related ' \
             'features',
        action='store_true',
        default=False)
    parser.add_argument('--lowercase_cngrams',
        help='lower-case the review text before extracting character n-gram' \
             ' features',
        action='store_true',
        default=False)
    parser.add_argument('--use_original_hours_values',
        help='use the original, uncollapsed hours played values',
        action='store_true',
        default=False)
    parser.add_argument('--just_extract_features',
        help='extract features from all of the test set reviews and insert ' \
             'them into the MongoDB database, but quit before generating ' \
             'any predictions or results',
        action='store_true',
        default=False)
    parser.add_argument('--try_to_reuse_extracted_features',
        help='try to make use of previously-extracted features that reside ' \
             'in the MongoDB database',
        action='store_true',
        default=True)
    parser.add_argument('--do_not_binarize_features',
        help='do not make all non-zero feature frequencies equal to 1',
        action='store_true',
        default=False)
    parser.add_argument('--mongodb_port', '-dbport',
        help='port that the MongoDB server is running',
        type=int,
        default=27017)
    parser.add_argument('--log_file_path', '-log',
        help='path for log file',
        type=str,
        default=join(project_dir,
                     'logs',
                     'replog_eval.txt'))
    args = parser.parse_args()

    # Initialize logging system
    logger = logging.getLogger('eval')
    logger.setLevel(logging.INFO)

    # Create file handler
    fh = logging.FileHandler(abspath(args.log_file_path))
    fh.setLevel(logging.INFO)

    # Create console handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    # Add nicer formatting
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -'
                                  ' %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # Get predictions/results output file path and models directory path and
    # make sure model exists
    predictions_path = None
    results_path = None
    if not args.just_extract_features:
        # Get directories and paths
        models_dir = join(project_dir,
                          'models')
        if not exists(join(models_dir,
                           '{}.model'.format(args.model))):
            logger.error('Could not find model with prefix {} in models ' \
                         'directory ({}). Exiting.'.format(args.model,
                                                           models_dir))
            sys.exit(1)
        if args.predictions_path:
            predictions_path = abspath(predictions_path)
        if args.results_path:
            results_path = abspath(results_path)

    # Make sure command-line arguments make sense
    if not (args.results_path
            or args.predictions_path
            or args.just_extract_features):
        logger.error('evaluate.py has to at least do one thing, so specify ' \
                     'a predictions file path or a results file path or use' \
                     'the --just_extract_features option. Exiting.')
        sys.exit(1)
    if args.just_extract_features \
       and (args.results_path
            or args.predictions_path):
        logger.error('Unable to do feature extraction AND generate ' \
                     'results/predictions. Exiting.')
        sys.exit(1)

    if args.try_to_reuse_extracted_features \
       and (args.lowercase_cngrams
            or args.do_not_lowercase_text):
        logger.warning('If trying to reuse previously extracted features, ' \
                       'then the values picked for the --lowercase_cngrams ' \
                       'and --do_not_lowercase_text should match the values' \
                       ' used to build the models.')

    bins = not args.use_original_hours_values
    binarize = not args.do_not_binarize_features

    # Establish connection to MongoDB database
    connection_string = 'mongodb://localhost:{}'.format(args.mongodb_port)
    try:
        connection = pymongo.MongoClient(connection_string)
    except pymongo.errors.ConnectionFailure as e:
        logger.error('Unable to connect to to Mongo server at ' \
                     '{}'.format(connection_string))
        sys.exit(1)
    db = connection['reviews_project']
    reviewdb = db['reviews']
    reviewdb.write_concern['w'] = 0

    # Initialize an English-language spaCy NLP analyzer instance
    spaCy_nlp = English()

    # Initialize JSONEncoder, JSONDecoder objects
    if args.just_extract_features:
        json_encoder = JSONEncoder()
    if args.try_to_reuse_extracted_features:
        json_decoder = JSONDecoder()

    # Iterate over the game files, looking for test set reviews
    # Get list of games
    game_files = []
    if args.game_files == "all":
        game_files = [f for f in listdir(data_dir) if f.endswith('.txt')]
        del game_files[game_files.index('sample.txt')]
    else:
        game_files = args.game_files.split(',')

    # Open results/predictions files
    if predictions_path:
        predictions_file = open(predictions_path)
    if results_path:
        results_file = open(results_path)

    # Iterate over game files, generating/fetching features
    for game_file in game_files:

        game = game_file[:-4]
        appid = APPID_DICT[game]

        # Get test reviews
        logger.info('Extracting features from the training data for {}' \
                    '...'.format(game))
        game_docs = reviewdb.find({'game': game,
                                   'partition': 'test'},
                                  {'features': 0,
                                   'game': 0,
                                   'partition': 0})

        if game_docs.count() == 0:
            logger.error('No matching documents were found in the MongoDB ' \
                         'collection in the test partition for game {}. ' \
                         'Exiting.'.format(game))
            sys.exit(1)

        for game_doc in game_docs:

            if args.try_to_reuse_extracted_features:
                