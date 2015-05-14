#!/usr/env python3.4
'''
@author: Matt Mulholland, Janette Martinez, Emily Olshefski
@date: 3/18/15

Script used to train models on datasets (or multiple datasets combined).
'''
import sys
import pymongo
import logging
import argparse
from time import sleep
from os import listdir
from data import APPID_DICT
from spacy.en import English
from collections import Counter
from skll import run_configuration
from pymongo.errors import AutoReconnect
from json import dumps, JSONEncoder, JSONDecoder
from os.path import join, dirname, realpath, abspath
from src.feature_extraction import (Review, extract_features_from_review,
                                    write_config_file)

project_dir = dirname(dirname(realpath(__file__)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage='python train.py --game_files ' \
        'GAME_FILE1,GAME_FILE2,...[ OPTIONS]',
        description='Build a machine learning model based on the features ' \
                    'that are extracted from a set of reviews relating to a' \
                    ' specific game or set of games.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game_files',
        help='comma-separated list of file-names or "all" for all of the ' \
             'files (the game files should reside in the "data" directory)',
        type=str,
        required=True)
    parser.add_argument('--combine',
        help='combine all game files together to make one big model',
        action='store_true',
        required=False)
    parser.add_argument('--combined_model_prefix',
        help='prefix to use when naming the combined model (required if ' \
             'the --combine flag is used)',
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
        help='extract features from all of the reviews, generate .jsonlines' \
             ' files, etc., but quit before training any models',
        action='store_true',
        default=False)
    parser.add_argument('--try_to_reuse_extracted_features',
        help='try to make use of previously-extracted features that reside ' \
             'in the MongoDB database',
        action='store_true',
        default=True)
    parser.add_argument('--run_configuration', '-run_cfg',
        help='assumes that .jsonlines files have already been created and ' \
             'attempts to run the configuration',
         action='store_true',
         default=False)
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
                     'replog_train.txt'))
    args = parser.parse_args()

    # Initialize logging system
    logger = logging.getLogger('train')
    logger.setLevel(logging.ERROR)

    # Create file handler with a high logging level specificity
    fh = logging.FileHandler(abspath(args.log_file_path))
    fh.setLevel(logging.ERROR)

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

    # Get paths to the project, data, working, and models directories
    data_dir = join(project_dir,
                    'data')
    working_dir = join(project_dir,
                      'working')
    models_dir = join(project_dir,
                      'models')
    cfg_dir = join(project_dir,
                   'config')
    logs_dir = join(project_dir,
                   'logs')
    logger.debug('project directory: {}'.format(project_dir))
    logger.debug('data directory: {}'.format(data_dir))
    logger.debug('working directory: {}'.format(working_dir))
    logger.debug('models directory: {}'.format(models_dir))
    logger.debug('configuration directory: {}'.format(cfg_dir))
    logger.debug('logs directory: {}'.format(logs_dir))

    binarize = not args.do_not_binarize_features
    logger.debug('Binarize features? {}'.format(binarize))
    lowercase_text = not args.do_not_lowercase_text
    logger.debug('Lower-case text as part of the normalization step? ' \
                 '{}'.format(lowercase_text))
    logger.debug('Just extract features? ' \
                 '{}'.format(args.just_extract_features))
    logger.debug('Try to reuse extracted features? ' \
                 '{}'.format(args.try_to_reuse_extracted_features))

    bins = not args.use_original_hours_values

    # Make sure that, if --combine is being used, there is also a file prefix
    # being passed in via --combined_model_prefix for the combined model
    if args.combine and not args.combined_model_prefix:
        logger.error('When using the --combine flag, you must also specify ' \
                     'a model prefix, which can be passed in via the ' \
                     '--combined_model_prefix option argument. Exiting.')
        sys.exit(1)

    if not args.run_configuration:
        # Establish connection to MongoDB database
        connection_string = 'mongodb://localhost:{}'.format(args.mongodb_port)
        try:
            connection = pymongo.MongoClient(connection_string)
        except pymongo.errors.ConnectionFailure as e:
            logger.error('Unable to connect to to Mongo server at {}. ' \
                         'Exiting.'.format(connection_string))
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

    # Get list of games
    game_files = []
    if args.game_files == "all":
        game_files = [f for f in listdir(data_dir) if f.endswith('.txt')]
        del game_files[game_files.index('sample.txt')]
    else:
        game_files = args.game_files.split(',')

    # Train a combined model with all of the games or train models for each
    # individual game dataset
    if args.combine:

        if not args.run_configuration:
            logger.info('Extracting features to train a combined model with' \
                        ' training data from the following games: ' \
                        '{}'.format(', '.join(game_files)))

        if not args.run_configuration:
            # Initialize empty list for holding all of the feature
            # dictionaries from each review in each game and then extract
            # features from each game's training data
            feature_dicts = []

        # Open JSONLINES file
        jsonlines_filename = '{}.jsonlines'.format(args.combined_model_prefix)
        jsonlines_filepath = join(working_dir,
                                  jsonlines_filename)
        if not args.run_configuration:
            logger.info('Writing {} to working directory' \
                        '...'.format(jsonlines_filename))
            jsonlines_file = open(jsonlines_filepath,
                                      'w')

        for game_file in game_files:

            # Get the training reviews for this game from the Mongo
            # database
            game = game_file[:-4]

            if not args.run_configuration:
                logger.info('Extracting features from the training data for' \
                            ' {}...'.format(game))
                appid = APPID_DICT[game]
                game_docs = reviewdb.find({'game': game,
                                           'partition': 'training'},
                                          {'features': 0,
                                           'game': 0,
                                           'partition': 0})
                if game_docs.count() == 0:
                    logger.error('No matching documents were found in the ' \
                                 'MongoDB collection in the training ' \
                                 'partition for game {}. Exiting' \
                                 '.'.format(game))
                    sys.exit(1)

            if not args.run_configuration:

                # Iterate over all training documents for the given game
                for game_doc in game_docs:

                    # Instantiate a Review object
                    _Review = Review(game_doc['review'],
                                     game_doc['hours_bin'] if bins else \
                                         game_doc['hours'],
                                     game,
                                     appid,
                                     spaCy_nlp,
                                     lower=lowercase_text)

                    # Extract features from the review text
                    found_features = False
                    if args.try_to_reuse_extracted_features:
                        features_doc = reviewdb.find_one(
                            {'_id': game_doc['_id']},
                            {'_id': 0,
                             'features': 1})
                        features = features_doc.get('features')
                        if features \
                           and game_doc.get('binarized') == binarize:
                            features = json_decoder.decode(features)
                            found_features = True

                    if not found_features:
                        features = \
                           extract_features_from_review(_Review,
                                     lowercase_cngrams=args.lowercase_cngrams)

                    # If binarize is True, make all values 1
                    if binarize and not (found_features
                                         and game_doc.get('binarized')):
                        features = dict(Counter(list(features)))

                    # Update Mongo database game doc with new key "features",
                    # which will be mapped to game_features, and a new key
                    # "binarized", which will be set to True if features were
                    # extracted with the --do_not_binarize_features flag or
                    # False otherwise
                    if not found_features:
                        tries = 0
                        while tries < 5:
                            try:
                                reviewdb.update(
                              {'_id': game_doc['_id']},
                              {'$set': {'features': json_encoder.encode(
                                            features),
                                        'binarized': binarize}})
                                break
                            except AutoReconnect as e:
                                logger.warning('Encountered ' \
                                               'ConnectionFailure error, ' \
                                               'attempting to reconnect ' \
                                               'automatically...')
                                tries += 1
                                if tries >= 5:
                                    logger.error('Unable to update database' \
                                                 'even after 5 tries. ' \
                                                 'Exiting.')
                                    sys.exit(1)
                                sleep(20)

                    # Write features to line of JSONLINES output file
                    jsonlines_file.write('{}\n'.format(
                        dumps({'id': str(game_doc['_id']),
                               'y': game_doc['hours_bin'] if bins else
                                   game_doc['hours'],
                               'x': features}).encode('utf-8').decode(
                                                                    'utf-8')))

        if not args.run_configuration:
            # Close JSONLINES file and take features out of memory
            jsonlines_file.close()
            features = None

        if not args.run_configuration:

            # Set up SKLL job arguments
            learner_name = 'RescaledSVR'
            param_grid_list = [{'C': [10.0 ** x for x in range(-3, 4)]}]
            grid_objective = 'quadratic_weighted_kappa'

            # Create a template for the SKLL config file
            # Note that all values must be strings
            cfg_dict_base = {"General": {},
                             "Input": {"train_location": working_dir,
                                       "ids_to_floats": "False",
                                       "label_col": "y",
                                       "featuresets": \
                                            dumps([[
                                                args.combined_model_prefix]]),
                                       "suffix": '.jsonlines',
                                       "learners": dumps([learner_name])
                                       },
                             "Tuning": {"feature_scaling": "none",
                                        "grid_search": "True",
                                        "min_feature_count": "1",
                                        "objective": grid_objective,
                                        "param_grids": dumps(
                                                           [param_grid_list]),
                                        },
                             "Output": {"probability": "False",
                                        "log": join(logs_dir,
                                                    '{}.log'.format(
                                                  args.combined_model_prefix))
                                        }
                             }

        cfg_filename = '{}.cfg'.format(args.combined_model_prefix)
        cfg_filepath = join(cfg_dir,
                            cfg_filename)
        if not args.run_configuration:

            # Set up the job for training the model
            logger.info('Generating configuration file...')
            cfg_dict_base["General"]["task"] = "train"
            cfg_dict_base["General"]["experiment_name"] = \
                args.combined_model_prefix
            cfg_dict_base["Output"]["models"] = models_dir
            write_config_file(cfg_dict_base,
                              cfg_filepath)

        if not args.just_extract_features:
            # Run the SKLL configuration, producing a model file
            logger.info('Training combined model...')
            run_configuration(cfg_filepath)
    else:
        for game_file in game_files:

            game = game_file[:-4]

            if not args.run_configuration:
                logger.info('Extracting features to train a model with ' \
                            'training data from {}...'.format(game))

                # Initialize empty list for holding all of the feature
                # dictionaries from each review and then extract features from
                # all reviews
                feature_dicts = []

                # Get the training reviews for this game from the Mongo
                # database
                logger.info('Extracting features from the training data for' \
                            ' {}...'.format(game))
                appid = APPID_DICT[game]
                game_docs = reviewdb.find({'game': game,
                                           'partition': 'training'},
                                          {'features': 0,
                                           'game': 0,
                                           'partition': 0})
                if game_docs.count() == 0:
                    logger.error('No matching documents were found in the ' \
                                 'MongoDB collection in the training ' \
                                 'partition for game {}. Exiting' \
                                 '.'.format(game))
                    sys.exit(1)

            # Open JSONLINES file
            jsonlines_filename = '{}.jsonlines'.format(game)
            jsonlines_filepath = join(working_dir,
                                      jsonlines_filename)

            if not args.run_configuration:
                logger.info('Writing {} to working directory' \
                            '...'.format(jsonlines_filename))
                jsonlines_file = open(jsonlines_filepath,
                                      'w')

                # Iterate over all training documents for the given game
                for game_doc in game_docs:

                    if bins:
                        hours = game_doc['hours_bin']
                    else:
                        hours = game_doc['hours']
                    review_text = game_doc['review']

                    # Instantiate a Review object
                    _Review = Review(game_doc['review'],
                                 game_doc['hours_bin'] if bins else \
                                     game_doc['hours'],
                                 game,
                                 appid,
                                 spaCy_nlp,
                                 lower=lowercase_text)

                    # Extract features from the review text
                    found_features = False
                    if args.try_to_reuse_extracted_features:
                        features_doc = reviewdb.find_one(
                            {'_id': game_doc['_id']},
                            {'_id': 0,
                             'features': 1})
                        features = features_doc.get('features')
                        if features and game_doc.get('binarized') == binarize:
                            features = json_decoder.decode(features)
                            found_features = True

                    if not found_features:
                        features = \
                            extract_features_from_review(_Review,
                                lowercase_cngrams=args.lowercase_cngrams)

                    # If binarize is True, make all values 1
                    if binarize and not (found_features
                                         and game_doc.get('binarized')):
                        features = dict(Counter(list(features)))

                    # Update Mongo database game doc with new key "features",
                    # which will be mapped to game_features, and a new key
                    # "binarized", which will be set to True if features were
                    # extracted with the --do_not_binarize_features flag or
                    # False otherwise
                    if not found_features:
                        tries = 0
                        while tries < 5:
                            try:
                                reviewdb.update(
                                    {'_id': game_doc['_id']},
                                    {'$set': {'features':
                                                  json_encoder.encode(
                                                                    features),
                                              'binarized': binarize}})
                                break
                            except AutoReconnect as e:
                                logger.warning('Encountered ' \
                                               'ConnectionFailure error, ' \
                                               'attempting to reconnect ' \
                                               'automatically...\n')
                                tries += 1
                                if tries >= 5:
                                    logger.error('Unable to update database' \
                                                 'even after 5 tries. ' \
                                                 'Exiting.')
                                    sys.exit(1)
                                sleep(20)

                    # Write features to line of JSONLINES output file
                    jsonlines_file.write('{}\n'.format(
                        dumps({'id': str(game_doc['_id']),
                               'y': game_doc['hours_bin'] if bins else
                                    game_doc['hours'],
                               'x': features}).encode('utf-8').decode(
                                                                    'utf-8')))

                # Close JSONLINES file and take features from memory
                jsonlines_file.close()
                features = None

                # Set up SKLL job arguments
                learner_name = 'RescaledSVR'
                param_grid_list = [{'C': [10.0 ** x for x in range(-3, 4)]}]
                grid_objective = 'quadratic_weighted_kappa'

                # Create a template for the SKLL config file
                # Note that all values must be strings
                cfg_dict_base = {"General": {},
                                 "Input": {"train_location": working_dir,
                                           "ids_to_floats": "False",
                                           "label_col": "y",
                                           "featuresets": dumps([[game]]),
                                           "suffix": '.jsonlines',
                                           "learners": dumps([learner_name])
                                           },
                                 "Tuning": {"feature_scaling": "none",
                                            "grid_search": "True",
                                            "min_feature_count": "1",
                                            "objective": grid_objective,
                                            "param_grids": dumps(
                                                           [param_grid_list]),
                                            },
                                 "Output": {"probability": "False",
                                            "log": join(logs_dir,
                                                        '{}.log'.format(game))
                                            }
                                 }

            cfg_filename = '{}.train.cfg'.format(game)
            cfg_filepath = join(cfg_dir,
                                cfg_filename)
            if not args.run_configuration:
                # Set up the job for training the model
                logger.info('Generating configuration file...')
                cfg_dict_base["General"]["task"] = "train"
                cfg_dict_base["General"]["experiment_name"] = \
                    '{}.train'.format(game)
                cfg_dict_base["Output"]["models"] = models_dir
                write_config_file(cfg_dict_base,
                                  cfg_filepath)

            if not args.just_extract_features:
                # Run the SKLL configuration, producing a model file
                logger.info('Training model for {}...'.format(game))
                run_configuration(cfg_filepath)

    logger.info('Complete.')