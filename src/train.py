'''
:author: Matt Mulholland, Janette Martinez, Emily Olshefski
:date: March 18, 2015

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
from os.path import join, dirname, realpath, abspath, exists
from src.feature_extraction import (Review, extract_features_from_review,
                                    generate_config_file)

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
             'the --combine flag is used); basically, just combine the ' \
             'names of the games together in a way that doesn\'t make the ' \
             'file-name 150 characters long...',
        type=str,
        required=False)
    parser.add_argument('--learner',
        help='machine learning algorithm to use (only regressors are ' \
             'supported); both regular and rescaled versions of each ' \
             'learner are available',
        choices=['AdaBoost', 'DecisionTree', 'ElasticNet',
                 'GradientBoostingRegressor', 'KNeighborsRegressor', 'Lasso',
                 'LinearRegression', 'RandomForestRegressor', 'Ridge',
                 'SGDRegressor', 'SVR', 'RescaledAdaBoost',
                 'RescaledDecisionTree', 'RescaledElasticNet',
                 'RescaledGradientBoostingRegressor',
                 'RescaledKNeighborsRegressor', 'RescaledLasso',
                 'RescaledLinearRegression', 'RescaledRandomForestRegressor',
                 'RescaledRidge', 'RescaledSGDRegressor', 'RescaledSVR'],
        default='RescaledSVR')
    parser.add_argument('--objective_function', '-obj',
        help='objective function used for tuning',
        choices=['unweighted_kappa', 'linear_weighted_kappa',
                 'quadratic_weighted_kappa', 'uwk_off_by_one',
                 'lwk_off_by_one', 'qwk_off_by_one', 'r2',
                 'mean_squared_error'],
        default='quadratic_weighted_kappa')
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
    logger.setLevel(logging.DEBUG)

    # Create file handler
    fh = logging.FileHandler(abspath(args.log_file_path))
    fh.setLevel(logging.DEBUG)

    # Create console handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)

    # Add nicer formatting
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -'
                                  ' %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # Get paths to the project, data, working, and models directories

    # Print out some logging information about the upcoming tasks
    logger.debug('project directory: {}'.format(project_dir))
    logger.debug('Learner: {}'.format(args.learner))
    logger.debug('Objective function used for tuning: ' \
                 '{}'.format(args.objective_function))
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
    logger.debug('Use original hours values? {}'.format(not bins))

    # Make sure that, if --combine is being used, there is also a file prefix
    # being passed in via --combined_model_prefix for the combined model
    if args.combine and not args.combined_model_prefix:
        logger.error('When using the --combine flag, you must also specify ' \
                     'a model prefix, which can be passed in via the ' \
                     '--combined_model_prefix option argument. Exiting.')
        sys.exit(1)

    # Make sure command-line arguments make sense
    if args.just_extract_features \
       and (args.combine
            or args.combined_model_prefix
            or args.learner
            or args.objective_function
            or args.run_configuration):
       logger.error('Cannot use --just_extract_features flag in ' \
                    'combination with other options related to training a ' \
                    'model. Exiting.')
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
        json_encoder = JSONEncoder()
        json_decoder = JSONDecoder()

    # Get list of games
    if args.game_files == "all":
        game_files = [f for f in listdir(join(project_dir,
                                              'data')) if f.endswith('.txt')]
        del game_files[game_files.index('sample.txt')]
    else:
        game_files = args.game_files.split(',')

    # Get short names for the learner and objective function to use in the
    # experiment name(s)
    if len(args.learner) < 5:
        learner_short = args.learner
    elif args.learner.islower():
        learner_short = args.learner[:5]
    else:
        learner_short = '.{}'.format(''.join([c for c in args.learner if
                                              c.isupper()]))
    if len(args.objective_function) < 5:
        objective_function_short = args.objective_function
    elif args.objective_function.islower():
        objective_function_short = args.objective_function[:5]
    else:
        objective_function_short = \
            '.{}'.format(''.join([c for c in args.objective_function if
                                  c.isupper()]))

    # Train a combined model with all of the games or train models for each
    # individual game dataset
    if args.combine:

        combined_model_prefix = args.combined_model_prefix
        if args.do_not_lowercase_text:
            combined_model_prefix += '.nolc'
        if args.lowercase_cngrams:
            combined_model_prefix += '.lccngrams'
        if args.use_original_hours_values:
            combined_model_prefix += '.orghrs'
        if args.do_not_binarize_features:
            combined_model_prefix += '.nobin'
        jsonlines_filename = combined_model_prefix + '.jsonlines'
        jsonlines_filepath = join(project_dir,
                                  'working',
                                  jsonlines_filename)

        # Get experiment name
        expid = '{}.{}.{}'.format(combined_model_prefix,
                                  learner_short,
                                  objective_function_short)

        # Get config file path
        cfg_filename = '{}.train.cfg'.format(expid)

        if not args.run_configuration:

            logger.info('Extracting features to train a combined model with' \
                        ' training data from the following games: ' \
                        '{}'.format(', '.join(game_files)))

            # Initialize empty list for holding all of the feature
            # dictionaries from each review in each game and then extract
            # features from each game's training data
            feature_dicts = []

            logger.info('Writing {} to working directory' \
                        '...'.format(jsonlines_filename))
            jsonlines_file = open(jsonlines_filepath,
                                  'w')

            # Get the training reviews for this game from the MongoDB database
            for game_file in game_files:

                game = game_file[:-4]
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

                # Iterate over all training documents for the given game
                while game_docs.alive:

                    game_doc = game_docs.next()

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

                    jsonlines_file.write('{}\n'.format(
                        dumps({'id': str(game_doc['_id']),
                               'y': game_doc['hours_bin'] if bins else
                                   game_doc['hours'],
                               'x': features}).encode('utf-8').decode(
                                                                    'utf-8')))

            # Close JSONLINES file and take features out of memory
            jsonlines_file.close()
            features = None

            # Set up the job for training the model
            logger.info('Generating configuration file...')
            generate_config_file(expid,
                                 combined_model_prefix,
                                 args.learner,
                                 args.objective_function,
                                 project_dir,
                                 cfg_filename)

        if args.just_extract_features:
            logger.info('Complete.')
            sys.exit(0)

        # Make sure the jsonlines and config files exist
        if not any([exists(fpath) for fpath in [jsonlines_filepath,
                                                cfg_filepath]]):
            logging.error('Could not find either the .jsonlines file or the' \
                          ' config file or both ({}, ' \
                          '{})'.format(jsonlines_filepath,
                                       cfg_filepath))

        # Run the SKLL configuration, producing a model file
        logger.info('Training combined model...')
        run_configuration(cfg_filepath,
                          local=True)

    else:

        # Build model, extract features, etc. for each game separately
        for game_file in game_files:

            game = game_file[:-4]
            model_prefix = game
            if args.do_not_lowercase_text:
                model_prefix += '.nolc'
            if args.lowercase_cngrams:
                model_prefix += '.lccngrams'
            if args.use_original_hours_values:
                model_prefix += '.orghrs'
            if args.do_not_binarize_features:
                model_prefix += '.nobin'
            jsonlines_filename = model_prefix + '.jsonlines'
            jsonlines_filepath = join(project_dir,
                                      'working',
                                      jsonlines_filename)

            # Get experiment name
            expid = '{}.{}.{}'.format(model_prefix,
                                      learner_short,
                                      objective_function_short)

            # Get config file path
            cfg_filename = '{}.train.cfg'.format(expid)

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

                logger.info('Writing {} to working directory' \
                            '...'.format(jsonlines_filename))
                jsonlines_file = open(jsonlines_filepath,
                                      'w')

                # Iterate over all training documents for the given game
                while game_docs.alive:

                    game_doc = game_docs.next()

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

                # Set up the job for training the model
                logger.info('Generating configuration file...')
                generate_config_file(expid,
                                     model_prefix,
                                     args.learner,
                                     args.objective_function,
                                     project_dir,
                                     cfg_filename)

            if args.just_extract_features:
                logger.info('Complete.')
                sys.exit(0)

            # Make sure the jsonlines and config files exist
            if not any([exists(fpath) for fpath in [jsonlines_filepath,
                                                    cfg_filepath]]):
                logging.error('Could not find either the .jsonlines ' \
                              'file or the config file or both ({}, ' \
                              '{})'.format(jsonlines_filepath,
                                           cfg_filepath))

            # Run the SKLL configuration, producing a model file
            logger.info('Training model for {}...'.format(game))
            run_configuration(cfg_filepath,
                              local=True)

    logger.info('Complete.')