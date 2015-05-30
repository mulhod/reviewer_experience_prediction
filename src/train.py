'''
:author: Matt Mulholland, Janette Martinez, Emily Olshefski
:date: March 18, 2015

Script used to train models on datasets (or multiple datasets combined).
'''
import logging
from sys import exit
from os import listdir
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from os.path import join, dirname, realpath, abspath, exists

project_dir = dirname(dirname(realpath(__file__)))


if __name__ == '__main__':

    parser = ArgumentParser(usage='python train.py --game_files GAME_FILE1,' \
                                  'GAME_FILE2,...[ OPTIONS]',
        description='Build a machine learning model based on the features ' \
                    'that are extracted from a set of reviews relating to a' \
                    ' specific game or set of games.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser_add_argument = parser.add_argument
    parser_add_argument('--game_files',
        help='comma-separated list of file-names or "all" for all of the ' \
             'files (the game files should reside in the "data" directory)',
        type=str,
        required=True)
    parser_add_argument('--combine',
        help='combine all game files together to make one big model',
        action='store_true',
        required=False)
    parser_add_argument('--combined_model_prefix',
        help='prefix to use when naming the combined model (required if ' \
             'the --combine flag is used); basically, just combine the ' \
             'names of the games together in a way that doesn\'t make the ' \
             'file-name 150 characters long...',
        type=str,
        required=False)
    parser_add_argument('--learner',
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
    parser_add_argument('--objective_function', '-obj',
        help='objective function used for tuning',
        choices=['unweighted_kappa', 'linear_weighted_kappa',
                 'quadratic_weighted_kappa', 'uwk_off_by_one',
                 'lwk_off_by_one', 'qwk_off_by_one', 'r2',
                 'mean_squared_error'],
        default='quadratic_weighted_kappa')
    parser_add_argument('--do_not_lowercase_text',
        help='do not make lower-casing part of the review text ' \
             'normalization step, which affects word n-gram-related ' \
             'features',
        action='store_true',
        default=False)
    parser_add_argument('--lowercase_cngrams',
        help='lower-case the review text before extracting character n-gram' \
             ' features',
        action='store_true',
        default=False)
    parser_add_argument('--use_original_hours_values',
        help='use the original, uncollapsed hours played values',
        action='store_true',
        default=False)
    parser_add_argument('--just_extract_features',
        help='extract features from all of the reviews, generate .jsonlines' \
             ' files, etc., but quit before training any models',
        action='store_true',
        default=False)
    parser_add_argument('--try_to_reuse_extracted_features',
        help='try to make use of previously-extracted features that reside ' \
             'in the MongoDB database',
        action='store_true',
        default=True)
    parser_add_argument('--run_configuration', '-run_cfg',
        help='assumes that .jsonlines files have already been created and ' \
             'attempts to run the configuration',
         action='store_true',
         default=False)
    parser_add_argument('--do_not_binarize_features',
        help='do not make all non-zero feature frequencies equal to 1',
        action='store_true',
        default=False)
    parser_add_argument('--mongodb_port', '-dbport',
        help='port that the MongoDB server is running',
        type=int,
        default=27017)
    parser_add_argument('--log_file_path', '-log',
        help='path for log file',
        type=str,
        default=join(project_dir,
                     'logs',
                     'replog_train.txt'))
    args = parser.parse_args()

    # Make local copies of arguments
    game_files = args.game_files
    combine = args.combine
    combined_model_prefix = args.combined_model_prefix
    learner = args.learner
    objective_function = args.objective_function
    do_not_lowercase_text = args.do_not_lowercase_text
    lowercase_cngrams = args.lowercase_cngrams
    use_original_hours_values = args.use_original_hours_values
    just_extract_features = args.just_extract_features
    try_to_reuse_extracted_features = args.try_to_reuse_extracted_features
    _run_configuration = args.run_configuration
    do_not_binarize_features = args.do_not_binarize_features

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

    loginfo = logger.info
    logdebug = logger.debug
    logerror = logger.error
    logwarn = logger.warning

    # Get paths to the project, data, working, and models directories

    # Print out some logging information about the upcoming tasks
    logdebug('project directory: {}'.format(project_dir))
    logdebug('Learner: {}'.format(learner))
    logdebug('Objective function used for tuning: ' \
             '{}'.format(objective_function))
    binarize = not do_not_binarize_features
    logdebug('Binarize features? {}'.format(binarize))
    lowercase_text = not do_not_lowercase_text
    logdebug('Lower-case text as part of the normalization step? ' \
             '{}'.format(lowercase_text))
    logdebug('Just extract features? {}'.format(just_extract_features))
    logdebug('Try to reuse extracted features? ' \
                 '{}'.format(try_to_reuse_extracted_features))
    bins = not use_original_hours_values
    logdebug('Use original hours values? {}'.format(not bins))

    # Make sure that, if --combine is being used, there is also a file prefix
    # being passed in via --combined_model_prefix for the combined model
    if combine and not combined_model_prefix:
        logerror('When using the --combine flag, you must also specify a ' \
                 'model prefix, which can be passed in via the ' \
                 '--combined_model_prefix option argument. Exiting.')
        exit(1)

    # Make sure command-line arguments make sense
    if just_extract_features \
       and (combine
            or combined_model_prefix
            or learner
            or objective_function
            or _run_configuration):
       logerror('Cannot use --just_extract_features flag in combination ' \
                'with other options related to training a model. Exiting.')
       exit(1)

    if not _run_configuration:
        # Import some functions, etc., that will only be needed if this code
        # gets executed
        from time import sleep
        from data import APPID_DICT
        from spacy.en import English
        from collections import Counter
        from json import JSONEncoder, JSONDecoder, dumps
        from pymongo import MongoClient
        from pymongo.errors import AutoReconnect, ConnectionFailure
        from src.feature_extraction import (Review,
                                            extract_features_from_review,
                                            generate_config_file)
        # Establish connection to MongoDB database
        connection_string = 'mongodb://localhost:{}'.format(args.mongodb_port)
        try:
            connection = MongoClient(connection_string)
        except ConnectionFailure as e:
            logerror('Unable to connect to to Mongo server at {}. ' \
                     'Exiting.'.format(connection_string))
            exit(1)
        db = connection['reviews_project']
        reviewdb = db['reviews']
        reviewdb.write_concern['w'] = 0
        reviewdb_find = reviewdb.find
        reviewdb_find_one = reviewdb.find_one
        reviewdb_update = reviewdb.update

        # Initialize an English-language spaCy NLP analyzer instance
        spaCy_nlp = English()

        # Initialize JSONEncoder, JSONDecoder objects
        json_encoder = JSONEncoder()
        json_encode = json_encoder.encode
        json_decoder = JSONDecoder()
        json_decode = json_decoder.decode

    if not just_extract_features:
        from skll import run_configuration

    # Get list of games
    if game_files == "all":
        game_files = [f for f in listdir(join(project_dir,
                                              'data')) if f.endswith('.txt')]
        del game_files[game_files.index('sample.txt')]
    else:
        game_files = game_files.split(',')

    # Get short names for the learner and objective function to use in the
    # experiment name(s)
    if len(learner) < 5:
        learner_short = learner
    elif learner.islower():
        learner_short = learner[:5]
    else:
        learner_short = '{}'.format(''.join([c for c in learner if
                                             c.isupper()]))
    if len(objective_function) < 5:
        objective_function_short = objective_function
    elif objective_function.islower():
        objective_function_short = objective_function[:5]
    else:
        objective_function_short = \
            '.{}'.format(''.join([c for c in objective_function if
                                  c.isupper()]))

    # Train a combined model with all of the games or train models for each
    # individual game dataset
    if combine:
        combined_model_prefix = combined_model_prefix
        if do_not_lowercase_text:
            combined_model_prefix = '{}.nolc'.format(combined_model_prefix)
        if lowercase_cngrams:
            combined_model_prefix = '{}.lccngrams'.format(
                                        combined_model_prefix)
        if use_original_hours_values:
            combined_model_prefix = '{}.orghrs'.format(combined_model_prefix)
        if do_not_binarize_features:
            combined_model_prefix = '{}.nobin'.format(combined_model_prefix)
        jsonlines_filename = '{}.jsonlines'.format(combined_model_prefix)
        jsonlines_filepath = join(project_dir,
                                  'working',
                                  jsonlines_filename)

        # Get experiment name
        expid = '{}.{}.{}'.format(combined_model_prefix,
                                  learner_short,
                                  objective_function_short)

        # Get config file path
        cfg_filename = '{}.train.cfg'.format(expid)
        cfg_filepath = join(project_dir,
                            'config',
                            cfg_filename)

        if not _run_configuration:

            loginfo('Extracting features to train a combined model with ' \
                    'training data from the following games: ' \
                    '{}'.format(', '.join(game_files)))

            # Initialize empty list for holding all of the feature
            # dictionaries from each review in each game and then extract
            # features from each game's training data
            feature_dicts = []

            loginfo('Writing {} to working directory' \
                    '...'.format(jsonlines_filename))
            jsonlines_file = open(jsonlines_filepath,
                                  'w')
            jsonlines_write = jsonlines_file.write

            # Get the training reviews for this game from the MongoDB database
            for game_file in game_files:

                game = game_file[:-4]
                loginfo('Extracting features from the training data for ' \
                        '{}...'.format(game))

                appid = APPID_DICT[game]
                game_docs = reviewdb_find({'game': game,
                                           'partition': 'training'},
                                          {'features': 0,
                                           'game': 0,
                                           'partition': 0})
                if game_docs.count() == 0:
                    logerror('No matching documents were found in the ' \
                             'MongoDB collection in the training partition ' \
                             'for game {}. Exiting.'.format(game))
                    exit(1)

                # Iterate over all training documents for the given game
                for game_doc in iter(game_docs):

                    _get = game_doc.get
                    if bins:
                        hours = _get('hours_bin')
                    else:
                        hours = _get('hours')
                    review_text = _get('review')
                    _id = _get('_id')
                    _binarized = _get('binarized')

                    # Instantiate a Review object
                    _Review = Review(review_text,
                                     hours,
                                     game,
                                     appid,
                                     spaCy_nlp,
                                     lower=lowercase_text)

                    # Extract features from the review text
                    found_features = False
                    if try_to_reuse_extracted_features:
                        features_doc = reviewdb_find_one(
                            {'_id': _id},
                            {'_id': 0,
                             'features': 1})
                        features = features_doc.get('features')
                        if features \
                           and _binarized == binarize:
                            features = json_decode(features)
                            found_features = True

                    if not found_features:
                        features = \
                           extract_features_from_review(
                               _Review,
                               lowercase_cngrams=lowercase_cngrams)

                    # If binarize is True, make all values 1
                    if binarize and not (found_features
                                         and _binarized):
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
                                reviewdb_update(
                              {'_id': _id},
                              {'$set': {'features': json_encode(features),
                                        'binarized': binarize}})
                                break
                            except AutoReconnect as e:
                                logwarn('Encountered ConnectionFailure ' \
                                        'error, attempting to reconnect ' \
                                        'automatically...')
                                tries += 1
                                if tries >= 5:
                                    logerror('Unable to update database ' \
                                             'even after 5 tries. Exiting.')
                                    exit(1)
                                sleep(20)

                    jsonlines_write('{}\n'.format(dumps(
                        {'id': str(_id),
                         'y': hours,
                         'x': features})))

            # Close JSONLINES file and take features out of memory
            jsonlines_file.close()
            features = None

            # Set up the job for training the model
            loginfo('Generating configuration file...')
            generate_config_file(expid,
                                 combined_model_prefix,
                                 learner,
                                 objective_function,
                                 project_dir,
                                 cfg_filename)

        if just_extract_features:
            loginfo('Complete.')
            exit(0)

        # Make sure the jsonlines and config files exist
        if not any([exists(fpath) for fpath in [jsonlines_filepath,
                                                cfg_filepath]]):
            logerror('Could not find either the .jsonlines file or the ' \
                     'config file or both ({}, {})'.format(jsonlines_filepath,
                                                           cfg_filepath))

        # Run the SKLL configuration, producing a model file
        loginfo('Training combined model...')
        run_configuration(cfg_filepath,
                          local=True)

    else:
        # Build model, extract features, etc. for each game separately
        for game_file in game_files:

            game = game_file[:-4]
            model_prefix = game
            if do_not_lowercase_text:
                model_prefix = '{}.nolc'.format(model_prefix)
            if lowercase_cngrams:
                model_prefix = '{}.lccngrams'.format(model_prefix)
            if use_original_hours_values:
                model_prefix = '{}.orghrs'.format(model_prefix)
            if do_not_binarize_features:
                model_prefix = '{}.nobin'.format(model_prefix)
            jsonlines_filename = '{}.jsonlines'.format(model_prefix)
            jsonlines_filepath = join(project_dir,
                                      'working',
                                      jsonlines_filename)

            # Get experiment name
            expid = '{}.{}.{}'.format(model_prefix,
                                      learner_short,
                                      objective_function_short)

            # Get config file path
            cfg_filename = '{}.train.cfg'.format(expid)
            cfg_filepath = join(project_dir,
                                'config',
                                cfg_filename)

            if not _run_configuration:

                loginfo('Extracting features to train a model with ' \
                        'training data from {}...'.format(game))

                # Initialize empty list for holding all of the feature
                # dictionaries from each review and then extract features from
                # all reviews
                feature_dicts = []

                # Get the training reviews for this game from the Mongo
                # database
                loginfo('Extracting features from the training data for ' \
                        ' {}...'.format(game))
                appid = APPID_DICT[game]
                game_docs = reviewdb_find({'game': game,
                                           'partition': 'training'},
                                          {'features': 0,
                                           'game': 0,
                                           'partition': 0})
                if game_docs.count() == 0:
                    logerror('No matching documents were found in the ' \
                             'MongoDB collection in the training partition ' \
                             'for game {}. Exiting.'.format(game))
                    exit(1)

                loginfo('Writing {} to working directory' \
                        '...'.format(jsonlines_filename))
                jsonlines_file = open(jsonlines_filepath,
                                      'w')
                jsonlines_write = jsonlines_file.write

                # Iterate over all training documents for the given game
                for game_doc in iter(game_docs):

                    _get = game_doc.get
                    if bins:
                        hours = _get('hours_bin')
                    else:
                        hours = _get('hours')
                    review_text = _get('review')
                    _id = _get('_id')
                    _binarized = _get('binarized')

                    # Instantiate a Review object
                    _Review = Review(review_text,
                                     hours,
                                     game,
                                     appid,
                                     spaCy_nlp,
                                     lower=lowercase_text)

                    # Extract features from the review text
                    found_features = False
                    if try_to_reuse_extracted_features:
                        features_doc = reviewdb_find_one(
                            {'_id': _id},
                            {'_id': 0,
                             'features': 1})
                        features = features_doc.get('features')
                        if features and _binarized == binarize:
                            features = json_decode(features)
                            found_features = True

                    if not found_features:
                        features = \
                            extract_features_from_review(
                                _Review,
                                lowercase_cngrams=lowercase_cngrams)

                    # If binarize is True, make all values 1
                    if binarize and not (found_features
                                         and _binarized):
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
                                reviewdb_update(
                                    {'_id': _id},
                                    {'$set': {'features':
                                                  json_encode(features),
                                              'binarized': binarize}})
                                break
                            except AutoReconnect as e:
                                logwarn('Encountered ConnectionFailure ' \
                                        'error, attempting to reconnect ' \
                                        'automatically...\n')
                                tries += 1
                                if tries >= 5:
                                    logerror('Unable to update database ' \
                                             'even after 5 tries. Exiting.')
                                    exit(1)
                                sleep(20)

                    # Write features to line of JSONLINES output file
                    jsonlines_write('{}\n'.format(dumps(
                        {'id': str(_id),
                         'y': hours,
                         'x': features})))

                # Close JSONLINES file and take features from memory
                jsonlines_file.close()
                features = None

                # Set up the job for training the model
                loginfo('Generating configuration file...')
                generate_config_file(expid,
                                     model_prefix,
                                     learner,
                                     objective_function,
                                     project_dir,
                                     cfg_filename)

            if just_extract_features:
                loginfo('Complete.')
                exit(0)

            # Make sure the jsonlines and config files exist
            if not any([exists(fpath) for fpath in [jsonlines_filepath,
                                                    cfg_filepath]]):
                logerror('Could not find either the .jsonlines ' \
                         'file or the config file or both ({}, ' \
                         '{})'.format(jsonlines_filepath,
                                      cfg_filepath))

            # Run the SKLL configuration, producing a model file
            loginfo('Training model for {}...'.format(game))
            run_configuration(cfg_filepath,
                              local=True)

    loginfo('Complete.')