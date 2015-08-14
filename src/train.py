'''
:author: Matt Mulholland, Janette Martinez, Emily Olshefski
:date: March 18, 2015

Script used to train models on datasets (or multiple datasets combined).
'''
from os.path import (join,
                     dirname,
                     realpath,
                     exists,
                     splitext)
from argparse import (ArgumentParser,
                      ArgumentDefaultsHelpFormatter)

project_dir = dirname(dirname(realpath(__file__)))


def make_train_dirs():
    '''
    Make sure that training-related directories exist.

    :returns: None
    '''

    from os import makedirs
    makedirs(cfg_dir_path,
             exist_ok=True)
    makedirs(working_dir_path,
             exist_ok=True)
    makedirs(join(project_dir,
                  'models'),
             exist_ok=True)
    makedirs(join(project_dir,
                  'logs'),
             exist_ok=True)
    makedirs(join(project_dir,
                  'predictions'),
             exist_ok=True)
    makedirs(join(project_dir,
                  'results'),
             exist_ok=True)
    makedirs(join(project_dir,
                  'outputs'),
             exist_ok=True)


if __name__ == '__main__':

    parser = ArgumentParser(usage='python train.py --game_files GAME_FILE1,'
                                  'GAME_FILE2,...[ OPTIONS]',
        description='Build a machine learning model based on the features '
                    'that are extracted from a set of reviews relating to a '
                    'specific game or set of games.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser_add_argument = parser.add_argument
    parser_add_argument('--game_files',
        help='comma-separated list of file-names or "all" for all of the '
             'files (the game files should reside in the "data" directory)',
        type=str,
        required=True)
    parser_add_argument('--combine',
        help='combine all game files together to make one big model',
        action='store_true',
        required=False)
    parser_add_argument('--combined_model_prefix',
        help='prefix to use when naming the combined model (required if '
             'the --combine flag is used); basically, just combine the names '
             'of the games together in a way that doesn\'t make the file-name'
             '150 characters long...',
        type=str,
        required=False)
    parser_add_argument('--learner',
        help='machine learning algorithm to use (only regressors are '
             'supported); both regular and rescaled versions of each learner '
             'are available',
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
        help='do not make lower-casing part of the review text '
             'normalization step, which affects word n-gram-related '
             'features',
        action='store_true',
        default=False)
    parser_add_argument('--lowercase_cngrams',
        help='lower-case the review text before extracting character n-gram '
             'features',
        action='store_true',
        default=False)
    parser_add_argument('--use_original_hours_values',
        help='use the original, uncollapsed hours played values',
        action='store_true',
        default=False)
    parser_add_argument('--just_extract_features',
        help='extract features from all of the reviews, generate .jsonlines '
             'files, etc., but quit before training any models',
        action='store_true',
        default=False)
    parser_add_argument('--try_to_reuse_extracted_features',
        help='try to make use of previously-extracted features that reside in'
             ' the MongoDB database',
        action='store_true',
        default=True)
    parser_add_argument('--run_configuration', '-run_cfg',
        help='assumes feature/config files have already been generated and '
             'attempts to run the configuration; not needed to run training '
             'task under normal circumstances, so use only if you know what '
             'you are doing',
         action='store_true',
         default=False)
    parser_add_argument('--do_not_binarize_features',
        help='do not make all non-zero feature frequencies equal to 1',
        action='store_true',
        default=False)
    parser_add_argument('--use_cluster', '-cluster',
        help='if run on a compute cluster, make use of the cluster rather '
             'than running everything locally on one machine',
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

    # Imports
    import logging
    from sys import exit
    from os import listdir

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
    local=not args.use_cluster

    # Initialize logging system
    logging_debug = logging.DEBUG
    logger = logging.getLogger('train')
    logger.setLevel(logging_debug)

    # Create file handler
    fh = logging.FileHandler(realpath(args.log_file_path))
    fh.setLevel(logging_debug)

    # Create console handler
    sh = logging.StreamHandler()
    sh.setLevel(logging_debug)

    # Add nicer formatting
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -'
                                  ' %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    loginfo = logger.info
    logdebug = logger.debug
    logerr = logger.error
    logwarn = logger.warning

    # Get paths to directories related to the training/evaluation tasks and
    # make them global variables
    global cfg_dir_path, working_dir_path
    cfg_dir_path = join(project_dir,
                        'config')
    working_dir_path = join(project_dir,
                            'working')
    make_train_dirs()

    # Print out some logging information about the upcoming tasks
    logdebug('project directory: {}'.format(project_dir))
    logdebug('Learner: {}'.format(learner))
    logdebug('Objective function used for tuning: {}'
             .format(objective_function))
    binarize = not do_not_binarize_features
    logdebug('Binarize features? {}'.format(binarize))
    lowercase_text = not do_not_lowercase_text
    logdebug('Lower-case text as part of the normalization step? {}'
             .format(lowercase_text))
    logdebug('Just extract features? {}'.format(just_extract_features))
    logdebug('Try to reuse extracted features? {}'
             .format(try_to_reuse_extracted_features))
    bins = not use_original_hours_values
    logdebug('Use original hours values? {}'.format(not bins))

    # Make sure that, if --combine is being used, there is also a file prefix
    # being passed in via --combined_model_prefix for the combined model
    if (combine
        and not combined_model_prefix):
        logerr('When using the --combine flag, you must also specify a '
               'model prefix, which can be passed in via the '
               '--combined_model_prefix option argument. Exiting.')
        exit(1)

    # Make sure command-line arguments make sense
    if (just_extract_features
        and (combine
             or combined_model_prefix
             or learner
             or objective_function
             or _run_configuration)):
       logerr('Cannot use --just_extract_features flag in combination with '
              'other options related to training a model. Exiting.')
       exit(1)

    if not _run_configuration:
        # Import some functions, etc., that will only be needed if this code
        # gets executed
        from time import sleep
        from copy import deepcopy
        from data import APPID_DICT
        from spacy.en import English
        from collections import Counter
        from json import (JSONEncoder,
                          dumps)
        from pymongo import MongoClient
        from pymongo.errors import (AutoReconnect,
                                    ConnectionFailure)
        from src.feature_extraction import (Review,
                                            extract_features_from_review,
                                            generate_config_file)
        from util.mongodb import get_review_features_from_db
        # Establish connection to MongoDB database
        connection_string = 'mongodb://localhost:{}'.format(args.mongodb_port)
        try:
            connection = MongoClient(connection_string)
        except ConnectionFailure as e:
            logerr('Unable to connect to to Mongo server at {}. Exiting.'
                   .format(connection_string))
            exit(1)
        db = connection['reviews_project']
        reviewdb = db['reviews']
        reviewdb.write_concern['w'] = 0
        reviewdb_find = reviewdb.find
        reviewdb_update = reviewdb.update

        # Initialize an English-language spaCy NLP analyzer instance
        spaCy_nlp = English()

        # Make local binding to JSONEncoder method attribute
        json_encoder = JSONEncoder()
        json_encode = json_encoder.encode

    if not just_extract_features:
        from skll import run_configuration

    # Get list of games
    if game_files == "all":
        game_files = [f for f in listdir(join(project_dir,
                                              'data'))
                      if f.endswith('.jsonlines')]
        del game_files[game_files.index('sample.jsonlines')]
    else:
        game_files = game_files.split(',')

    # Get short names for the learner and objective function to use in the
    # experiment name(s)
    learner_abbrs = {'AdaBoost': 'AdaBoost',
                     'DecisionTree': 'DTree',
                     'ElasticNet': 'ENet',
                     'GradientBoostingRegressor': 'GBReg',
                     'KNeighborsRegressor': 'KNReg',
                     'Lasso': 'Lasso',
                     'LinearRegression': 'LReg',
                     'RandomForestRegressor': 'RFReg',
                     'Ridge': 'Ridge',
                     'SGDRegressor': 'SGDReg',
                     'SVR': 'SVR',
                     'RescaledAdaBoost': 'RAdaBoost',
                     'DecisionTree': 'RDTree',
                     'ElasticNet': 'RENet',
                     'GradientBoostingRegressor': 'RGBReg',
                     'KNeighborsRegressor': 'RKNReg',
                     'Lasso': 'RLasso',
                     'LinearRegression': 'RLReg',
                     'RandomForestRegressor': 'RRFReg',
                     'Ridge': 'RRidge',
                     'SGDRegressor': 'RSGDReg',
                     'RescaledSVR': 'RSVR'}
    obj_func_abbrs = {'unweighted_kappa': 'uwk',
                      'linear_weighted_kappa': 'lwk',
                      'quadratic_weighted_kappa': 'qwk',
                      'uwk_off_by_one': 'kappa_offby1',
                      'lwk_off_by_one': 'lwk_offby1',
                      'qwk_off_by_one': 'qwk_offby1',
                      'r2': 'r2',
                      'mean_squared_error': 'mse'}
    learner_short = learner_abbrs[learner]
    objective_function_short = obj_func_abbrs[objective_function]

    # Train a combined model with all of the games or train models for each
    # individual game dataset
    if combine:
        combined_model_prefix = combined_model_prefix
        if do_not_lowercase_text:
            combined_model_prefix = '{}.nolc'.format(combined_model_prefix)
        if lowercase_cngrams:
            combined_model_prefix = ('{}.lccngrams'
                                     .format(combined_model_prefix))
        if use_original_hours_values:
            combined_model_prefix = '{}.orghrs'.format(combined_model_prefix)
        if do_not_binarize_features:
            combined_model_prefix = '{}.nobin'.format(combined_model_prefix)
        jsonlines_file_name = '{}.jsonlines'.format(combined_model_prefix)
        jsonlines_file_path = join(working_dir_path,
                                   jsonlines_file_name)

        # Get experiment name
        expid = '{}.{}.{}'.format(combined_model_prefix,
                                  learner_short,
                                  objective_function_short)

        # Get config file path and make path to 'config' directory if it
        # doesn't exist
        cfg_file_name = '{}.train.cfg'.format(expid)
        cfg_file_path = join(cfg_dir_path,
                             cfg_file_name)

        if not _run_configuration:

            loginfo('Extracting features to train a combined model with '
                    'training data from the following games: {}'
                    .format(', '.join(game_files)))

            # Initialize empty list for holding all of the feature
            # dictionaries from each review in each game and then extract
            # features from each game's training data
            feature_dicts = []

            loginfo('Writing {} to working directory...'
                    .format(jsonlines_file_name))
            jsonlines_file = open(jsonlines_file_path,
                                  'w')
            jsonlines_write = jsonlines_file.write

            # Get the training reviews for this game from the MongoDB database
            for game_file in game_files:

                game = splitext(game_file)[0]
                loginfo('Extracting features from the training data for {}...'
                        .format(game))

                appid = APPID_DICT[game]
                game_docs = reviewdb_find({'game': game,
                                           'partition': 'training'},
                                          {'features': 0,
                                           'game': 0,
                                           'partition': 0})
                if game_docs.count() == 0:
                    logerr('No matching documents were found in the MongoDB '
                           'collection in the training partition for game {}.'
                           ' Exiting.'.format(game))
                    exit(1)

                # Iterate over all training documents for the given game
                for game_doc in iter(game_docs):

                    _get = game_doc.get
                    hours = _get('total_game_hours_bin'
                                 if bins
                                 else 'total_game_hours')
                    review_text = _get('review')
                    _id = _get('_id')
                    _binarized = _get('binarized')

                    # Extract NLP features by querying the database (if they
                    # are available and the --try_to_reuse_extracted_features
                    # flag was used); otherwise, extract features from the
                    # review text directly (and try to update the database)
                    found_features = False
                    if (try_to_reuse_extracted_features
                        and _binarized == binarize):
                        features = get_review_features_from_db(reviewdb,
                                                               _id)
                        found_features = True if features else False

                    if not found_features:
                        features = extract_features_from_review(
                                       Review(review_text,
                                              hours,
                                              game,
                                              appid,
                                              spaCy_nlp,
                                              lower=lowercase_text),
                                       lowercase_cngrams=lowercase_cngrams)

                    # If binarize is True, make all NLP feature values 1
                    # (except for the mean cosine similarity feature
                    if (binarize
                        and not (found_features
                                 and _binarized)):
                        _features = deepcopy(features)
                        del _features['mean_cos_sim']
                        _features = dict(Counter(list(_features)))
                        _features['mean_cos_sim'] = features['mean_cos_sim']
                        features = _features

                    # Update Mongo database game doc with new key "features",
                    # which will be mapped to NLP features, and a new key
                    # "binarized", which will be set to True if NLP features
                    # were extracted with the --do_not_binarize_features flag
                    # or False otherwise
                    if not found_features:
                        tries = 0
                        while tries < 5:
                            try:
                                reviewdb_update(
                              {'_id': _id},
                              {'$set': {'features': json_encode(features),
                                        'binarized': binarize}})
                                break
                            except AutoReconnect:
                                logwarn('Encountered ConnectionFailure error,'
                                        ' attempting to reconnect '
                                        'automatically...')
                                tries += 1
                                if tries >= 5:
                                    logerr('Unable to update database even '
                                           'after 5 tries. Exiting.')
                                    exit(1)
                                sleep(20)

                    # Get features collected from Steam (non-NLP features) and
                    # add them to the features dictionary
                    achievement_dict = _get('achievement_progress')
                    features.update(
                        {'total_game_hours_last_two_weeks':
                             _get('total_game_hours_last_two_weeks'),
                         'num_found_funny': _get('num_found_funny'),
                         'num_found_helpful': _get('num_found_helpful'),
                         'found_helpful_percentage':
                             _get('found_helpful_percentage'),
                         'num_friends': _get('num_friends'),
                         'friend_player_level': _get('friend_player_level'),
                         'num_groups': _get('num_groups'),
                         'num_screenshots': _get('num_screenshots'),
                         'num_workshop_items': _get('num_workshop_items'),
                         'num_comments': _get('num_comments'),
                         'num_games_owned': _get('num_games_owned'),
                         'num_reviews': _get('num_reviews'),
                         'num_guides': _get('num_guides'),
                         'num_badges': _get('num_badges'),
                         'updated': 1 if _get('date_updated') else 0,
                         'num_achievements_attained':
                             (achievement_dict
                              .get('num_achievements_attained')),
                         'num_achievements_percentage':
                             (achievement_dict
                              .get('num_achievements_percentage')),
                         'rating': (1 if _get('rating') == "Recommended"
                                      else 0)})

                    # If any features have a value of None, then turn the
                    # values into zeroes
                    [features.pop(k) for k in features if not features[k]]

                    # Write JSON object to file
                    jsonlines_write('{}\n'.format(dumps({'id': hash(str(_id)),
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
                                 cfg_file_path)

        if just_extract_features:
            loginfo('Complete.')
            exit(0)

        # Make sure the jsonlines and config files exist
        if not all([exists(fpath) for fpath in [jsonlines_file_path,
                                                cfg_file_path]]):
            logerr('Could not find either the .jsonlines file or the config '
                   'file or both ({}, {})'.format(jsonlines_file_path,
                                                  cfg_file_path))
            exit(1)

        # Run the SKLL configuration, producing a model file
        loginfo('Training combined model {}...'
                .format('locally' if local else 'on cluster'))
        run_configuration(cfg_file_path,
                          local=local)

    else:
        # Build model, extract features, etc. for each game separately
        for game_file in game_files:

            game = splitext(game_file)[0]
            model_prefix = game
            if do_not_lowercase_text:
                model_prefix = '{}.nolc'.format(model_prefix)
            if lowercase_cngrams:
                model_prefix = '{}.lccngrams'.format(model_prefix)
            if use_original_hours_values:
                model_prefix = '{}.orghrs'.format(model_prefix)
            if do_not_binarize_features:
                model_prefix = '{}.nobin'.format(model_prefix)
            jsonlines_file_name = '{}.jsonlines'.format(model_prefix)
            jsonlines_file_path = join(working_dir_path,
                                       jsonlines_file_name)

            # Get experiment name
            expid = '{}.{}.{}'.format(model_prefix,
                                      learner_short,
                                      objective_function_short)

            # Get config file path
            cfg_file_name = '{}.train.cfg'.format(expid)
            cfg_file_path = join(cfg_dir_path,
                                 cfg_file_name)

            if not _run_configuration:

                loginfo('Extracting features to train a model with training '
                        'data from {}...'.format(game))

                # Initialize empty list for holding all of the feature
                # dictionaries from each review and then extract features from
                # all reviews
                feature_dicts = []

                # Get the training reviews for this game from the Mongo
                # database
                loginfo('Extracting features from the training data for {}...'
                        .format(game))
                appid = APPID_DICT[game]
                game_docs = reviewdb_find({'game': game,
                                           'partition': 'training'},
                                          {'features': 0,
                                           'game': 0,
                                           'partition': 0})
                if game_docs.count() == 0:
                    logerr('No matching documents were found in the MongoDB '
                           'collection in the training partition for game {}.'
                           ' Exiting.'.format(game))
                    exit(1)

                loginfo('Writing {} to working directory...'
                        .format(jsonlines_file_name))
                jsonlines_file = open(jsonlines_file_path,
                                      'w')
                jsonlines_write = jsonlines_file.write

                # Iterate over all training documents for the given game
                for game_doc in iter(game_docs):
                    _get = game_doc.get
                    hours = _get('total_game_hours_bin'
                                 if bins
                                 else 'total_game_hours')
                    review_text = _get('review')
                    _id = _get('_id')
                    _binarized = _get('binarized')

                    # Extract NLP features by querying the database (if they
                    # are available and the --try_to_reuse_extracted_features
                    # flag was used); otherwise, extract features from the
                    # review text directly (and try to update the database)
                    found_features = False
                    if (try_to_reuse_extracted_features
                        and _binarized == binarize):
                        features = get_review_features_from_db(reviewdb,
                                                               _id)
                        found_features = True if features else False

                    if not found_features:
                        # Instantiate a Review object
                        features = extract_features_from_review(
                                       Review(review_text,
                                              hours,
                                              game,
                                              appid,
                                              spaCy_nlp,
                                              lower=lowercase_text),
                                       lowercase_cngrams=lowercase_cngrams)

                    # If binarize is True, make all NLP feature values 1
                    # (except for the mean cosine similarity feature
                    if (binarize
                        and not (found_features
                                 and _binarized)):
                        _features = deepcopy(features)
                        del _features['mean_cos_sim']
                        _features = dict(Counter(list(_features)))
                        _features['mean_cos_sim'] = features['mean_cos_sim']
                        features = _features

                    # Update Mongo database game doc with new key "features",
                    # which will be mapped to NLP features, and a new key
                    # "binarized", which will be set to True if NLP features
                    # were extracted with the --do_not_binarize_features flag
                    # or False otherwise
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
                            except AutoReconnect:
                                logwarn('Encountered ConnectionFailure error,'
                                        ' attempting to reconnect '
                                        'automatically...\n')
                                tries += 1
                                if tries >= 5:
                                    logerr('Unable to update database even '
                                           'after 5 tries. Exiting.')
                                    exit(1)
                                sleep(20)

                    # Get features collected from Steam (non-NLP features) and
                    # add them to the features dictionary
                    achievement_dict = _get('achievement_progress')
                    features.update(
                        {'total_game_hours_last_two_weeks':
                             _get('total_game_hours_last_two_weeks'),
                         'num_found_funny': _get('num_found_funny'),
                         'num_found_helpful': _get('num_found_helpful'),
                         'found_helpful_percentage':
                             _get('found_helpful_percentage'),
                         'num_friends': _get('num_friends'),
                         'friend_player_level': _get('friend_player_level'),
                         'num_groups': _get('num_groups'),
                         'num_screenshots': _get('num_screenshots'),
                         'num_workshop_items': _get('num_workshop_items'),
                         'num_comments': _get('num_comments'),
                         'num_games_owned': _get('num_games_owned'),
                         'num_reviews': _get('num_reviews'),
                         'num_guides': _get('num_guides'),
                         'num_badges': _get('num_badges'),
                         'updated': 1 if _get('date_updated') else 0,
                         'num_achievements_attained':
                             (achievement_dict
                              .get('num_achievements_attained')),
                         'num_achievements_percentage':
                             (achievement_dict
                              .get('num_achievements_percentage')),
                         'rating': _get('rating')})

                    # If any features have a value of None, then turn the
                    # values into zeroes
                    [features.update({k: 0}) for k, v in features.items()
                     if v == None]

                    # Write features to line of JSONLINES output file
                    jsonlines_write('{}\n'.format(dumps({'id': hash(str(_id)),
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
                                     cfg_file_path)

            if just_extract_features:
                loginfo('Complete.')
                exit(0)

            # Make sure the jsonlines and config files exist
            if not all([exists(fpath) for fpath in [jsonlines_file_path,
                                                    cfg_file_path]]):
                logerr('Could not find either the .jsonlines file or the '
                       'config file or both ({}, {})'
                       .format(jsonlines_file_path,
                               cfg_file_path))
                exit(1)

            # Run the SKLL configuration, producing a model file
            loginfo('Training model for {} {}...'
                    .format(game,
                            'locally' if local else 'on cluster'))
            run_configuration(cfg_file_path,
                              local=local)

    loginfo('Complete.')