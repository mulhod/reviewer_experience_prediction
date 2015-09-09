'''
:author: Matt Mulholland, Janette Martinez, Emily Olshefski
:date: March 18, 2015

Script used to train models on datasets (or multiple datasets combined).
'''
import sys
from sys import exit
from time import sleep
from os import listdir
from os.path import (join,
                     exists,
                     dirname,
                     realpath,
                     splitext)
from collections import Counter
from argparse import (ArgumentParser,
                      ArgumentDefaultsHelpFormatter)

project_dir = dirname(dirname(realpath(__file__)))

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
        help='Comma-separated list of file-names or "all" for all of the '
             'files (the game files should reside in the "data" directory).',
        type=str,
        required=True)
    parser_add_argument('--combine',
        help='Combine all game files together to make one big model.',
        action='store_true',
        required=False)
    parser_add_argument('--combined_model_prefix',
        help='Prefix to use when naming the combined model (required if '
             'the --combine flag is used); basically, just combine the names '
             'of the games together in a way that doesn\'t make the file-name'
             '150 characters long...',
        type=str,
        required=False)
    parser_add_argument('--learner',
        help='Machine learning algorithm to use (only regressors are '
             'supported); both regular and rescaled versions of each learner '
             'are available.',
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
    parser_add_argument('--objective_function',
                        '-obj',
        help='Objective function used for tuning.',
        choices=['unweighted_kappa', 'linear_weighted_kappa',
                 'quadratic_weighted_kappa', 'uwk_off_by_one',
                 'lwk_off_by_one', 'qwk_off_by_one', 'r2',
                 'mean_squared_error'],
        default='quadratic_weighted_kappa')
    parser_add_argument('--do_not_lowercase_text',
        help='Do not make lower-casing part of the review text '
             'normalization step, which affects word n-gram-related '
             'features.',
        action='store_true',
        default=False)
    parser_add_argument('--lowercase_cngrams',
        help='Lower-case the review text before extracting character n-gram '
             'features.',
        action='store_true',
        default=False)
    parser_add_argument('--use_original_hours_values',
        help='Use the raw hours played values from Steam.',
        action='store_true',
        default=False)
    parser_add_argument('--just_extract_features',
        help='Extract features from all of the reviews, generate .jsonlines '
             'files, and generate configuration files, etc., but quit before '
             'training any models.',
        action='store_true',
        default=False)
    parser_add_argument('--reuse_features',
        help='Try to make use of previously-extracted features that reside in'
             ' the MongoDB database.',
        action='store_true',
        required=False)
    parser_add_argument('--finish_off_jsonlines_file',
        help='Read in .jsonlines file and finish it off with reviews that '
             'were not originally included (due to program crashing, for '
             'example). Note: Configuration options that were set when '
             'features were originally extracted from the earlier set of '
             'reviews are not remembered across sessions. It is up to the '
             'user to make a note of how features were being extracted, i.e.,'
             ' whether original hours values were being used, etc.',
        action='store_true',
        default=False)
    parser_add_argument('--run_configuration', '-run_cfg',
        help='Assumes feature/config files have already been generated and '
             'attempts to run the configuration; not needed to run training '
             'task under normal circumstances, so use only if you know what '
             'you are doing.',
         action='store_true',
         default=False)
    parser_add_argument('--do_not_binarize_features',
        help='Do not make all non-zero feature frequencies equal to 1.',
        action='store_true',
        default=False)
    parser_add_argument('--limit_training_samples', '-nsamples',
        help='Limit training samples to n samples. Will disregard if the '
             'value n is a value that exceeds the number of samples in the '
             'database. Defaults to no limit (-1).',
        type=int,
        default=-1)
    parser_add_argument('--use_cluster', '-cluster',
        help='If run on a compute cluster, make use of the cluster rather '
             'than running everything locally on one machine.',
        action='store_true',
        default=False)
    parser_add_argument('--partition',
        help='Data partition, i.e., "training", "test", etc. Value must be a '
             'valid "partition" value in the Mongo database.',
        type=str,
        default='training')
    parser_add_argument('--mongodb_port', '-dbport',
        help='Port that the MongoDB server is running.',
        type=int,
        default=27017)
    parser_add_argument('--log_file_path', '-log',
        help='Path to log file.',
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
    reuse_features = args.reuse_features
    nsamples = args.limit_training_samples
    finish_off_jsonlines_file = args.finish_off_jsonlines_file
    _run_configuration = args.run_configuration
    do_not_binarize_features = args.do_not_binarize_features
    local=not args.use_cluster
    partition = args.partition
    mongodb_port = args.mongodb_port

    # Imports
    from json import loads

    # Setup logger and create logging handlers
    import logging
    logger = logging.getLogger('train')
    logging_debug = logging.DEBUG
    logger.setLevel(logging_debug)
    loginfo = logger.info
    logdebug = logger.debug
    logerr = logger.error
    logwarn = logger.warning
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -'
                                  ' %(message)s')
    sh = logging.StreamHandler()
    sh.setLevel(logging_debug)
    fh = logging.FileHandler(realpath(args.log_file_path))
    fh.setLevel(logging_debug)
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)

    # Get paths to directories related to the training/evaluation tasks and
    # make them global variables
    global cfg_dir_path, working_dir_path
    cfg_dir_path = join(project_dir,
                        'config')
    working_dir_path = join(project_dir,
                            'working')
    data_dir_path = join(project_dir,
                         'data')
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
             .format(reuse_features))
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
             or learner != 'RescaledSVR'
             or objective_function != 'quadratic_weighted_kappa'
             or _run_configuration)):
       logerr('Cannot use --just_extract_features flag in combination with '
              'other options related to training a model. Exiting.')
       exit(1)

    if (_run_configuration
        and finish_off_jsonlines_file):
        logerr('Cannot use the --finish_off_jsonlines_file option in '
               'combination with the -run_cfg/--run_configuration option. '
               'Exiting.')
        exit(1)

    if nsamples:
        if _run_configuration:
            logerr('Cannot use the --limit_training_samples/-nsamples option '
                   'in combindation with the -run_cfg/--run_configuration '
                   'option. Exiting.')
            exit(1)
        if nsamples == -1:
            nsamples = 0
        elif nsamples == 0:
            logwarn('Using 0 for the training sample limit is the same as '
                    'setting it to no limit...')

    if not _run_configuration:
        # Import some functions, etc., that will only be needed if this code
        # gets executed
        from copy import deepcopy
        from util.mongodb import connect_to_db
        from util.datasets import get_game_files
        from src.features import (process_features,
                                  generate_config_file)

        # Establish connection to MongoDB database collection
        reviewdb = connect_to_db(mongodb_port)
        reviewdb.write_concern['w'] = 0

    if not just_extract_features:
        from skll import run_configuration

    # Get list of games
    game_files = get_game_files(game_files,
                                data_dir_path)

    # Get short versions of learner/objective function names
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
        jsonlines_file_name = ('{}{}.jsonlines'
                               .format(combined_model_prefix,
                                       '.{}'.format(partition)
                                           if partition != 'training'
                                           else ''))
        jsonlines_file_path = join(working_dir_path,
                                   jsonlines_file_name)

        finished_reviews_ids = []
        if finish_off_jsonlines_file:
            # Compile list of already-complete reviews
            if exists(jsonlines_file_path):
                finished_reviews_ids = [str(loads(l).get('id'))
                                        for l in open(jsonlines_file_path)]
            else:
                logerr('Could not find .jsonlines file at {}. Exiting.'
                       .format(jsonlines_file_path))
                exit(1)

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

            # Extract/get features from all training documents for the given
            # game, update the database, and write features to .jsonlines file
            loginfo('Writing {} to working directory...'
                    .format(jsonlines_file_name))
            with open(jsonlines_file_path,
                      'a' if finished_reviews_ids
                          else 'w') as jsonlines_file:
                # Get the training reviews for this game from the MongoDB
                # database
                for game_file in game_files:
                    game = splitext(game_file)[0]
                    loginfo('Extracting features from the training data for '
                            '{}...'.format(game))
                    process_features(reviewdb,
                                     partition,
                                     game,
                                     jsonlines_file=jsonlines_file,
                                     use_bins=bins,
                                     reuse_features=reuse_features,
                                     binarize_feats=binarize,
                                     lowercase_text=lowercase_text,
                                     lowercase_cngrams=lowercase_cngrams,
                                     ids_to_ignore=finished_reviews_ids,
                                     nsamples=nsamples)

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
                .format('locally' if local
                                  else 'on cluster'))
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
            # Build the .jsonlines file-name and append a suffix to reflect
            # the partition if the partition is anything other than "training"
            jsonlines_file_name = ('{}{}.jsonlines'
                                   .format(model_prefix,
                                           '.{}'.format(partition)
                                               if partition != 'training'
                                               else ''))
            jsonlines_file_path = join(working_dir_path,
                                       jsonlines_file_name)

            finished_reviews_ids = []
            if finish_off_jsonlines_file:
                # Compile list of already-complete reviews
                if exists(jsonlines_file_path):
                    finished_reviews_ids = [str(loads(l).get('id')) for l
                                            in open(jsonlines_file_path)]
                else:
                    logerr('Could not find .jsonlines file at {}. Exiting.'
                          .format(jsonlines_file_path))
                    exit(1)

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

                # Get the training reviews for this game from the Mongo
                # database
                loginfo('Extracting features from the training data for {}...'
                        .format(game))
                loginfo('Writing {} to working directory...'
                        .format(jsonlines_file_name))
                with open(jsonlines_file_path,
                          'a' if finished_reviews_ids
                              else 'w') as jsonlines_file:
                    process_features(reviewdb,
                                     partition,
                                     game,
                                     jsonlines_file=jsonlines_file,
                                     use_bins=bins,
                                     reuse_features=reuse_features,
                                     binarize_feats=binarize,
                                     lowercase_text=lowercase_text,
                                     lowercase_cngrams=lowercase_cngrams,
                                     ids_to_ignore=finished_reviews_ids,
                                     nsamples=nsamples)

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
                continue

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
                            'locally' if local
                                      else 'on cluster'))
            run_configuration(cfg_file_path,
                              local=local)

    loginfo('Complete.')
