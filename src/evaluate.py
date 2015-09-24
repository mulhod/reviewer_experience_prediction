'''
:author: Matt Mulholland
:date: May 13, 2015

Script used to make predictions for datasets (or multiple datasets combined) and generate evaluation metrics.
'''
from os.path import (join,
                     exists,
                     dirname,
                     abspath,
                     realpath,
                     splitext)
from argparse import (ArgumentParser,
                      ArgumentDefaultsHelpFormatter)

# Get path to project and data directories
project_dir = dirname(dirname(realpath(__file__)))
data_dir = join(project_dir,
                'data')


def main():
    parser = ArgumentParser(usage='python evaluate.py --game_files '
        'GAME_FILE1,GAME_FILE2,... --model MODEL_PREFIX[ --results_path '
        'PATH|--predictions_path PATH|--just_extract_features][ OPTIONS]',
        description='Generate predictions for a data-set\'s test set '
                    'reviews and output evaluation metrics.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser_add_argument = parser.add_argument
    parser_add_argument('--game_files',
        help='Comma-separated list of file-names or "all" for all of the '
             'files (the game files should reside in the "data" directory; '
             'the .jsonlines suffix is not necessary, but the file-names '
             'should be exact matches otherwise).',
        type=str,
        required=True)
    parser_add_argument('--model', '-m',
        help='Model prefix (this will be the model that is used to generate '
             'predictions for all test reviews for the game files input via '
             'the --game_files option argument).',
        type=str,
        required=True)
    parser_add_argument('--results_path', '-r',
        help='Destination directory for results output file.',
        type=str,
        required=False)
    parser_add_argument('--predictions_path', '-p',
        help='Destination directory for predictions file.',
        type=str,
        required=False)
    parser_add_argument('--do_not_lowercase_text',
        help='Do not make lower-casing part of the review text normalization '
             'step, which affects word n-gram-related features.',
        action='store_true',
        default=False)
    parser_add_argument('--lowercase_cngrams',
        help='Lower-case the review text before extracting character n-gram '
             'features.',
        action='store_true',
        default=False)
    parser_add_argument('--use_original_hours_values',
        help='Use the original, uncollapsed hours played values.',
        action='store_true',
        default=False)
    parser_add_argument('--just_extract_features',
        help='Extract features from all of the test set reviews and insert '
             'them into the MongoDB database, but quit before generating '
             'any predictions or results.',
        action='store_true',
        default=False)
    parser_add_argument('--reuse_features',
        help='Try to make use of previously-extracted features that reside '
             'in the MongoDB database.',
        action='store_true',
        default=True)
    parser_add_argument('--do_not_binarize_features',
        help='Do not make all non-zero feature frequencies equal to 1.',
        action='store_true',
        default=False)
    parser_add_argument('--eval_combined_games',
        help='Print evaluation metrics across all games.',
        action='store_true',
        default=False)
    parser_add_argument('--partition',
        help='Data partition, i.e., "test", "training", etc. Value must be a '
             'valid "partition" value in the Mongo database.',
        type=str,
        default='test')
    parser_add_argument('-dbhost', '--mongodb_host',
        help='Host that the MongoDB server is running on.',
        type=str,
        default='localhost')
    parser_add_argument('-dbport', '--mongodb_port',
        help='Port that the MongoDB server is running on.',
        type=int,
        default=27017)
    parser_add_argument('--log_file_path', '-log',
        help='Destination path for log file.',
        type=str,
        default=join(project_dir,
                     'logs',
                     'replog_eval.txt'))
    args = parser.parse_args()

    # Imports
    import logging
    from sys import exit
    from os import listdir
    from collections import Counter
    from util.mongodb import connect_to_db
    from util.datasets import get_game_files
    from src.features import (process_features,
                              make_confusion_matrix,
                              extract_features_from_review)

    # Make local copies of arguments
    game_files = args.game_files
    model = args.model
    results_path = args.results_path
    predictions_path = args.predictions_path
    do_not_lowercase_text = args.do_not_lowercase_text
    lowercase_cngrams = args.lowercase_cngrams
    use_original_hours_values = args.use_original_hours_values
    just_extract_features = args.just_extract_features
    reuse_features = args.reuse_features
    do_not_binarize_features = args.do_not_binarize_features
    eval_combined_games = args.eval_combined_games
    partition = args.partition
    mongodb_host = args.mongodb_host
    mongodb_port = args.mongodb_port

    # Initialize logging system
    logging_debug = logging.DEBUG
    logger = logging.getLogger('eval')
    logger.setLevel(logging_debug)

    # Create file handler
    fh = logging.FileHandler(abspath(args.log_file_path))
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

    # Redirect warnings to the logging system
    logging.captureWarnings(True)

    loginfo = logger.info
    logdebug = logger.debug
    logerr = logger.error
    logwarn = logger.warning

    # Get predictions/results output file path and models directory path and
    # make sure model exists
    if not just_extract_features:

        # Import methods and stuff that will be used if not only feature
        # extraction is being done
        from numpy import (array,
                           chararray)
        from skll import Learner
        load_learner = Learner.from_file
        from skll.metrics import (kappa,
                                  pearson)
        from skll.data.featureset import FeatureSet

        # Since the skll Learner.predict method will probably print out a
        # warning each time a prediction is made, let's try to stop the
        # warnings in all but the first case
        import warnings
        warnings.filterwarnings('once')

        # Get directories and paths
        models_dir = join(project_dir,
                          'models')
        if not exists(join(models_dir,
                           '{}.model'.format(model))):
            logerr('Could not find model with prefix {} in models directory '
                   '({}). Exiting.'.format(model,
                                           models_dir))
            exit(1)
        paths = []
        if predictions_path:
            predictions_path = abspath(predictions_path)
            paths.append(predictions_path)
        if results_path:
            results_path = abspath(results_path)
            paths.append(results_path)
        if (paths
            and not any(map(exists,
                            paths))):
            logerr('Could not verify the existence of the destination '
                   'directories for the predictions and/or results output '
                   'files. Exiting.')
            exit(1)

    # Make sure command-line arguments make sense
    if (just_extract_features
        and (results_path
             or predictions_path
             or eval_combined_games)):
        logerr('If the --just_extract_features flag is used, then any other '
               'flags used for evaluation-related tasks cannot be used '
               '(since the program skips evaluation if it is just extracting '
               'the features and putting them in the MondoDB). Exiting.')
        exit(1)

    if (reuse_features
        and (lowercase_cngrams
             or do_not_lowercase_text)):
        logwarn('If trying to reuse previously extracted features, then the '
                'values picked for the --lowercase_cngrams and '
                '--do_not_lowercase_text should match the values used to '
                'build the models.')

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

    # Establish connection to MongoDB database collection
    loginfo('Connecting to MongoDB database on mongodb://{}:{}...'
            .format(mongodb_host,
                    mongodb_port))
    reviewdb = connect_to_db(host=mongodb_host,
                             port=mongodb_port)
    reviewdb.write_concern['w'] = 0

    # Iterate over the game files, looking for test set reviews
    # Get list of games
    game_files = get_game_files(game_files,
                                data_dir)

    # If the --eval_combined_games flag was used but there's only one game
    # file to evaluate on, print a warning that the "combined" stats will only
    # be for one game...
    if (eval_combined_games
        and len(game_files) == 1):
        logwarn('The --eval_combined_games flag was used, but there was only '
                'one game that predictions were generated for.')

    if model:
        loginfo('Loading model file: {} ...'.format(model))
        learner = load_learner(join(models_dir,
                                    '{}.model'.format(model)))
        learner_predict = learner.predict

    # Lists of original and predicted hours values
    total_hours_values = []
    total_hours_values_extend = total_hours_values.extend
    total_predicted_hours_labels = []
    total_predicted_hours_labels_extend = total_predicted_hours_labels.extend

    # Iterate over game files, generating/fetching features, etc., and putting
    # them in lists
    for game_file in game_files:
        game = splitext(game_file)[0]
        # Get/generate test review features, etc.
        loginfo('Extracting features from the test data for {}'
                '...'.format(game))
        if just_extract_features:
            process_features(reviewdb,
                             partition,
                             game,
                             just_extract_features=True,
                             use_bins=bins,
                             reuse_features=reuse_features,
                             binarize_feats=binarize,
                             lowercase_text=lowercase_text,
                             lowercase_cngrams=lowercase_cngrams)
            loginfo('Exiting after extracting features and updating the '
                    'database...')
            exit(0)
        else:
            review_data_dicts = \
                process_features(reviewdb,
                                 partition,
                                 game,
                                 review_data=True,
                                 use_bins=bins,
                                 reuse_features=reuse_features,
                                 binarize_feats=binarize,
                                 lowercase_text=lowercase_text,
                                 lowercase_cngrams=lowercase_cngrams)

        # Extract values from review dictionaries
        hours_values = []
        features_dicts = []
        _ids = []
        reviews = []
        for review_dict in review_data_dicts:
            hours_values.append(int(review_dict.get('hours')))
            features_dicts.append(review_dict.get('features'))
            _ids.append(review_dict.get('_id'))
            reviews.append(review_dict.get('review'))
        review_data_dicts = None

        # Make list of FeatureSet instances
        fs = FeatureSet('{}.test'.format(game),
                        array(_ids,
                              dtype=chararray),
                        features=features_dicts)

        # Generate predictions
        predictions = learner_predict(fs)
        predicted_labels = [int(p) for p in predictions]

        # Make sure all the lists are equal
        if not any([map(lambda x, y: len(x) == len(y),
                        [_ids, reviews, hours_values, predicted_labels])]):
            logerr('Lists of values not of expected length:\n\n{}\n\nExiting.'
                   .format(str([_ids,
                                reviews,
                                hours_values,
                                predicted_labels])))
            exit(1)

        # Save predicted/expected values for final evaluation
        total_predicted_hours_labels_extend(predicted_labels)
        total_hours_values_extend(hours_values)

        if predictions_path:
            from features import write_predictions_to_file
            # Write predictions file for game
            loginfo('Writing predictions file for {}...'.format(game))
            write_predictions_to_file(predictions_path,
                                      game,
                                      model,
                                      zip(_ids,
                                          reviews,
                                          hours_values,
                                          predicted_labels))

        if results_path:
            from features import write_results_file
            # Write evaluation report for game
            loginfo('Writing results file for {}...'.format(game))
            write_results_file(results_path,
                               game,
                               model,
                               hours_values,
                               predicted_labels)

    # Do evaluation on all predicted/expected values across all games or exit
    if not eval_combined_games:
        loginfo('Complete.')
        exit(0)
    loginfo('Printing out evaluation metrics for the performance of the '
            'model across all games...')
    loginfo('Using predicted/expected values for the following games: {}'
            .format(', '.join(game_files)))
    loginfo('Kappa: {}'.format(kappa(total_hours_values,
                                     total_predicted_hours_labels)))
    loginfo('Kappa (allow off-by-one): {}'
            .format(kappa(total_hours_values,
                          total_predicted_hours_labels,
                          allow_off_by_one=True)))
    loginfo('Pearson: {}'
            .format(pearson(total_hours_values,
                            total_predicted_hours_labels)))
    loginfo('Confusion Matrix (predicted along top, actual along side)\n\n{}'
            .format(
                make_confusion_matrix(
                    total_hours_values,
                    total_predicted_hours_labels)['string']))
    loginfo('Complete.')

if __name__ == '__main__':
    main()
