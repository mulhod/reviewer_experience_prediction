'''
:author: Matt Mulholland
:date: May 13, 2015

Script used to make predictions for datasets (or multiple datasets combined) and generate evaluation metrics.
'''
from os.path import realpath, dirname, abspath, join, exists
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

project_dir = dirname(dirname(realpath(__file__)))


if __name__ == '__main__':

    parser = ArgumentParser(usage='python evaluate.py --game_files' \
        ' GAME_FILE1,GAME_FILE2,... --model MODEL_PREFIX[ --results_path ' \
        'PATH|--predictions_path PATH|--just_extract_features][ OPTIONS]',
        description='generate predictions for a data-set\'s test set ' \
                    'reviews and output evaluation metrics',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser_add_argument = parser.add_argument
    parser_add_argument('--game_files',
        help='comma-separated list of file-names or "all" for all of the ' \
             'files (the game files should reside in the "data" directory)',
        type=str,
        required=True)
    parser_add_argument('--model', '-m',
        help='model prefix (this will be the model that is used to generate' \
             ' predictions for all test reviews for the game files input ' \
             'via the --game_files option argument)',
        type=str,
        required=True)
    parser_add_argument('--results_path', '-r',
        help='destination directory for results output file',
        type=str,
        required=False)
    parser_add_argument('--predictions_path', '-p',
        help='destination directory for predictions file',
        type=str,
        required=False)
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
        help='extract features from all of the test set reviews and insert ' \
             'them into the MongoDB database, but quit before generating ' \
             'any predictions or results',
        action='store_true',
        default=False)
    parser_add_argument('--try_to_reuse_extracted_features',
        help='try to make use of previously-extracted features that reside ' \
             'in the MongoDB database',
        action='store_true',
        default=True)
    parser_add_argument('--do_not_binarize_features',
        help='do not make all non-zero feature frequencies equal to 1',
        action='store_true',
        default=False)
    parser_add_argument('--eval_combined_games',
        help='print evaluation metrics across all games',
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
                     'replog_eval.txt'))
    args = parser.parse_args()

    # Imports
    import logging
    from sys import exit
    from os import listdir
    from data import APPID_DICT
    from spacy.en import English
    from collections import Counter
    from pymongo import MongoClient
    from json import JSONEncoder, JSONDecoder
    from pymongo.errors import AutoReconnect, ConnectionFailure
    from src.feature_extraction import (Review,
                                        extract_features_from_review,
                                        make_confusion_matrix)

    # Make local copies of arguments
    game_files = args.game_files
    model = args.model
    results_path = args.results_path
    predictions_path = args.predictions_path
    do_not_lowercase_text = args.do_not_lowercase_text
    lowercase_cngrams = args.lowercase_cngrams
    use_original_hours_values = args.use_original_hours_values
    just_extract_features = args.just_extract_features
    try_to_reuse_extracted_features = args.try_to_reuse_extracted_features
    do_not_binarize_features = args.do_not_binarize_features
    eval_combined_games = args.eval_combined_games

    # Initialize logging system
    logger = logging.getLogger('eval')
    logger.setLevel(logging.DEBUG)

    # Create file handler
    fh = logging.FileHandler(abspath(args.log_file_path))
    fh.setLevel(logging.DEBUG)

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

    # Redirect warnings to the logging system
    logging.captureWarnings(True)

    loginfo = logger.info
    logdebug = logger.debug
    logerror = logger.error
    logwarn = logger.warning

    # Get predictions/results output file path and models directory path and
    # make sure model exists
    if not just_extract_features:

        # Import methods and stuff that will be used if not only feature
        # extraction is being done
        from numpy import array, chararray
        from skll import Learner
        load_learner = Learner.from_file
        from skll.metrics import kappa, pearson
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
            logerror('Could not find model with prefix {} in models ' \
                     'directory ({}). Exiting.'.format(model,
                                                       models_dir))
            exit(1)
        paths = []
        if predictions_path:
            predictions_path = abspath(predictions_path)
            paths.append(predictions_path)
        if results_path:
            results_path = abspath(results_path)
            paths.append(results_path)
        if paths and not any(map(exists,
                                 paths)):
            logerror('Could not verify the existence of the destination ' \
                     'directories for the predictions and/or results ' \
                     'output files. Exiting.')
            exit(1)
        if predictions_path:
            import csv

    # Make sure command-line arguments make sense
    if just_extract_features \
       and (results_path
            or predictions_path
            or eval_combined_games):
        logerror('If the --just_extract_features flag is used, then any ' \
                 'other flags used for evaluation-related tasks cannot be ' \
                 'used (since the program skips evaluation if it is just ' \
                 'extracting the features and putting them in the MondoDB).' \
                 ' Exiting.')
        exit(1)

    if try_to_reuse_extracted_features \
       and (lowercase_cngrams
            or do_not_lowercase_text):
        logwarn('If trying to reuse previously extracted features, then the' \
                ' values picked for the --lowercase_cngrams and ' \
                '--do_not_lowercase_text should match the values used to ' \
                'build the models.')

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

    # Establish connection to MongoDB database
    connection_string = 'mongodb://localhost:{}'.format(args.mongodb_port)
    try:
        connection = MongoClient(connection_string)
    except ConnectionFailure as e:
        logerror('Unable to connect to to Mongo server at ' \
                 '{}'.format(connection_string))
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

    # Iterate over the game files, looking for test set reviews
    # Get list of games
    if game_files == "all":
        game_files = [f for f in listdir(data_dir) if f.endswith('.txt')]
        del game_files[game_files.index('sample.txt')]
    else:
        game_files = game_files.split(',')

    # If the --eval_combined_games flag was used but there's only one game
    # file to evaluate on, print a warning that the "combined" stats will only
    # be for one game...
    if (eval_combined_games
        and len(game_files) == 1):
        logwarn('The --eval_combined_games flag was used, but there was ' \
                'only one game that predictions were generated for.')

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

        _ids = []
        _ids_append = _ids.append
        hours_values = []
        hours_values_append = hours_values.append
        reviews = []
        reviews_append = reviews.append
        features_dicts = []
        features_dicts_append = features_dicts.append

        game = game_file[:-4]
        appid = APPID_DICT[game]

        # Get test reviews
        loginfo('Extracting features from the test data for {}' \
                '...'.format(game))
        game_docs = reviewdb_find({'game': game,
                                   'partition': 'test'},
                                  {'features': 0,
                                   'game': 0,
                                   'partition': 0})

        if game_docs.count() == 0:
            logerror('No matching documents were found in the MongoDB ' \
                     'collection in the test partition for game {}. ' \
                     'Exiting.'.format(game))
            exit(1)

        for game_doc in game_docs:

            _get = game_doc.get
            if bins:
                hours = _get('hours_bin')
            else:
                hours = _get('hours')
            review_text = _get('review')
            _id = _get('_id')
            _binarized = _get('binarized')

            _ids_append(repr(_id))
            hours_values_append(hours)
            reviews_append(review_text)

            found_features = None
            if try_to_reuse_extracted_features:
                features_doc = reviewdb_find_one({'_id': _id},
                                                 {'_id': 0,
                                                  'features': 1})
                features = features_doc.get('features')
                if features \
                   and _binarized == binarize:
                    features = json_decode(features)
                    found_features = True

            if not found_features:
                _Review = Review(review_text,
                                 hours,
                                 game,
                                 appid,
                                 spaCy_nlp,
                                 lower=lowercase_text)
                features = \
                    extract_features_from_review(
                        _Review,
                        lowercase_cngrams=lowercase_cngrams)

            # If binarize is True, make all values 1
            if (binarize
                and not (found_features
                         and _binarized)):
                features = dict(Counter(list(features)))

            # Update Mongo database game doc with new key "features", which
            # will be mapped to game_features, and a new key "binarized",
            # which will be set to True if features were extracted with the --
            # do_not_binarize_features flag or False otherwise
            if (not found_features
                and just_extract_features):
                tries = 0
                while tries < 5:
                    try:
                        reviewdb_update(
                            {'_id': _id},
                            {'$set': {'features': json_encode(features),
                                      'binarized': binarize}})
                        break
                    except AutoReconnect as e:
                        logwarn('Encountered ConnectionFailure error, ' \
                                'attempting to reconnect automatically...')
                        tries += 1
                        if tries >= 5:
                            logerror('Unable to update database even ' \
                                     'after 5 tries. Exiting.')
                            exit(1)
                        sleep(20)

            # Go to next game document if all that's being done is extracting
            # features
            if just_extract_features:
                continue

            # Append feature dict to end of list
            features_dicts_append(features)

        # Make all hours values natural numbers (rounding down)
        hours_values = [int(h) for h in hours_values]

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
            logerror('Lists of values not of expected length:\n\n{}\n\n' \
                     'Exiting.'.format(str([_ids,
                                            reviews,
                                            hours_values,
                                            predicted_labels])))
            exit()

        # Save predicted/expected values for final evaluation
        total_predicted_hours_labels_extend(predicted_labels)
        total_hours_values_extend(hours_values)

        # Open predictions/results file(s) (if applicable) for specific game
        # file
        if predictions_path:
            loginfo('Writing predictions file for {}...'.format(game))
            with open(join(predictions_path,
                           '{}.test_{}_predictions.csv'.format(game,
                                                               model)),
                      'w') as preds_file:
                preds_file_csv = csv.writer(preds_file,
                                            delimiter=',')
                preds_file_csv_writerow
                preds_file_csv_writerow(['id',
                                        'review',
                                        'hours_played',
                                        'prediction'])
                for _id, review, hours_value, pred in zip(_ids,
                                                          reviews,
                                                          hours_values,
                                                          predicted_labels):
                    preds_file_csv_writerow([_id,
                                             review,
                                             hours_value,
                                             pred])

        if results_path:
            loginfo('Writing results file for {}...'.format(game))
            with open(join(results_path,
                           '{}.test_{}_results.txt'.format(game,
                                                           model)),
                      'w') as results_file:
                results_file_write = results_file.write
                results_file_write('Results Summary\n\n')
                results_file_write('- Game: {}\n'.format(game))
                results_file_write('- Model: {}\n\n'.format(model))
                results_file_write('Evaluation Metrics\n\n')
                results_file_write('Kappa: {}\n'.format(
                                       kappa(hours_values,
                                             predicted_labels)))
                results_file_write('Kappa (allow off-by-one): {}\n'.format(
                    kappa(hours_values,
                          predicted_labels,
                          allow_off_by_one=True)))
                results_file_write('Pearson: {}\n\n'.format(
                                       pearson(hours_values,
                                               predicted_labels)))
                results_file_write('Confusion Matrix\n')
                results_file_write('(predicted along top, actual along ' \
                                   'side)\n\n')
                results_file_write('{}\n'.format(
                    make_confusion_matrix(hours_values,
                                          predicted_labels)['string']))

    # Do evaluation on all predicted/expected values across all games or exit
    if not eval_combined_games:
        loginfo('Complete.')
        exit(0)
    loginfo('Printing out evaluation metrics for the performance of the ' \
            'model across all games...')
    loginfo('Using predicted/expected values for the following games: ' \
            '{}'.format(', '.join(game_files)))
    loginfo('Kappa: {}'.format(kappa(total_hours_values,
                                     total_predicted_hours_labels)))
    loginfo('Kappa (allow off-by-one): {}'.format(
                kappa(total_hours_values,
                      total_predicted_hours_labels,
                      allow_off_by_one=True)))
    loginfo('Pearson: {}'.format(pearson(
                                     total_hours_values,
                                     total_predicted_hours_labels)))
    loginfo('Confusion Matrix (predicted along top, actual along side)\n\n' \
            '{}'.format(make_confusion_matrix(
                            total_hours_values,
                            total_predicted_hours_labels)['string']))
    loginfo('Complete.')