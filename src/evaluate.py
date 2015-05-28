'''
:author: Matt Mulholland
:date: May 13, 2015

Script used to make predictions for datasets (or multiple datasets combined) and generate evaluation metrics.
'''
import sys
import csv
import pymongo
import logging
import argparse
import numpy as np
from os import listdir
from data import APPID_DICT
from spacy.en import English
from collections import Counter
from skll import Learner, metrics
from pymongo.errors import AutoReconnect
from skll.data.featureset import FeatureSet
from json import dumps, JSONEncoder, JSONDecoder
from os.path import realpath, dirname, abspath, join, exists
from src.feature_extraction import Review, extract_features_from_review

# Since the skll Learner.predict method will probably print out a warning each
# time a prediction is made, let's try to stop the warnings in all but the
# first case
import warnings
warnings.filterwarnings('once')

project_dir = dirname(dirname(realpath(__file__)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage='python evaluate.py --game_files' \
        ' GAME_FILE1,GAME_FILE2,... --model MODEL_PREFIX[ --results_path ' \
        'PATH|--predictions_path PATH|--just_extract_features][ OPTIONS]',
        description='generate predictions for a data-set\'s test set ' \
                    'reviews and output evaluation metrics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game_files',
        help='comma-separated list of file-names or "all" for all of the ' \
             'files (the game files should reside in the "data" directory)',
        type=str,
        required=True)
    parser.add_argument('--model', '-m',
        help='model prefix (this will be the model that is used to generate' \
             ' predictions for all test reviews for the game files input ' \
             'via the --game_files option argument)',
        type=str,
        required=True)
    parser.add_argument('--results_path', '-r',
        help='destination directory for results output file',
        type=str,
        required=False)
    parser.add_argument('--predictions_path', '-p',
        help='destination directory for predictions file',
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
    parser.add_argument('--eval_combined_games',
        help='print evaluation metrics across all games',
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
        paths = []
        if args.predictions_path:
            predictions_path = abspath(args.predictions_path)
            paths.append(predictions_path)
        if args.results_path:
            results_path = abspath(args.results_path)
            paths.append(results_path)
        if paths and not any(map(exists, paths)):
            logger.error('Could not verify the existence of the destination' \
                         ' directories for the predictions and/or results ' \
                         'output files. Exiting.')
            sys.exit(1)

    # Make sure command-line arguments make sense
    if args.just_extract_features \
       and (args.results_path
            or args.predictions_path
            or args.eval_combined_games):
        logger.error('If the --just_extract_features flag is used, then any' \
                     ' other flags used for evaluation-related tasks cannot' \
                     ' be used (since the program skips evaluation if it is' \
                     ' just extracting the features and putting them in the' \
                     ' MondoDB). Exiting.')
        sys.exit(1)

    if args.try_to_reuse_extracted_features \
       and (args.lowercase_cngrams
            or args.do_not_lowercase_text):
        logger.warning('If trying to reuse previously extracted features, ' \
                       'then the values picked for the --lowercase_cngrams ' \
                       'and --do_not_lowercase_text should match the values' \
                       ' used to build the models.')

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
    json_encoder = JSONEncoder()
    json_decoder = JSONDecoder()

    # Iterate over the game files, looking for test set reviews
    # Get list of games
    game_files = []
    if args.game_files == "all":
        game_files = [f for f in listdir(data_dir) if f.endswith('.txt')]
        del game_files[game_files.index('sample.txt')]
    else:
        game_files = args.game_files.split(',')

    # If the --eval_combined_games flag was used but there's only one game
    # file to evaluate on, print a warning that the "combined" stats will only
    # be for one game...
    if args.eval_combined_games and len(game_files) == 1:
        logger.warning('The --eval_combined_games flag was used, but there ' \
                       'was only one game that predictions were generated ' \
                       'for.')

    if args.model:
        learner = Learner.from_file(join(models_dir,
                                         '{}.model'.format(args.model)))

    # Lists of original and predicted hours values
    total_hours_values = []
    total_predicted_hours_values = []

    # Iterate over game files, generating/fetching features, etc., and putting
    # them in lists
    for game_file in game_files:

        _ids = []
        hours_values = []
        reviews = []
        features_dicts = []

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

            _ids.append(repr(game_doc['_id']))
            hours_values.append(game_doc['hours_bin'] if bins else \
                                game_doc['hours'])
            reviews.append(game_doc['review'])

            found_features = None
            if args.try_to_reuse_extracted_features:
                features_doc = reviewdb.find_one({'_id': game_doc['_id']},
                                                 {'_id': 0,
                                                  'features': 1})
                features = features_doc.get('features')
                if features \
                   and game_doc.get('binarized') == binarize:
                    features = json_decoder.decode(features)
                    found_features = True

            if not found_features:
                _Review = Review(game_doc['review'],
                                 game_doc['hours_bin'] if bins else \
                                     game_doc['hours'],
                                 game,
                                 appid,
                                 spaCy_nlp,
                                 lower=lowercase_text)
                features = \
                    extract_features_from_review(_Review,
                        lowercase_cngrams=args.lowercase_cngrams)

            # If binarize is True, make all values 1
            if binarize and not (found_features
                                 and game_doc.get('binarized')):
                features = dict(Counter(list(features)))

            # Update Mongo database game doc with new key "features", which
            # will be mapped to game_features, and a new key "binarized",
            # which will be set to True if features were extracted with the --
            # do_not_binarize_features flag or False otherwise
            if not found_features and args.just_extract_features:
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
                        logger.warning('Encountered ConnectionFailure ' \
                                       'error, attempting to reconnect ' \
                                       'automatically...')
                        tries += 1
                        if tries >= 5:
                            logger.error('Unable to update database even ' \
                                         'after 5 tries. Exiting.')
                            sys.exit(1)
                        sleep(20)

            # Go to next game document if all that's being done is extracting
            # features
            if args.just_extract_features:
                continue

            # Append feature dict to end of list
            features_dicts.append(features)

        # Make all hours values natural numbers (rounding down)
        hours_values = [int(h) for h in hours_values]

        # Make list of FeatureSet instances
        fs = FeatureSet('{}.test'.format(game),
                        np.array(_ids,
                                 dtype=np.chararray),
                        feature_dicts)

        # Generate predictions
        predictions = [int(learner.predict(_fs)) for _fs in fs]

        # Make sure all the lists are equal
        if not any([map(lambda x, y: len(x) == len(y),
                        [_ids, reviews, hours_values, predictions])]):
            logger.error('Lists of values not of expected length:\n\n' \
                         '{}\n\nExiting.'.format(str([_ids,
                                                      reviews,
                                                      hours_values,
                                                      predictions])))
            sys.exit()

        # Save predicted/expected values for final evaluation
        total_predicted_hours_values.extend(predictions)
        total_hours_values.extend(hours_values)

        # Open predictions/results file(s) (if applicable) for specific game
        # file
        if args.predictions_path:
            logger.info('Writing predictions file for {}...'.format(game))
            preds_file = open(join(args.predictions_path,
                                   '{}.test_{}_predictions.csv'.format(
                                       game,
                                       args.model)),
                              'w')
            preds_file_csv = csv.writer(preds_file,
                                        delimiter=',')
            preds_file_csv.writerow(['id',
                                     'review',
                                     'hours_played',
                                     'prediction'])
            for _id, review, hours_value, pred in zip(_ids,
                                                      reviews,
                                                      hours_values,
                                                      predictions):
                preds_file_csv.writerow([_id,
                                         review,
                                         hours_value,
                                         pred])
            preds_file.close()

        if args.results_path:
            logger.info('Writing results file for {}...'.format(game))
            results_file = open(join(args.results_path,
                                     '{}.test_{}_results.md'.format(
                                         game,
                                         args.model)),
                                'w')
            results_file.write('#Results Summary\n')
            results_file.write('- Game: {}\n'.format(game))
            results_file.write('- Model: {}\n'.format(args.model))
            results_file.write('##Evaluation Metrics\n')
            results_file.write('Kappa: {}\n'.format(metrics.kappa(
                                                        hours_values,
                                                        predictions)))
            results_file.write('Kappa (allow off-by-one): {}\n'.format(
                metrics.kappa(hours_values,
                              predictions)))
            results_file.write('Pearson: {}\n'.format(metrics.pearson(
                                                          hours_values,
                                                          predictions)))
            results_file.write('##Confusion Matrix (predicted along top, ' \
                               'actual along side)\n')
            results_file.write('{}\n'.format(metrics.use_score_func(
                                                 'confusion_matrix',
                                                 hours_values,
                                                 predictions)))
            results_file.close()

    # Do evaluation on all predicted/expected values across all games or exit
    if not args.eval_combined_games:
        logger.info('Complete.')
        sys.exit(0)
    logger.info('Printing out evaluation metrics for the performance of the' \
                ' model across all games...')
    logger.info('Using predicted/expected values for the following games: ' \
                '{}'.format(', '.join(game_files)))
    logger.info('Kappa: {}'.format(metrics.kappa(
                                       total_hours_values,
                                       total_predicted_hours_values)))
    logger.info('Kappa (allow off-by-one): {}'.format(
                    metrics.kappa(total_hours_values,
                                  total_predicted_hours_values)))
    logger.info('Pearson: {}'.format(metrics.pearson(
                                         total_hours_values,
                                         total_predicted_hours_values)))
    logger.info('Confusion Matrix (predicted along top, actual along side)' \
                '\n\n{}'.format(metrics.use_score_func(
                                    'confusion_matrix',
                                    total_hours_values,
                                    total_predicted_hours_values)))
    logger.info('Complete.')