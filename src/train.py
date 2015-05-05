#!/usr/env python3.4
'''
@author: Matt Mulholland, Janette Martinez, Emily Olshefski
@date: 3/18/15

Script used to train models on datasets (or multiple datasets combined).
'''
import sys
import pymongo
import argparse
from os import listdir
from data import APPID_DICT
from spacy.en import English
from collections import Counter
from skll import run_configuration
from json import dumps, JSONEncoder, JSONDecoder
from os.path import join, dirname, realpath, abspath
from src.feature_extraction import (Review, extract_features_from_review,
                                    write_config_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage='python train.py',
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
             'features (defaults to False)',
        action='store_true',
        default=False)
    parser.add_argument('--lowercase_cngrams',
        help='lower-case the review text before extracting character n-gram' \
             ' features (defaults to False)',
        action='store_true',
        default=False)
    parser.add_argument('--just_extract_features',
        help='exract features from all of the reviews, generate .jsonlines ' \
             'files, etc., but quit before training any models (defaults to' \
             'False)',
        action='store_true',
        default=False)
    parser.add_argument('--try_to_reuse_extracted_features',
        help='try to make use of previously-extracted features that reside ' \
             'in the MongoDB database (defaults to False)',
        action='store_true',
        default=False)
    parser.add_argument('--do_not_binarize_features',
        help='do not make all non-zero feature frequencies equal to 1 ' \
             '(defaults to False)',
        action='store_true',
        default=False)
    parser.add_argument('--mongodb_port', '-dbport',
        help='port that the MongoDB server is running (defaults to 27017',
        type=int,
        default=27017)
    args = parser.parse_args()

    # Get paths to the project, data, working, and models directories
    project_dir = dirname(dirname(abspath(realpath(__file__))))
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
    sys.stderr.write('project directory: {}\ndata directory: {}\nworking ' \
                     'directory: {}\nmodels directory: {}\nconfiguration ' \
                     'directory: {}\nlogs directory: {}\n'.format(project_dir,
                                                                  data_dir,
                                                                  working_dir,
                                                                  models_dir,
                                                                  cfg_dir,
                                                                  logs_dir))

    binarize = not args.do_not_binarize_features
    sys.stderr.write('Binarize features? {}\n'.format(binarize))
    lowercase_text = not args.do_not_lowercase_text
    sys.stderr.write('Lower-case text as part of the normalization step? ' \
                     '{}\n'.format(lowercase_text))

    # Make sure that, if --combine is being used, there is also a file prefix
    # being passed in via --combined_model_prefix for the combined model
    if args.combine and not args.combined_model_prefix:
        sys.exit('ERROR: When using the --combine flag, you must also ' \
                 'specify a model prefix, which can be passed in via the ' \
                 '--combined_model_prefix option argument. Exiting.\n')

    # Establish connection to MongoDB database
    connection_string = 'mongodb://localhost:{}'.format(args.mongodb_port)
    try:
        connection = pymongo.MongoClient(connection_string)
    except pymongo.errors.ConnectionFailure as e:
        sys.exit('ERROR: Unable to connecto to Mongo server at ' \
                 '{}'.format(connection_string))
    db = connection['reviews_project']
    reviewdb = db['reviews']

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
    else:
        game_files = args.game_files.split(',')

    # Train a combined model with all of the games or train models for each
    # individual game dataset
    if args.combine:

        sys.stderr.write('Extracting features to train a combined model ' \
                         'with training data from the following games: {}' \
                         '\n'.format(', '.join(game_files)))

        # Initialize empty list for holding all of the feature dictionaries
        # from each review in each game and then extract features from each
        # game's training data
        feature_dicts = []
        for game_file in game_files:

            # Get the training reviews for this game from the Mongo
            # database
            game = game_file[:-4]
            sys.stderr.write('Extracting features from the training data ' \
                             'for {}...\n'.format(game))
            appid = APPID_DICT[game]
            game_docs = list(reviewdb.find({'game': game,
                                            'partition': 'training'}))
            if len(game_docs) == 0:
                sys.exit('ERROR: No matching documents were found in the ' \
                         'MongoDB collection in the training partition ' \
                         'for game {}. Exiting.\n'.format(game))

            # Iterate over all training documents for the given game
            for game_doc in game_docs:

                # Get the game_doc ID, the hours played value, and the
                # original review text from the game_doc
                _id = game_doc['_id']
                hours = game_doc['hours']
                review_text = game_doc['review']

                # Instantiate a Review object
                _Review = Review(review_text,
                                 hours,
                                 game,
                                 appid,
                                 spaCy_nlp,
                                 lower=lowercase_text)

                # Extract features from the review text
                found_features = False
                if args.try_to_reuse_extracted_features:
                    features = game_doc.get('features')
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
                # extracted with the --do_not_binarize_features flag or False
                # otherwise
                if not found_features:
                    reviewdb.update(
                        {'_id': _id},
                        {'$set': {'features': json_encoder.encode(features),
                                  'binarized': binarize}})

                # Append a feature dictionary for the review to feature_dicts
                feature_dicts.append({'id': str(_id),
                                      'y': hours,
                                      'x': features})

        # Write .jsonlines file
        jsonlines_filename = '{}.jsonlines'.format(args.combined_model_prefix)
        jsonlines_filepath = join(working_dir,
                                  jsonlines_filename)
        sys.stderr.write('Writing {} to working directory...'.format(
                                                          jsonlines_filename))
        with open(jsonlines_filepath, 'w') as jsonlines_file:
            [jsonlines_file.write('{}\n'.format(
                                   dumps(fd).encode('utf-8').decode('utf-8')))
             for fd in feature_dicts]

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
                                        dumps([[args.combined_model_prefix]]),
                                   "suffix": '.jsonlines',
                                   "learners": dumps([learner_name])
                                   },
                         "Tuning": {"feature_scaling": "none",
                                    "grid_search": "True",
                                    "min_feature_count": "1",
                                    "objective": grid_objective,
                                    "param_grids": dumps([param_grid_list]),
                                    },
                         "Output": {"probability": "False",
                                    "log": join(logs_dir,
                                                '{}.log'.format(
                                                  args.combined_model_prefix))
                                    }
                         }

        # Set up the job for training the model
        sys.stderr.write('Generating configuration file...')
        cfg_filename = '{}.cfg'.format(args.combined_model_prefix)
        cfg_filepath = join(cfg_dir,
                            cfg_filename)
        cfg_dict_base["General"]["task"] = "train"
        cfg_dict_base["General"]["experiment_name"] = \
            args.combined_model_prefix
        cfg_dict_base["Output"]["models"] = models_dir
        write_config_file(cfg_dict_base,
                          cfg_filepath)

        if not args.just_extract_features:
            # Run the SKLL configuration, producing a model file
            sys.stderr.write('Training combined model...\n')
            run_configuration(cfg_filepath)
    else:
        for game_file in game_files:

            game = game_file[:-4]

            sys.stderr.write('Extracting features to train a model with ' \
                             'training data from {}...\n'.format(game))

            # Initialize empty list for holding all of the feature
            # dictionaries from each review and then extract features from all
            # reviews
            feature_dicts = []

            # Get the training reviews for this game from the Mongo
            # database
            sys.stderr.write('Extracting features from the training data ' \
                             'for {}...\n'.format(game))
            appid = APPID_DICT[game]
            game_docs = list(reviewdb.find({'game': game,
                                            'partition': 'training'}))
            if len(game_docs) == 0:
                sys.exit('ERROR: No matching documents were found in the ' \
                         'MongoDB collection in the training partition ' \
                         'for game {}. Exiting.\n'.format(game))

            # Iterate over all training documents for the given game
            for game_doc in game_docs:

                # Get the game_doc ID, the hours played value, and the
                # original review text from the game_doc
                _id = game_doc['_id']
                hours = game_doc['hours']
                review_text = game_doc['review']

                # Instantiate a Review object
                _Review = Review(review_text,
                                 hours,
                                 game,
                                 appid,
                                 spaCy_nlp,
                                 lower=lowercase_text)

                # Extract features from the review text
                found_features = False
                if args.try_to_reuse_extracted_features:
                    features = game_doc.get('features')
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
                # extracted with the --do_not_binarize_features flag or False
                # otherwise
                if not found_features:
                    reviewdb.update(
                        {'_id': _id},
                        {'$set': {'features': json_encoder.encode(features),
                                  'binarized': binarize}})

                # Append a feature dictionary for the review to feature_dicts
                feature_dicts.append({'id': str(_id),
                                      'y': hours,
                                      'x': features})

            # Write .jsonlines file
            jsonlines_filename = '{}.jsonlines'.format(game)
            jsonlines_filepath = join(working_dir,
                                      jsonlines_filename)
            sys.stderr.write('Writing {} to working directory...'.format(
                                                          jsonlines_filename))
            with open(jsonlines_filepath, 'w') as jsonlines_file:
                [jsonlines_file.write('{}\n'.format(
                                   dumps(fd).encode('utf-8').decode('utf-8')))
                 for fd in feature_dicts]

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

            # Set up the job for training the model
            sys.stderr.write('Generating configuration file...')
            cfg_filename = '{}.train.cfg'.format(game)
            cfg_filepath = join(cfg_dir,
                                cfg_filename)
            cfg_dict_base["General"]["task"] = "train"
            cfg_dict_base["General"]["experiment_name"] = \
                '{}.train'.format(game)
            cfg_dict_base["Output"]["models"] = models_dir
            write_config_file(cfg_dict_base,
                              cfg_filepath)

            if not args.just_extract_features:
                # Run the SKLL configuration, producing a model file
                sys.stderr.write('Training model for {}...\n'.format(game))
                run_configuration(cfg_filepath)

    sys.stderr.write('Complete.\n')
