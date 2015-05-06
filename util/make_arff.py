#!/usr/env python3.4
import os
import sys
import pymongo
import argparse
from re import sub
from src.feature_extraction import write_arff_file
from util.datasets import get_and_describe_dataset
from os.path import realpath, abspath, dirname, join, basename


if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage='python make_arff.py',
        description='Build .arff files for a specific game file, all game ' \
                    'files combined, or for each game file separately.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game_files',
        help='comma-separated list of file-names or "all" for all of the ' \
             'files (the game files should reside in the "data" directory)',
        type=str,
        required=True)
    parser.add_argument('--mode',
        help='make .arff file for each game file separately ("separate") or' \
             ' for all game files combined ("combined")',
        choices=["separate", "combined"],
        default="combined")
    parser.add_argument('--combined_file_prefix',
        help='if the "combined" value was passed in via the --mode flag ' \
             '(which happens by default unless specified otherwise), an ' \
             'output file prefix must be passed in via this option flag',
        type=str,
        required=False)
    parser.add_argument('--make_train_test_sets',
        help='search the MongoDB collection for training/test set reviews ' \
             'and make ARFF files using them only (the file suffix ".train"' \
             '/".test" will be appended onto the end of the output file ' \
             'name to distinguish the different files)',
        action='store_true',
        default=False)
    parser.add_argument('--mongodb_port', '-dbport',
        help='port that the MongoDB server is running (defaults to 27017',
        type=int,
        default=27017)
    args = parser.parse_args()

    # Get paths to the data and arff_files directories
    project_dir = dirname(dirname(abspath(realpath(__file__))))
    data_dir = join(project_dir,
                    'data')
    arff_files_dir = join(project_dir,
                          'arff_files')
    sys.stderr.write('data directory: {}\n'.format(data_dir))
    sys.stderr.write('arff_files directory: {}\n'.format(arff_files_dir))

    # Make sure there is a combined output file prefix if "combine" is the
    # value passed in via --mode
    if args.mode == 'combined' and not args.combined_file_prefix:
        sys.exit('ERROR: A combined output file prefix must be specified ' \
                 'in cases where the "combined" value was passed in via ' \
                 'the --mode option flag (or --mode was not specified, in' \
                 ' which case "combined" is the default value). Exiting.\n')

    # See if the --make_train_test_sets flag was used, in which case we have
    # to make a connection to the MongoDB collection
    # And, if it wasn't used, then print out warning if the --mongodb_port
    # flag was used (since it will be ignored)
    if args.make_train_test_sets:
        connection = pymongo.MongoClient('mongodb://localhost:' \
                                         '{}'.format(args.mongodb_port))
        db = connection['reviews_project']
        reviewdb = db['reviews']
    elif args.mongodb_port:
        sys.stderr.write('WARNING: Ignoring argument passed in via the ' \
                         '--mongodb_port option flag since the ' \
                         '--make_train_test_sets flag was not also used, ' \
                         'which means that the MongoDB database is not ' \
                         'going to be used for this task.\n')

    mode = args.mode
    game_files = []
    if args.game_files == "all":
        game_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    else:
        game_files = args.game_files.split(',')
    if len(game_files) == 1:
        mode = "separate"
        # Print out warning message if --mode was set to "combined" and there
        # was only one file n the list of game files since only a single ARFF
        # file will be created
        if args.mode == 'combined':
            sys.stderr.write('WARNING: The --mode flag was used with the ' \
                             'value "combined" (or was unspecified) even ' \
                             'though only one game file was passed in via ' \
                             'the --game_files flag. Only one file will be ' \
                             'written and it will be named after the game.\n')

    # Make a list of dicts corresponding to each review and write .arff files
    sys.stderr.write('Reading in data from reviews files...\n')
    if mode == "combined":

        review_dicts_list = []

        if not args.make_train_test_sets:
            for game_file in game_files:

                sys.stderr.write('Getting review data from {}...' \
                                 '\n'.format(game_file))

                dataset = get_and_describe_dataset(join(data_dir,
                                                        game_file),
                                                   report=False)
                review_dicts_list.extend(dataset['reviews'])

        file_names = [game[:-4] for game in game_files]
        arff_file = join(arff_files_dir,
                         '{}.arff'.format(args.combined_file_prefix))

        if args.make_train_test_sets:
            sys.stderr.write('Generating ARFF files for the combined ' \
                             'training sets and the combined test sets, ' \
                             'respectively, of the following games:\n\n' \
                             '{}\n'.format(', '.join([sub(r'_',
                                                          r' ',
                                                          fname) for fname
                                                      in file_names])))
            write_arff_file(arff_file,
                            file_names,
                            reviewdb=reviewdb,
                            make_train_test=True)
        else:
            sys.stderr.write('Generating {}...\n'.format(arff_file))
            write_arff_file(arff_file,
                            file_names,
                            reviews=review_dicts_list)
        sys.stderr.write('Generated ARFF file(s) for {}...' \
                         '\n'.format(arff_file))
    else:
        for game_file in game_files:

            sys.stderr.write('Getting review data from {}...' \
                             '\n'.format(game_file))

            if not args.make_train_test_sets:
                dataset = get_and_describe_dataset(join(data_dir,
                                                        game_file),
                                                   report=False)
                review_dicts_list.extend(dataset['reviews'])

            arff_file = join(arff_files_dir,
                             '{}.arff'.format(game_file[:-4]))

            if args.make_train_test_sets:
                sys.stderr.write('Generating ARFF files for the training ' \
                                 'and test sets for each of the following' \
                                 ' games:\n\n{}\n'.format(
                                     ', '.join([sub(r'_',
                                                    r' ',
                                                    fname) for fname in
                                                file_names])))
                write_arff_file(arff_file,
                                file_names,
                                reviewdb=reviewdb,
                                make_train_test=True)
            else:
                sys.stderr.write('Generating {}...\n'.format(arff_file))
                write_arff_file(arff_file,
                                file_names,
                                reviews=review_dicts_list)
            sys.stderr.write('Generated ARFF file(s) for {}...' \
                             '\n'.format(arff_file))