'''
:author: Matt Mulholland
:date: April 15, 2015

Script used to generate ARFF files usable in Weka for the video game review data-sets.
'''
import os
import sys
import logging
from re import sub
from pymongo import MongoClient
from os.path import realpath, abspath, dirname, join, basename
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from util.datasets import (get_and_describe_dataset, get_bin_ranges,
                           write_arff_file)

project_dir = dirname(dirname(realpath(__file__)))

if __name__ == '__main__':

    parser = ArgumentParser(
        usage='python make_arff.py --game_files GAME_FILE1,GAME_FILE2[ ' \
              'OPTIONS]',
        description='Build .arff files for a specific game file, all game ' \
                    'files combined, or for each game file separately.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser_add_argument = parser.add_argument
    parser_add_argument('--game_files',
        help='comma-separated list of file-names or "all" for all of the ' \
             'files (the game files should reside in the "data" directory)',
        type=str,
        required=True)
    parser_add_argument('--mode',
        help='make .arff file for each game file separately ("separate") or' \
             ' for all game files combined ("combined")',
        choices=["separate", "combined"],
        default="combined")
    parser_add_argument('--combined_file_prefix',
        help='if the "combined" value was passed in via the --mode flag ' \
             '(which happens by default unless specified otherwise), an ' \
             'output file prefix must be passed in via this option flag',
        type=str,
        required=False)
    parser_add_argument('--use_original_hours_values',
        help='use the unmodified hours played values; otherwise, use the ' \
             'collapsed values',
        action='store_true',
        default=False)
    parser_add_argument('--make_train_test_sets',
        help='search the MongoDB collection for training/test set reviews ' \
             'and make ARFF files using them only (the file suffix ".train"' \
             '/".test" will be appended onto the end of the output file ' \
             'name to distinguish the different files); note that, by ' \
             'default, collapsed hours played values will be used (if this ' \
             'is not desired, use the --use_original_hours_values flag)',
        action='store_true',
        default=False)
    parser_add_argument('--nbins',
        help='specify the number of bins in which to collapse hours played ' \
             'values; to be used if the --make_train_test_sets flag is not ' \
             'being used, in which case pre-computed hours played values ' \
             'will not be read in from the database, but you still want the' \
             ' values to be in the form of bins (i.e., 1 for 0-100, 2 for ' \
             '101-200, etc., depending on the minimum and maximum values ' \
             'and the number of bins specified)',
        type=int,
        required=False)
    parser_add_argument('--mongodb_port', '-dbport',
        help='port that the MongoDB server is running',
        type=int,
        default=27017)
    parser_add_argument('--log_file_path', '-log',
        help='path for log file',
        type=str,
        default=join(project_dir,
                     'logs',
                     'replog_make_arff.txt'))
    args = parser.parse_args()

    # Make local copies of arguments
    game_files = args.game_files
    mode = args.mode
    combined_file_prefix = args.combined_file_prefix
    make_train_test_sets = args.make_train_test_sets
    nbins = args.nbins
    mongodb_port = args.mongodb_port

    # Initialize logging system
    logger = logging.getLogger('make_arff')
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

    bins = not args.use_original_hours_values

    # Make sure --bins option flag makes sense
    if nbins:
        if make_train_test_sets:
            logger.info('If the --make_train_test_sets flag is used, a ' \
                        'number of bins in which to collapse the hours ' \
                        'played values cannot be specified (since the ' \
                        'values in the database were pre-computed). Exiting.')
            sys.exit(1)
        elif not bins:
            logger.info('Conflict between the --use_original_hours_values ' \
                        'and --nbins flags. Both cannot be used at the same' \
                        ' time.')
            sys.exit(1)
    elif (bins and not make_train_test_sets):
        logger.info('If both the --use_original_hours_values and ' \
                    '--make_train_test_sets flags are not used, then the ' \
                    'number of bins in which to collapse the hours played ' \
                    'values must be specified via the --nbins option ' \
                    'argument. Exiting.')
        sys.exit(1)

    # Get path to the data directory
    data_dir = join(project_dir,
                    'data')
    if bins:
        arff_files_dir = join(project_dir,
                              'arff_files_collapsed_values')
    else:
        arff_files_dir = join(project_dir,
                              'arff_files_original_values')
    logger.info('data directory: {}'.format(data_dir))
    logger.info('arff files directory: {}'.format(arff_files_dir))

    # Make sure there is a combined output file prefix if "combine" is the
    # value passed in via --mode
    if mode == 'combined' and not combined_file_prefix:
        logger.error('A combined output file prefix must be specified in ' \
                    'cases where the "combined" value was passed in via the' \
                    ' --mode option flag (or --mode was not specified, in ' \
                    'which case "combined" is the default value). Exiting.')
        sys.exit(1)

    # See if the --make_train_test_sets flag was used, in which case we have
    # to make a connection to the MongoDB collection
    # And, if it wasn't used, then print out warning if the --mongodb_port
    # flag was used (since it will be ignored) unless the value is equal to
    # the default value (since it probably wasn't specified in that case)
    if make_train_test_sets:
        connection = pymongo.MongoClient('mongodb://localhost:' \
                                         '{}'.format(mongodb_port))
        db = connection['reviews_project']
        reviewdb = db['reviews']
    elif mongodb_port and not mongodb_port == 27017:
        logger.warning('Ignoring argument passed in via the ' \
                       '--mongodb_port/-dbport option flag since the ' \
                       '--make_train_test_sets flag was not also used, ' \
                       'which means that the MongoDB database is not going ' \
                       'to be used.')
        sys.exit(1)

    if game_files == "all":
        game_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        del game_files[game_files.index('sample.txt')]
    else:
        game_files = game_files.split(',')
    if len(game_files) == 1:
        # Print out warning message if --mode was set to "combined" and there
        # was only one file n the list of game files since only a single ARFF
        # file will be created
        if mode == 'combined':
            logger.warning('The --mode flag was used with the value ' \
                           '"combined" (or was unspecified) even though ' \
                           'only one game file was passed in via the ' \
                           '--game_files flag. Only one file will be ' \
                           'written and it will be named after the game.')
            sys.exit(1)
        mode = "separate"

    # Make a list of dicts corresponding to each review and write .arff files
    logger.info('Reading in data from reviews files...')
    if mode == "combined":

        review_dicts_list = []

        if not make_train_test_sets:

            # Min/max values of hours played (i.e., game experience)
            if bins:
                minh = 0.0
                maxh = 0.0

            for game_file in game_files:

                logger.info('Getting review data from {}' \
                            '...'.format(game_file))

                dataset = get_and_describe_dataset(join(data_dir,
                                                        game_file),
                                                   report=False)
                review_dicts_list.extend(dataset['reviews'])

                # If the hours played values are to be divided into bins,
                # update the min/max values
                if bins:
                    if dataset['minh'] < minh:
                        minh = dataset['minh']
                    if dataset['max'] > maxh:
                        maxh = dataset['maxh']

            # If the hours played values are to be divided into bins, get the
            # range that each bin maps to
            if bins:
                bin_ranges = get_bin_ranges(minh,
                                            maxh,
                                            nbins)
            else:
                bin_ranges = False

        file_names = [game[:-4] for game in game_files]
        arff_file = join(arff_files_dir,
                         '{}.arff'.format(combined_file_prefix))

        if make_train_test_sets:
            logger.info('Generating ARFF files for the combined training ' \
                        'sets and the combined test sets, respectively, of ' \
                        'the following games:\n\n' \
                        '{}'.format(', '.join([sub(r'_',
                                                   r' ',
                                                   fname) for fname
                                               in file_names])))
            write_arff_file(arff_file,
                            file_names,
                            reviewdb=reviewdb,
                            make_train_test=True,
                            bins=True)
        else:
            logger.info('Generating {}...'.format(arff_file))
            write_arff_file(arff_file,
                            file_names,
                            reviews=review_dicts_list,
                            bins=bin_ranges)
    else:
        for game_file in game_files:

            logger.info('Getting review data from {}...'.format(game_file))

            if not make_train_test_sets:
                review_dicts_list = []
                dataset = get_and_describe_dataset(join(data_dir,
                                                        game_file),
                                                   report=False)
                review_dicts_list.extend(dataset['reviews'])

                if bins:

                    # Get min/max hours played values from results of
                    # get_and_describe_dataset() call
                    minh = dataset['minh']
                    maxh = dataset['maxh']

                    # Get the range that each bin maps to
                    bin_ranges = get_bin_ranges(minh,
                                                maxh,
                                                nbins)
                else:
                    bin_ranges = False

            game = game_file[:-4]
            arff_file = join(arff_files_dir,
                             '{}.arff'.format(game))

            if make_train_test_sets:
                logger.info('Generating ARFF file for the training and test' \
                            ' sets for {}...'.format(game))
                write_arff_file(arff_file,
                                [game],
                                reviewdb=reviewdb,
                                make_train_test=True,
                                bins=bins)
            else:
                logger.info('Generating {}...'.format(arff_file))
                write_arff_file(arff_file,
                                [game],
                                reviews=review_dicts_list,
                                bins=bin_ranges)
    logger.info('Complete.')