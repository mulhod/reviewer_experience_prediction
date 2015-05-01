#!/usr/env python3.4
import os
import sys
import time
import pymongo
import argparse
from re import sub
from util.read_data_files import get_reviews_for_game
from os.path import realpath, abspath, dirname, join, basename

# ARFF file template
ARFF_BASE = '''
% Generated on {}
% This ARFF file was generated with review data from the following game(s):
%     {}
% It is useful only for trying out machine learning algorithms on the
% bag-of-words representation of the reviews only.
@relation reviewer_experience
@attribute string_attribute string
@attribute numeric_attribute numeric

@data'''
TIMEF = '%A, %d. %B %Y %I:%M%p'

# Global variable used for MongoDB collection, if making training/test sets
reviewdb = None

def write_arff_file(reviews=None, file_path, file_names,
                    make_train_test=False):
    '''
    Write .arff file.

    :param reviews: list of dicts with hours/review keys-value mappings representing each data-point (defaults to None)
    :type reviews: list of dict
    :param file_path: path to output .arff file
    :type file_path: str
    :param file_names: list of extension-less game file-names
    :type file_names: list of str
    :param make_train_test: if True, use MongoDB collection to find reviews that are from the training and test partitions and make files for them instead of making one big file (defaults to False)
    :type make_train_test: boolean
    :returns: None
    '''

    global reviewdb

    # Replace underscores with spaces in the game names
    _file_names = [sub(r'_',
                       r' ',
                       f) for f in file_names]

    # Write ARFF file(s)
    if make_train_test:

        # Make an ARFF file for each partition
        for partition in ['training', 'test']:

            # Make empty list of lines to populate with ARFF-style lines,
            # one per review
            reviews_lines = []

            # Modify file-path by adding partition suffix
            suffix = 'train' if partition.startswith('train') else 'test'
            replacement = '.{}.arff'.format(suffix)
            _file_path = sub(r'\.arff$',
                             replacement,
                             file_path)

            # Get reviews for the given partition from all of the games
            game_docs_cursor = \
                reviewdb.find({'partition': partition,
                               'game': {'$in': file_names}})
            if game_docs_cursor.count() == 0:
                sys.exit('ERROR: No matching documents were found in the ' \
                         'MongoDB collection for the {} partition and the' \
                         ' following games:\n\n{}\nExiting.' \
                         '\n'.format(partition,
                                     file_names))
            
            game_docs = list(game_docs_cursor)
            for game_doc in game_docs:
                # Remove single/double quotes from the reviews first...
                review = sub(r'\'|"',
                             r'',
                             game_doc['review'].lower())
                # Get rid of backslashes since they only make things
                # confusing
                review = sub(r'\\',
                             r'',
                             review)
                reviews_lines.append('"{}",{}'.format(review,
                                                      game_doc['hours']))
            with open(_file_path,
                      'w') as out:
                out.write('{}\n{}'.format(ARFF_BASE.format(
                                              time.strftime(TIMEF),
                                              ' ,'.join(file_names)),
                              '\n'.join(reviews_lines)))
    else:

        if not reviews:
            sys.exit('ERROR: Empty list of reviews passed in to the ' \
                     'write_arff_file method. Exiting.\n')

        # Make empty list of lines to populate with ARFF-style lines,
        # one per review
        reviews_lines = []

        for review_dict in reviews:
            # Remove single/double quotes from the reviews first...
            review = sub(r'\'|"',
                         r'',
                         review_dict['review'].lower())
            # Get rid of backslashes since they only make things confusing
            review = sub(r'\\',
                         r'',
                         review)
            reviews_lines.append('"{}",{}'.format(review,
                                                  review_dict['hours']))
        with open(file_path,
                      'w') as out:
            out.write('{}\n{}'.format(ARFF_BASE.format(time.strftime(TIMEF),
                                                       ' ,'.join(file_names)),
                                      '\n'.join(reviews_lines)))


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
    if args.make_train_test_sets:
        connection = pymongo.MongoClient('mongodb://localhost:27017')
        db = connection['reviews_project']
        reviewdb = db['reviews']

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

                review_dicts_list.extend(get_reviews_for_game(
                                             join(data_dir,
                                                  game_file)))

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
                            make_train_test=True)
        else:
            sys.stderr.write('Generating {}...\n'.format(arff_file))
            write_arff_file(reviews=review_dicts_list,
                            arff_file,
                            file_names)
        sys.stderr.write('Generated ARFF file(s) for {}...' \
                         '\n'.format(arff_file))
    else:
        for game_file in game_files:

            sys.stderr.write('Getting review data from {}...' \
                             '\n'.format(game_file))

            if args.make_train_test_sets:
                review_dicts_list = get_reviews_for_game(join(data_dir,
                                                              game_file))

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
                                make_train_test=True)
            else:
                sys.stderr.write('Generating {}...\n'.format(arff_file))
                write_arff_file(reviews=review_dicts_list,
                                arff_file,
                                file_names)
            sys.stderr.write('Generated ARFF file(s) for {}...' \
                             '\n'.format(arff_file))