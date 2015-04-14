#!/usr/env python2.7
import sys
import os
import re
import argparse
import datetime
from os.path import realpath, abspath, dirname, splitext
from util.read_data_files import get_reviews_for_game

ARFF_BASE = \
'''
% Generated on {}
% This ARFF file for {} is for use with trying out machine learning algorithms
% on the bag-of-words representation of the reviews only.
@relation reviewer_experience
@attribute experience numeric
@attribute review string
@data
'''
TIMEF = '%A, %d. %B %Y %I:%M%p'

def write_arff_file(review_dicts, file_path):
    '''
    Write .arff file.

    :param review_dicts: list of dicts with hours/review keys representing each data-point
    :type review_dicts: list of dict
    :param file_path: path to output .arff file
    :type file_path: str
    :returns: None
    '''

    game_name = splitext(file_path)[0]
    with open(file_path,
              'w') as out:
        reviews_lines = []
        for review_dict in review_dicts:
            # Remove single quotes from the reviews first...
            review = re.sub(r'\'',
                            r'',
                            review_dict['review'].lower())
            reviews_lines.append('"{},{}"'.format(review,
                                                  review_dict['hours']))
        arff = ARFF_BASE.format(game_name,
                                datetime.utcnow().strftime(TIMEF))
        out.write('{}\n{}'.format(arff,
                                '\n'.join(reviews_lines)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage='python make_arff.py',
        description='Build .arff files for a specific game file, all game ' \
                    'files combined, or for each game file separately.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game_files',
        help='comma-separated list of game-files or "all" for all of the ' \
             'files (the game files should reside in the "data" directory)',
        type=str,
        required=True)
    parser.add_argument('--mode',
        help='make .arff file for each game file separately ("separate") or' \
             ' for all game files combined ("combined")',
        choices=["separate", "combined"],
        default="combined")
    args = parser.parse_args()

    # Get paths to the data and arff_files directories
    project_dir = dirname(dirname(abspath(realpath(__file__))))
    data_dir = join(project_dir,
                    'data')
    arff_files_dir = join(project_dir,
                          'arff_files')
    sys.stderr.write('data directory: {}\n'.format(data_dir))
    sys.stderr.write('arff_files directory: {}\n'.format(arff_files_dir))

    mode = args.mode
    game_files = []
    if args.game_files == "all":
        game_files = os.listdir(data_dir)
    else:
        game_files = args.game_files.split(',')
    if len(game_files) == 1:
        mode = "separate"
        # Print out warning message if --mode was set to "combined" and there
        # was only one file n the list of game files since only a single .arff
        # file will be created
        if args.mode == 'combined':
            sys.stderr.write('WARNING: The --mode flag was used with the ' \
                             'value "combined" even though only one game ' \
                             'file was passed in via the --game_files ' \
                             'flag. Only one file will be written and it ' \
                             'will be named after the game.\n')

    # Make a list of dicts corresponding to each review and write .arff files
    sys.stderr.write('Reading in data from reviews files...\n')
    if mode == "combined":
        review_dicts_list = []
        for game_file in game_files:
            sys.stderr.write('Reading data from {}...\n'.format(game_file))
            review_dicts_list.extend(get_reviews_for_game(join(data_dir,
                                                               game_file)))
        if args.game_files == 'all':
            arff_file = join(arff_files_dir,
                             'all.arff')
        else:
            arff_file = join(arff_files_dir,
                             '{}.arff'.format('_'.join(game_files)))
        sys.stderr.write('Writing to {}...\n'.format(arff_file))
        write_arff_file(review_dicts_list,
                        arff_file)
        sys.stderr.write('Finished writing {}...\n'.format(arff_file))
    else:
        for game_file in game_files:
            sys.stderr.write('Reading data from {}...\n'.format(game_file))
            review_dicts_list = get_reviews_for_game(join(data_dir,
                                                          game_file)))
            arff_file = join(arff_files_dir,
                             '{}.arff'.format(game_file))
            sys.stderr.write('Writing to {}...\n'.format(arff_file))
            write_arff_file(review_dicts_list,
                            arff_file)
            sys.stderr.write('Finished writing {}...\n'.format(arff_file))