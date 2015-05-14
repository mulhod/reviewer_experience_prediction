import sys
import logging
logger = logging.getLogger()
import argparse
from data import APPID_DICT
from os.path import dirname, abspath, realpath, join
from util.datasets import parse_appids, get_review_data_for_game

project_dir = dirname(dirname(realpath(__file__)))
data_dir = join(project_dir,
                'data')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage='./python get_review_data.py[ ' \
        '--appids APPID1,APPID2,...]',
        description='Make review data files for each game in the APPID_DICT' \
                    ', which is specified in the __init__.py module in the ' \
                    'the "data" directory. A specific list of game IDs ' \
                    '(same as "appid") can also be specified instead, but ' \
                    'they must map to games in APPID_DICT.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--appids',
        help='comma-separated list of game IDs for which to generate review' \
             ' data files (all IDs should map to games in APPID_DICT)',
        type=str,
        required=False)
    parser.add_argument('--log_file_path', '-log',
        help='path for log file',
        type=str,
        default=join(project_dir,
                     'logs',
                     'replog_get_review_data.txt'))
    args = parser.parse_args()

    # Initialize logging system
    logger = logging.getLogger('make_train_test_sets')
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

    # Make list of games for which to generate review data files
    if args.appids:
        appids = parse_appids(args.appids)
        games = []
        for appid in appids:
            for game in APPID_DICT:
                if APPID_DICT[game] == appid:
                    games.append(game)
    else:
        games = list(APPID_DICT)
        del games['sample.txt']

    # Generate review data files
    logger.info('Scraping review data from the Steam website...')
    for game in games:
        out_path = join(data_dir,
                        '{}.txt'.format(game))
        logger.info('Writing to {}...'.format(out_path))
        with open(out_path,
                  'w') as of:
            for review_set in get_review_data_for_game(APPID_DICT[game],
                                                       time_out=60.0):
                for review in review_set:
                    of.write('game-hours: {0[1]}\n' \
                             'review: {0[0]}\n'.format(review))
                of.flush()