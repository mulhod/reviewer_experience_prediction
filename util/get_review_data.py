'''
:author: Matt Mulholland
:date: April 1, 2015

Script used to run the web scraping tool in order to build the video game review corpus.
'''
import logging
from json import dumps
logger = logging.getLogger()
from data import APPID_DICT
from os.path import (dirname,
                     abspath,
                     realpath,
                     join)
from util.datasets import (parse_appids,
                           get_review_data_for_game)
from argparse import (ArgumentParser,
                      ArgumentDefaultsHelpFormatter)

project_dir = dirname(dirname(realpath(__file__)))
data_dir = join(project_dir,
                'data')

if __name__ == '__main__':

    parser = ArgumentParser(usage='./python get_review_data.py[ '
        '--appids APPID1,APPID2,...]',
        description='Make review data files for each game in the APPID_DICT, '
                    'which is specified in the __init__.py module in the '
                    '"data" directory. A specific list of games or game IDs '
                    '(same as "appid") can also be specified instead, but '
                    'they must map to games in APPID_DICT.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser_add_argument = parser.add_argument
    parser_add_argument('--games',
        help='comma-separated list of game names (underscores replacing all '
             'spaces) for which to generate review data files (all games '
             'should be included in APPID_DICT)',
        type=str,
        default='')
    parser_add_argument('--appids',
        help='comma-separated list of game IDs for which to generate review '
             'data files (all IDs should map to games in APPID_DICT) (can be '
             'specified instead of the game names)',
        type=str,
        default='')
    parser_add_argument('--wait',
        help='amount of time in seconds to wait between making requests for '
             'pages',
        type=int,
        default=30)
    parser_add_argument('--log_file_path', '-log',
        help='path for log file',
        type=str,
        default=join(project_dir,
                     'logs',
                     'replog_get_review_data.txt'))
    args = parser.parse_args()

    # Initialize logging system
    logging_info = logging.INFO
    logger = logging.getLogger('make_train_test_sets')
    logger.setLevel(logging_info)

    # Create file handler
    fh = logging.FileHandler(abspath(args.log_file_path))
    fh.setLevel(logging_info)

    # Create console handler
    sh = logging.StreamHandler()
    sh.setLevel(logging_info)

    # Add nicer formatting
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -'
                                  ' %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    loginfo = logger.info

    # Make list of games for which to generate review data files
    if args.games:
        games = args.games.split(',')
        if not all([game in APPID_DICT for game in games]):
            from sys import exit
            exit('Could not match in APPID_DICT at least one game in the '
                 'list of passed-in games. Exiting.\n')
    elif args.appids:
        appids = parse_appids(args.appids)
        games = []
        for appid in appids:
            game = [x[0] for x in APPID_DICT.items() if x[1] == appid]
            if game:
                games.append(game[0])
            else:
                exit('Could not find game in APPID_DICT that is associated '
                     'with the given appid: {}\nExiting.\n'.format(appid))
    else:
        games = list(APPID_DICT)
        del games['sample.txt']

    # Generate review data files
    loginfo('Scraping review data from the Steam website...')
    for game in games:
        flush_to_file = 10
        out_path = join(data_dir,
                        '{}.jsonlines'.format(game))
        loginfo('Writing to {}...'.format(out_path))
        with open(out_path,
                  'w') as jsonlines_file:
            jsonlines_file_write = jsonlines_file.write
            for review in get_review_data_for_game(APPID_DICT[game],
                                                   time_out=10.0,
                                                   wait=args.wait):
                jsonlines_file_write('{}\n'.format(dumps(review,
                                                         jsonlines_file)))
                if flush_to_file == 0:
                    jsonlines_file.flush()
                    flush_to_file = 10
                else:
                    flush_to_file -= 1
    loginfo('Complete.')