"""
:author: Matt Mulholland
:date: April 1, 2015

Script used to run the web scraping tool in order to build the video
game review corpus.
"""
import logging
from os import makedirs
from os.path import (join,
                     dirname,
                     realpath)

from argparse import (ArgumentParser,
                      ArgumentDefaultsHelpFormatter)

from src import (log_dir,
                 data_dir,
                 formatter)

# Initialize logging system
logger = logging.getLogger(__name__)
logging_info = logging.INFO
logger.setLevel(logging_info)
sh = logging.StreamHandler()
sh.setLevel(logging_info)
sh.setFormatter(formatter)
logger.addHandler(sh)


def main():
    parser = ArgumentParser(usage='./python get_review_data.py[ --appids '
                                  'APPID1,APPID2,...]',
                            description='Make review data files for each game'
                                        ' in the APPID_DICT, which is '
                                        'specified in the `__init__.py` file '
                                        'in the `data` package. A specific '
                                        'list of games or game IDs (same as '
                                        '`appid`) can also be specified '
                                        'instead, but they must map to games '
                                        'in `APPID_DICT`.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    _add_arg = parser.add_argument
    _add_arg('--games',
             help='Comma-separated list of game names (underscores replacing '
                  'all spaces) for which to generate review data files (all '
                  'games should be included in `APPID_DICT`)',
             type=str,
             default='')
    _add_arg('--appids',
             help='Comma-separated list of game IDs for which to generate '
                  'review data files (all IDs should map to games in '
                  '`APPID_DICT`) (can be specified instead of the game '
                  'names).',
             type=str,
             default='')
    _add_arg('--wait',
             help='Amount of time in seconds to wait between making requests '
                  'for pages.',
             type=int,
             default=30)
    _add_arg('--log_file_path', '-lo.g',
             help='Path for log file',
             type=str,
             default=join(log_dir, 'replog_get_review_data.txt'))
    args = parser.parse_args()

    # Imports
    from json import dumps
    from data import (APPID_DICT,
                      parse_appids)
    from src import log_format_string
    from src.datasets import get_review_data_for_game

    # Make sure log file directory exists
    log_file_path = realpath(args.log_file_path)
    log_file_dir = dirname(log_file_path)
    if not exists(log_file_dir):
        makedirs(log_file_dir, exist_ok=True)

    # Make file handler
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging_info)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Make list of games for which to generate review data files
    if args.games:
        games = args.games.split(',')
        if not all(game in APPID_DICT for game in games):
            raise ValueError('Could not match in APPID_DICT at least one game'
                             ' in the list of passed-in games.')
    elif args.appids:
        appids = parse_appids(args.appids)
        games = []
        for appid in appids:
            game = [x[0] for x in APPID_DICT.items() if x[1] == appid]
            if game:
                games.append(game[0])
            else:
                raise ValueError('Could not find game in APPID_DICT that is '
                                 'associated with the given appid: {0}'
                                 .format(appid))
    else:
        games = list(APPID_DICT)
        del games['sample.txt']

    # Generate review data files
    loginfo('Scraping review data from the Steam website...')
    for game in games:
        flush_to_file = 10
        out_path = join(data_dir, '{0}.jsonlines'.format(game))
        loginfo('Writing to {0}...'.format(out_path))
        with open(out_path, 'w') as jsonlines_file:
            jsonlines_file_write = jsonlines_file.write
            for review in get_review_data_for_game(APPID_DICT[game], time_out=10.0,
                                                   wait=float(args.wait)):
                jsonlines_file_write('{0}\n'.format(dumps(review)))
                if flush_to_file == 0:
                    jsonlines_file.flush()
                    flush_to_file = 10
                else:
                    flush_to_file -= 1
    loginfo('Complete.')

if __name__ == '__main__':
    main()
