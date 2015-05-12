import sys
import re
import time
import logging
import argparse
import requests
from lxml import html
from os import listdir
from bs4 import UnicodeDammit
from os.path import dirname, abspath, realpath, join

main_dir = dirname(dirname(abspath(realpath(__file__))))
util_dir = join(main_dir, 'util')
data_dir = join(main_dir, 'data')
sys.path.append(main_dir)
from data import APPID_DICT

codecs = ["windows-1252", "utf8", "ascii", "cp500", "cp850", "cp852",
          "cp858", "cp1140", "cp1250", "iso-8859-1", "iso8859_2",
          "iso8859_15", "iso8859_16", "mac_roman", "mac_latin2", "utf32",
          "utf16"]

def get_review_data_for_game(appid, time_out=0.5, limit=0):
    '''
    Get list of tuples representing all review/game-hours played pairs for a given game.

    :param appid: ID corresponding to a given game
    :type appid: str
    :param timeout: amount of time allowed to go by without hearing response while using requests.get() method
    :type timeout: float
    :param limit: the maximum number of reviews to collect
    :type limit: int (default: 0, which signifies all)
    :yields: lists of tuples
    '''

    # Get reviews from each page that has content, starting at range_begin
    # = 0 and i = 1, yielding the list of review tuples as they're found
    range_begin = 0
    i = 1
    breaks = 0
    while True and breaks < 100:
        # Get unique URL for values of range_begin and i
        url = 'http://steamcommunity.com/app/{2}/homecontent/?userreviews' \
              'offset={0}&p=1&itemspage={1}&screenshotspage={1}&videospag' \
              'e={1}&artpage={1}&allguidepage={1}&webguidepage={1}&integr' \
              'atedguidepage={1}&discussionspage={1}&appid={2}&appHubSubS' \
              'ection=10&appHubSubSection=10&l=english&browsefilter=topra' \
              'ted&filterLanguage=default&searchText=&forceanon=1'.format(
                  range_begin,
                  i,
                  appid)
        # Try to get the URL content
        page = None
        try:
            page = requests.get(url, timeout=time_out)
        except requests.exceptions.Timeout as e:
            logger.info("There was a Timeout error...")
            breaks += 1
            continue
        # If there's nothing at this URL, page might have no value at all,
        # in which case we should break out of the loop
        if not page:
            breaks += 1
            continue
        elif not page.text.strip():
            breaks += 1
            continue
        # Preprocess the HTML source a little bit
        text = re.sub(r'[\n\t\r ]+',
                      r' ',
                      re.sub(r'\<br\>',
                             r' ',
                             page.text.strip())) # Replace the string "<br>"
            # with a space and replace any sequence of carriage returns or
            # whitespace characters with a single space
        # Try to decode the HTML source and then re-encode it with
        # the 'ascii' encoding
        text = UnicodeDammit(text,
                             codecs).unicode_markup.encode('ascii',
                                                           'ignore')
        # Get the parse tree from source html
        tree = html.fromstring(text.strip())
        # Get lists of review texts and values for game-hours played
        range_reviews = \
            tree.xpath('//div[@class="apphub_CardTextContent"]/text()')
        hours = tree.xpath('//div[@class="hours"]/text()')
        # Truncate the list of reviews by getting rid of elements that are
        # either empty or have only a single space
        range_reviews = [x.strip() for x in range_reviews if x.strip()]
        # Make sure that the assumption that the lists of data line up
        # correctly actually is true and, if not, print out some debugging
        # info to STDERR and skip to the next review page without adding
        # any reviews to the file
        try:
            assert len(hours) == len(range_reviews)
        except AssertionError:
            logger.debug('Warning: len(hours) ({}) not equal to ' \
                         'len(range_reviews) ({}).\n'.format(len(hours),
                                                          len(range_reviews)))
            logger.debug('URL: {}'.format(url))
            logger.debug('{}\n'.format([h[:5] for h in hours]))
            logger.debug('{}\n\n'.format([r[:20] for r in range_reviews]))
            range_begin += 10
            i += 1
            time.sleep(120)
            continue
        # Try to decode the reviews with a number of different formats and
        # then encode all to utf-8
        # Zip the values together, processing them also
        yield [(z.strip(),
                float(re.sub(r',',
                             r'',
                             w.split(' ',
                                     1)[0]))) for z, w in zip(range_reviews,
                                                              hours)]
        # Increment the range_begin and i variables
        range_begin += 10
        i += 1
        # If a limit was defined and processing 10 more essays will push us
        # over the limit, stop here
        if limit and range_begin + 10 > limit:
            break
        time.sleep(120)


def parse_appids(appids):
    '''
    Parse the command-line argument passed in with the --appids flag, exiting if any of the resulting IDs do not map to games in APPID_DICT.

    :param appids: game IDs
    :type appids: str
    :returns: list of game IDs
    '''

    global APPID_DICT
    appids = appids.split(',')
    for appid in appids:
        if not appid in APPID_DICT.values():
            logger.info('ERROR: {} not found in APPID_DICT. ' \
                        'Exiting.'.format(appid))
    return appids


if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage='./python get_review_data.py',
        description='Make review data files for each game in the APPID_DICT' \
                    ', which is specified in the __init__.py module in the ' \
                    'the "data" directory. A specific list of game IDs ' \
                    '(same as "appid") can also be specified instead, but ' \
                    'they must map to games in APPID_DICT.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--appids',
        help='comma-separated list of game IDs for which to generate review' \
             ' data files (all IDs should map to games to APPID_DICT)',
        type=str,
        required=False)
    args = parser.parse_args()

    # Initialize logging system
    logger = logging.getLogger('rep.get')
    logger.setLevel(logging.DEBUG)

    # Create console handler with a high logging level specificity
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)

    # Add nicer formatting
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -'
                                  ' %(message)s')
    #fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    #logger.addHandler(fh)
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
    for game in games:
        with open(join(data_dir,
                       '{}.txt'.format(game)),
                  'w') as of:
            for review_set in get_review_data_for_game(APPID_DICT[game],
                                                       time_out=60.0):
                for review in review_set:
                    of.write('game-hours: {0[1]}\n' \
                             'review: {0[0]}\n'.format(review))
                of.flush()