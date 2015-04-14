#!/usr/env python2.7
import sys
import re
import requests
import time
from lxml import html
from bs4 import UnicodeDammit
from os import listdir
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
            print("There was a Timeout error...")
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
            sys.stderr.write('Warning: len(hours) ({}) not equal to ' \
                             'len(range_reviews) ' \
                             '({}).\n\n'.format(len(hours),
                                                len(range_reviews)))
            sys.stderr.write('{}\n\n'.format([h[:5] for h in hours]))
            sys.stderr.write('{}\n\n\n'.format(
                [r[:20] for r in range_reviews]))
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


if __name__ == '__main__':
    # Iterate over all games in APPID_DICT, getting all reviews available
    # and putting it all in the data directory
    for game in APPID_DICT:
        with open(join(data_dir,
                       '{}.txt'.format(game)),
                  'w') as of:
            for review_set in get_review_data_for_game(APPID_DICT[game],
                                                       time_out=120.0):
                for review in review_set:
                    of.write('game-hours: {0[1]}\n' \
                             'review: {0[0]}\n'.format(review))
