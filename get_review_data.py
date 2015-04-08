#!/usr/env python2.7
import sys
import re
import requests
import time
from lxml import html
from bs4 import UnicodeDammit

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
    while True:
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
        time.sleep(60)
        try:
            page = requests.get(url, timeout=time_out)
        except requests.exceptions.Timeout as e:
            print("There was a Timeout error...")
            break
        # If there's nothing at this URL, page might have no value at all,
        # in which case we should break out of the loop
        if not page:
            break
        elif not page.text.strip():
            break
        # Preprocess the HTML source a little bit
        text = re.sub(r'[\n\t\r ]+',
                      r' ',
                      re.sub(r'\<br\>',
                             r' ',
                             page.text.strip())) # Replace the string "<br>"
            # with a space and replace any sequence of carriage returns or
            # whitespace characters with a single space
        # Convert to UTF-8
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
        # Try to decode the reviews with a number of different formats and
        # then encode all to utf-8
        # Zip the values together, processing them also
        yield [(z.strip(),
                float(re.sub(r',',
                             r'',
                             w.split(' ',
                                     1)[0]))) for z, w in zip(range_reviews,
                                                              hours)]
        # If a limit was defined and processing 10 more essays will push us
        # over the limit, stop here
        if limit and range_begin > limit:
            break
        range_begin += 10
        i += 1


if __name__ == '__main__':

    with open('out.tsv',
              'w') as of:
        of.write('review\thours\n')
        for review_set in get_review_data_for_game('730',
                                                   time_out=2.0,
                                                   limit=500):
            for review in review_set:
                of.write('{0[0]}\t{0[1]}\n'.format(review))
            sys.stdout.flush()