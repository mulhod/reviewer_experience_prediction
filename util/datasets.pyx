'''
:author: Matt Mulholland
:date: May 5, 2015

Module of code that reads review data from raw text files and returns a list of files, describes the data, etc.
'''
import sys
import time
import logging
logger = logging.getLogger()
import requests
import numpy as np
from re import sub
import pandas as pd
import seaborn as sns
from time import strftime
from data import APPID_DICT
from bs4 import (BeautifulSoup,
                 UnicodeDammit)
from langdetect import detect
import matplotlib.pyplot as plt
from os.path import (abspath,
                     basename,
                     dirname,
                     realpath,
                     join)
from langdetect.lang_detect_exception import LangDetectException

# Define a couple useful regular expressions
SPACE = re.compile(r'[\s]+')
BREAKS_REGEX = re.compile(r'\<br\>')
COMMA = re.compile(r',')

# Codecs for use with UnicodeDammit
codecs = ["windows-1252", "utf8", "ascii", "cp500", "cp850", "cp852",
          "cp858", "cp1140", "cp1250", "iso-8859-1", "iso8859_2",
          "iso8859_15", "iso8859_16", "mac_roman", "mac_latin2", "utf32",
          "utf16"]

def get_review_data_for_game(appid, time_out=0.5, limit=0, sleep=10):
    '''
    Generate dictionaries for each review for a given game.

    The dictionaries will contain keys for the review text, the reviewer ID,
    the reviewer's user-name, the number of friends the reviewer has, the
    the number of reviews the reviewer has written, and much more.

    :param appid: ID corresponding to a given game
    :type appid: str
    :param timeout: amount of time allowed to go by without hearing
                    response while using requests.get() method
    :type timeout: float
    :param limit: the maximum number of reviews to collect
    :type limit: int (default: 0, which signifies all)
    :param sleep: amount of time to wait between reading different pages on
                  the Steam websites
    :type sleep: int/float
    :yields: dictionary with keys for various pieces of data related to a
             single review, including the review itself, the number of hours
             the reviewer has played the game, etc.
    '''

    logger.info('Collecting review data for {} ({})...'
                .format(APPID_DICT[appid],
                        appid))
    logger.info('TIME_OUT = {} seconds'.format(time_out))
    logger.info('LIMIT = {} reviews'.format(limit))
    logger.info('SLEEP = {} seconds'.format(sleep))
    if limit == 0:
        limit = -1
    cdef int reviews_count = 0
    cdef int range_begin = 0
    cdef int i = 1
    cdef int breaks = 0
    while True and breaks < 100:
        # Get unique URL for values of range_begin and i
        base_url = 'http://steamcommunity.com/app/{2}/homecontent/?user' \
                   'reviewsoffset={0}&p=1&itemspage={1}&screenshotspage' \
                   '={1}&videospage={1}&artpage={1}&allguidepage={1}&web' \
                   'guidepage={1}&integratedguidepage={1}&discussionspage' \
                   '={1}&appid={2}&appHubSubSection=10&appHubSubSection=' \
                   '10&l=english&browsefilter=toprated&filterLanguage=' \
                   'default&searchText=&forceanon=1'.format(range_begin,
                                                            i,
                                                            appid)
        logger.debug('')
        # Get the URL content
        base_page = None
        time.sleep(sleep)
        # Get the HTML page; if there's a timeout error, then catch it and
        # exit out of the loop, effectively ending the function.
        try:
            base_page = requests.get(base_url,
                                     timeout=time_out)
        except requests.exceptions.Timeout as e:
            logger.error('There was a Timeout error...')
            breaks += 1
        continue
        # If there's nothing at this URL, page might have no value at all,
        # in which case we should skip the URL
        # Another situation where we'd want to skip is if page.text contains
        # only an empty string or a string that has only a sequence of one or
        # more spaces
        if not base_page:
            breaks += 1
            continue
        elif not base_page.text.strip():
            breaks += 1
            continue
        # Preprocess the HTML source, getting rid of "<br>" tags and
        # replacing any sequence of one or more carriage returns or
        # whitespace characters with a single space
        base_html = SPACE.sub(r' ',
                              BREAKS_REGEX.sub(r' ',
                                               base_page.text.strip()))
        # Try to decode the HTML to unicode and then re-encode the text
        # with ASCII, ignoring any characters that can't be represented
        # with ASCII
        base_html = UnicodeDammit(base_html,
                                  codecs).unicode_markup.encode('ascii',
                                                                'ignore')

        # Parse the source HTML with BeautifulSoup
        source_soup = BeautifulSoup(base_html,
                                    'lxml')
        reviews = soup.find_all('div',
                                'apphub_Card interactable')

        # Iterate over the reviews in the source HTML and find data for
        # each review, yielding a dictionary
        for review in reviews:

            # Get links to review URL, profile URL, Steam ID number
            review_url = review.attrs['onclick'].split(' ',
                                                       2)[1].strip("',")
            review_url_split = review_url.split('/')
            steam_id_number = review_url_split[4]
            profile_url = '/'.join(review_url_split[:5])

            # Get other data within the base reviews page
            stripped_strings = list(review.stripped_strings)
            # Parsing the HTML in this way depends on stripped_strings
            # having a length of at least 8
            if len(stripped_strings) >= 8:
                print(stripped_strings)
                # Extracting data from the text that supplies the number
                # of users who found the review helpful and/or funny
                # depends on a couple facts
                helpful_and_funny_list = stripped_strings[0].split()
                if (helpful_and_funny_list[8] == 'helpful'
                    and len(helpful_and_funny_list) == 15):
                    helpful = helpful_and_funny_list[:9]
                    funny = helpful_and_funny_list[9:]
                    num_found_helpful = int(COMMA.sub(r'',
                                                  helpful[0]))
                    num_voted_helpfulness = int(COMMA.sub(r'',
                                                          helpful[2]))
                    num_found_unhelpful = \
                        num_voted_helpfulness - num_found_helpful
                    found_helpful_percentage = \
                        float(num_found_helpful)/num_voted_helpfulness
                    num_found_funny = funny[0]
                recommended = stripped_strings[1]
                total_game_hours = COMMA.sub(r'',
                                             stripped_strings[2]
                                             .split()[0])
                date_posted = '{}, 2015'.format(stripped_strings[3][8:])
                review_text = ' '.join(stripped_strings[4:-3])
                num_games_owned = stripped_strings[-2].split()[0]
            else:
                logger.warning('Found incorrect number of "stripped_strings" '
                               'in review HTML element. stripped_strings: {}'
                               '\nContinuing.'.format(stripped_strings))
                continue

            # Make dictionary for holding all the data related to the
            # review
            review_dict = \
                dict(review_url=review_url,
                     recommended=recommended,
                     total_game_hours=total_game_hours,
                     date_posted=date_posted,
                     review=review_text,
                     num_games_owned=num_games_owned,
                     num_found_helpful=num_found_helpful,
                     num_found_unhelpful=num_found_unhelpful,
                     num_voted_helpfulness=num_voted_helpfulness,
                     found_helpful_percentage=found_helpful_percentage,
                     num_found_funny=num_found_funny,
                     steam_id_number=steam_id_number,
                     profile_url=profile_url)

            # Follow links to profile and review pages and collect data
            # from there
            time.sleep(sleep)
            review_page = requests.get(review_dict['review_url'])
            time.sleep(sleep)
            profile_page = requests.get(review_dict['profile_url'])
            review_page_html = review_page.text
            profile_page_html = profile_page.text

            # Preprocess HTML and try to decode the HTML to unicode and
            # then re-encode the text with ASCII, ignoring any characters
            # that can't be represented with ASCII
            review_page_html = \
                SPACE.sub(r' ',
                          BREAKS_REGEX.sub(r' ',
                                           review_page_html.strip()))
            review_page_html = \
                UnicodeDammit(review_page_html,
                              codecs).unicode_markup.encode('ascii',
                                                            'ignore')
            profile_page_html = \
                SPACE.sub(r' ',
                          BREAKS_REGEX.sub(r' ',
                                           profile_page_html.strip()))
            profile_page_html = \
                UnicodeDammit(profile_page_html,
                              codecs).unicode_markup.encode('ascii',
                                                            'ignore')

            # Now use BeautifulSoup to parse the HTML
            review_soup = BeautifulSoup(review_page_html,
                                        'lxml')
            profile_soup = BeautifulSoup(profile_page_html,
                                         'lxml')

            # Get the user-name from the review page
            review_dict['username'] = \
                review_soup.find('span',
                                 'profile_small_header_name').string

            # Get the number of hours the reviewer played the game in the
            # last 2 weeks
            review_dict['hours_previous_2_weeks'] = \
                COMMA.sub(r'',
                          review_soup.find('div',
                                           'playTime').string.split()[0])

            # Get the number of comments users made on the review (if any)
            review_dict['num_comments'] = \
                COMMA.sub(r'',
                          list(review_soup
                               .find('div',
                                     'commentthread_count')
                               .strings)[1])

            # Get the reviewer's "level" (friend player level)
            friend_player_level = profile_soup.find('div',
                                                    'friendPlayerLevel')
            if friend_player_level:
                review_dict['friend_player_level'] = \
                    friend_player_level.string
            else:
                review_dict['friend_player_level'] = None

            # Get the game achievements summary data
            achievements = \
                profile_soup.find('span',
                                  'game_info_achievement_summary')
            if achievements:
                achievements = achievements.stripped_strings
                if achievements:
                    achievements = list(achievements)[1].split()
                    review_dict['achievement_progress'] = \
                        dict(num_achievements_attained=achievements[0],
                             num_achievements_possible=achievements[2])
                else:
                    review_dict['achievement_progress'] = \
                        dict(num_achievements_attained=None,
                             num_achievements_possible=None)
            else:
                review_dict['achievement_progress'] = \
                    dict(num_achievements_attained=None,
                         num_achievements_possible=None)

            # Get the number of badges the reviewer has earned on the site
            badges = profile_soup.find('div',
                                       'profile_badges')
            if badges:
                badges = badges.stripped_strings
                if badges:
                    review_dict['num_badges'] = list(badges)[1]
                else:
                    review_dict['num_badges'] = None
            else:
                review_dict['num_badges'] = None

            # Get the number of reviews the reviewer has written across all
            # games and the number of screenshots he/she has taken
            reviews_screens = profile_soup.find('div',
                                                'profile_item_links')
            if reviews_screens:
                reviews_screens = reviews_screens.stripped_strings
                if reviews_screens:
                    reviews_screens = list(reviews_screens)
                    review_dict['num_screenshots'] = reviews_screens[3]
                    review_dict['num_reviews'] = reviews_screens[5]
                else:
                    review_dict['num_screenshots'] = None
                    review_dict['num_reviews'] = None
            else:
                review_dict['num_screenshots'] = None
                review_dict['num_reviews'] = None

            # Get the number of groups the reviewer is part of on the site
            groups = profile_soup.find('div',
                                       'profile_group_links')
            if groups:
                groups = groups.stripped_strings
                if groups:
                    review_dict['num_groups'] = list(groups)[1]
                else:
                    review_dict['num_groups'] = None
            else:
                review_dict['num_groups'] = None

            # Get the number of friends the reviwer has on the site
            friends = profile_soup.find('div',
                                        'profile_friend_links')
            if friends:
                friends = friends.stripped_strings
                if friends:
                    review_dict['num_friends'] = list(friends)[1]
                else:
                    review_dict['num_friends'] = None
            else:
                review_dict['num_friends'] = None

            yield review_dict

            reviews_count += 1
            if reviews_count == limit:
                break

        if reviews_count == limit:
            break

        # Increment the range_begin and i variables, which will be used in the
        # generation of the next page of reviews
        range_begin += 10
        i += 1


def parse_appids(appids):
    '''
    Parse the command-line argument passed in with the --appids flag, exiting if any of the resulting IDs do not map to games in APPID_DICT.

    :param appids: game IDs
    :type appids: str
    :returns: list of game IDs
    '''

    appids = appids.split(',')
    for appid in appids:
        if not appid in APPID_DICT.values():
            logger.error('{} not found in APPID_DICT. Exiting.'.format(appid))
            sys.exit(1)
    return appids


cdef read_reviews_from_game_file(file_path):
    '''
    Get list of reviews from a single game file.

    :param file_path: path to reviews file
    :type file_path: str
    :returns: list of dicts
    '''

    reviews = []
    lines = open(abspath(file_path)).readlines()
    cdef int i = 0
    while i + 1 < len(lines): # We need to get every 2-line couplet
        # Extract the hours value and the review text from each 2-line
        # sequence
        try:
            h = float(lines[i].split()[1].strip())
            r = lines[i + 1].split(' ', 1)[1].strip()
        except (ValueError, IndexError) as e:
            i += 2
            continue
        # Skip reviews that don't have any characters
        if not len(r):
            i += 2
            continue
        # Skip reviews if they cannot be recognized as English
        try:
            if not detect(r) == 'en':
                i += 2
                continue
        except LangDetectException:
            i += 2
            continue
        # Now we append the 2-key dict to the end of reviews
        reviews.append(dict(hours=h,
                            review=r))
        i += 2 # Increment i by 2 since we need to go to the next
            # 2-line couplet
    return reviews


def get_and_describe_dataset(file_path, report=True):
    '''
    Return dictionary with a list of filtered review dictionaries as well as the filtering values for maximum/minimum review length and minimum/maximum hours played values and the number of original, English-language reviews (before filtering); also produce a report with some descriptive statistics and graphs.

    :param file_path: path to game reviews file
    :type file_path: str
    :param report: make a report describing the data-set (defaults to True)
    :type report: boolean
    :returns: dict containing a 'reviews' key mapped to the list of read-in review dictionaries and int values mapped to keys for MAXLEN, MINLEN, MAXHOURS, and MINHOURS
    '''

    if report:

        # Get path to reports directory and open report file
        reports_dir = join(dirname(dirname(realpath(__file__))),
                           'reports')
        game = basename(file_path)[:-4]
        output_path = join(reports_dir,
                           '{}_report.txt'.format(game))
        output = open(output_path,
                      'w')

        # Initialize seaborn-related stuff
        sns.set_palette("deep",
                        desat=.6)
        sns.set_context(rc={"figure.figsize": (14, 7)})

    # Get list of review dictionaries
    reviews = read_reviews_from_game_file(file_path)

    if report:
        output.write('Descriptive Report for {}\n============================'
                     '===================================================\n'
                     '\n'.format(sub(r'_',
                                     r' ',
                                     game)))
        output.write('Number of English-language reviews: {}\n'
                     '\n'.format(len(reviews)))

    # Look at review lengths to figure out what should be filtered out
    lengths = np.array([len(review['review']) for review in reviews])
    cdef float meanl = lengths.mean()
    cdef float stdl = lengths.std()
    if report:
        output.write('Review Lengths Distribution\n\n')
        output.write('Average review length: {}\n'.format(meanl))
        output.write('Minimum review length = {}\n'.format(min(lengths)))
        output.write('Maximum review length = {}\n'.format(max(lengths)))
        output.write('Standard deviation = {}\n\n\n'.format(stdl))

    # Use the standard deviation to define the range of acceptable reviews
    # (in terms of the length only) as within 2 standard deviations of the
    # mean (but with the added caveat that the reviews be at least 50
    # characters
    cdef float minl = 50.0 if (meanl - 2*stdl) < 50 else (meanl - 2*stdl)
    cdef float maxl = meanl + 2*stdl

    if report:
        # Generate length histogram
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(pd.Series(lengths))
        ax.set_label(game)
        ax.set_xlabel('Review length (in characters)')
        ax.set_ylabel('Total reviews')
        fig.savefig(join(reports_dir,
                         '{}_length_histogram'.format(game)))
        plt.close(fig)

    # Look at hours played values in the same way as above for length
    hours = np.array([review['hours'] for review in reviews])
    cdef float meanh = hours.mean()
    cdef float stdh = hours.std()
    if report:
        output.write('Review Experience Distribution\n\n')
        output.write('Average game experience (in hours played): {}'
                     '\n'.format(meanh))
        output.write('Minimum experience = {}\n'.format(min(hours)))
        output.write('Maximum experience = {}\n'.format(max(hours)))
        output.write('Standard deviation = {}\n\n\n'.format(stdh))

    # Use the standard deviation to define the range of acceptable reviews
    # (in terms of experience) as within 2 standard deviations of the mean
    # (starting from zero, actually)
    cdef float minh = 0.0
    cdef float maxh = meanh + 2*stdh

    # Write MAXLEN, MINLEN, etc. values to report
    if report:
        output.write('Filter Values\n'
                     'Minimum length = {}\n'
                     'Maximum length = {}\n'
                     'Minimum hours played = {}\n'
                     'Maximum hours played = {}\n'.format(minl,
                                                          maxl,
                                                          minh,
                                                          maxh))

    # Generate experience histogram
    if report:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(pd.Series(hours))
        ax.set_label(game)
        ax.set_xlabel('Game experience (in hours played)')
        ax.set_ylabel('Total reviews')
        fig.savefig(join(reports_dir,
                         '{}_experience_histogram'.format(game)))
        plt.close(fig)

    if report:
        output.close()
    cdef int orig_total_reviews = len(reviews)
    reviews = [r for r in reviews if len(r['review']) <= maxl
                                     and len(r['review']) >= minl
                                     and r['hours'] <= maxh]
    return dict(reviews=reviews,
                minl=minl,
                maxl=maxl,
                minh=minh,
                maxh=maxh,
                orig_total_reviews=orig_total_reviews)


def get_bin_ranges(float _min, float _max, int nbins):
    '''
    Return list of floating point number ranges (in increments of 0.1) that correspond to each bin in the distribution.

    :param _min: minimum value of the distribution
    :type _min: float
    :param _max: maximum value of the distribution
    :type _max: float
    :param nbins: number of bins into which the distribution is being sub-divided
    :type nbins: int
    :returns: list of tuples representing the minimum and maximum values of each bin
    '''

    cdef float bin_size = round((_max - _min)/<float>nbins, 1)
    bin_ranges = []
    cdef float _bin_start = _min - 0.1
    cdef float _bin_end = _min + bin_size
    cdef int b
    for b in range(1, nbins + 1):
        if not b == 1:
            _bin_start = _bin_end
        if b == nbins:
            _bin_end = _bin_start + bin_size + 1.0
        else:
            _bin_end = _bin_start + bin_size
        bin_ranges.append((_bin_start,
                           _bin_end))
    return bin_ranges


def get_bin(bin_ranges, float val):
    '''
    Return the index of the bin range in which the value falls.

    :param bin_ranges: list of ranges that define each bin
    :type bin_ranges: list of tuples representing the minimum and maximum values of a range of values
    :param val: value
    :type val: float
    :returns: int (-1 if val not in any of the bin ranges)
    '''

    cdef int i
    for i, bin_range in enumerate(bin_ranges):
        if (val > bin_range[0]
            and val <= bin_range[1]):
            return i + 1
    return -1


def write_arff_file(dest_path, file_names, reviews=None, reviewdb=None,
                    make_train_test=False, bins=False):
    '''
    Write .arff file either for a list of reviews read in from a file or list of files or for both the training and test partitions in the MongoDB database.

    :param reviews: list of dicts with hours/review keys-value mappings representing each data-point (defaults to None)
    :type reviews: list of dict
    :param reviewdb: MongoDB reviews collection
    :type reviewdb: pymongo.MongoClient object (None by default)
    :param dest_path: path for .arff output file
    :type dest_path: str
    :param file_names: list of extension-less game file-names
    :type file_names: list of str
    :param make_train_test: if True, use MongoDB collection to find reviews that are from the training and test partitions and make files for them instead of making one big file (defaults to False)
    :type make_train_test: boolean
    :param bins: if True or a list of bin range tuples, use collapsed hours played values (if make_train_test was also True, then the pre-computed collapsed hours values will be used (even if a list of ranges is passed in for some reason, i.e., the bin ranges will be ignored); if not, the passed-in value must be a list of 2-tuples representing the floating-point number ranges of the bins); if False, the original, unmodified hours played values will be used (default: False)
    :type bins: boolean or list of 2-tuples of floats
    :returns: None
    '''

    # Make sure that the passed-in keyword arguments make sense
    if (make_train_test
        and (reviews or not reviewdb)):
        logger.error('The make_train_test keyword argument was set to True '
                     'and either the reviewdb keyword was left unspecified or'
                     ' the reviews keyword was specified (or both). If the '
                     'make_train_test keyword is used, it is expected that '
                     'training/test reviews will be retrieved from the'
                     ' MongoDB database rather than a list of reviews passed '
                     'in via the reviews keyword. Exiting.')
        sys.exit(1)
    if (not make_train_test
        and reviewdb):
        if reviews:
            logger.warning('Ignoring passed-in reviewdb keyword value. '
                           'Reason: If a list of reviews is passed in via the'
                           ' reviews keyword argument, then the reviewdb '
                           'keyword argument should not be used at all since '
                           'it will not be needed.')
        else:
            logger.error('A list of review dictionaries was not specified. '
                         'Exiting.')
            sys.exit(1)
    if bins:
        if (make_train_test
            and type(bins) == list):
            logger.warning('The write_arff_file method was called with '
                           '\'make_train_test\' set to True and \'bins\' set '
                           'to a list of bin ranges ({}). Because the bin '
                           'values in the database were precomputed, the '
                           'passed-in list of bin ranges will be '
                           'ignored.'.format(repr(bins)))
        if (reviews
            and type(bins) == bool):
            logger.error('The write_arff_file method was called with a list '
                         'of review dictionaries and \'bins\' set to True. If'
                         ' the hours played values are to be collapsed and '
                         'precomputed values (as from the database, for '
                         'example) are not being used, then the bin ranges '
                         'must be specified. Exiting.')
            sys.exit(1)

    # ARFF file template
    ARFF_BASE = '''% Generated on {}
% This ARFF file was generated with review data from the following game(s): {}
% It is useful only for trying out machine learning algorithms on the bag-of-words representation of the reviews.
@relation reviewer_experience
@attribute string_attribute string
@attribute numeric_attribute numeric

@data'''
    TIMEF = '%A, %d. %B %Y %I:%M%p'

    # Replace underscores with spaces in game names and make
    # comma-separated list of games
    _file_names = str([sub(r'_',
                           r' ',
                           f) for f in file_names])

    # Write ARFF file(s)
    if make_train_test:

        # Make an ARFF file for each partition
        for partition in ['training', 'test']:

            # Make empty list of lines to populate with ARFF-style lines,
            # one per review
            reviews_lines = []

            # Get reviews for the given partition from all of the games
            game_docs = reviewdb.find({'partition': partition,
                                       'game': {'$in': file_names}})
            if game_docs.count() == 0:
                logger.error('No matching documents were found in the '
                             'MongoDB collection for the {} partition and the'
                             ' following games:\n\n{}\n\nExiting'
                             '.'.format(partition,
                                        file_names))
                sys.exit(1)

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
                hours = game_doc['hours_bin'] if bins else game_doc['hours']
                reviews_lines.append('"{}",{}'.format(review,
                                                      hours))

            # Modify file-path by adding suffix(es)
            suffix = 'train' if partition.startswith('train') else 'test'
            replacement = suffix + '.arff'
            if bins:
                replacement = 'bins.' + replacement
            _dest_path = dest_path[:-4] + replacement

            # Write to file
            with open(_dest_path,
                      'w') as out:
                out.write('{}\n{}'.format(ARFF_BASE.format(strftime(TIMEF),
                                                           _file_names),
                                          '\n'.join(reviews_lines)))
    else:

        if not reviews:
            logger.error('Empty list of reviews passed in to the '
                         'write_arff_file method. Exiting.')
            sys.exit(1)

        # Make empty list of lines to populate with ARFF-style lines,
        # one per review
        reviews_lines = []

        for rd in reviews:
            # Remove single/double quotes from the reviews first...
            review = sub(r'\'|"',
                         r'',
                         rd['review'].lower())
            # Get rid of backslashes since they only make things confusing
            review = sub(r'\\',
                         r'',
                         review)
            if bins:
                hours = get_bin(bins,
                                rd['hours'])
                if hours < 0:
                    logger.error('The given hours played value ({}) was not '
                                 'found in the list of possible bin ranges '
                                 '({}). Exiting.'.format(rd['hours'],
                                                         bins))
                    sys.exit(1)
            else:
                hours = rd['hours']
            reviews_lines.append('"{}",{}'.format(review,
                                                  hours))

        # Modify file-path by adding partition suffix
        if bins:
            dest_path = dest_path[:-4] + 'bins.arff'

        # Write to file
        with open(dest_path,
                  'w') as out:
            out.write('{}\n{}'.format(ARFF_BASE.format(strftime(TIMEF),
                                                       _file_names),
                                      '\n'.join(reviews_lines)))