"""
:author: Matt Mulholland
:date: May 5, 2015

Cython extension for functions that collect review data from Steam,
read review data from raw text files, describe review data in terms of
descriptive statistics, write ARFF format files for use with Weka,
convert raw hours played values to a scale of a given number of values,
etc.
"""
import logging
from os import listdir
from json import loads
from time import (sleep,
                  strftime)
from os.path import (join,
                     exists,
                     dirname,
                     basename,
                     realpath,
                     splitext)

import numpy as np
import pandas as pd
import seaborn as sns
from langdetect import detect
from pymongo import collection
from bs4 import (BeautifulSoup,
                 UnicodeDammit)
import matplotlib.pyplot as plt
from requests import get as rget
from requests.exceptions import (Timeout,
                                 ConnectionError)
from langdetect.lang_detect_exception import LangDetectException

from data import APPID_DICT
from src import (comma_sub,
                 space_sub,
                 breaks_sub,
                 quotes_sub,
                 backslash_sub,
                 underscore_sub,
                 ACHIEVEMENTS_LABELS,
                 comment_re_1_search,
                 comment_re_2_search,
                 LABELS_WITH_PCT_VALUES,
                 helpful_or_funny_search,
                 test_float_decimal_places,
                 date_end_with_year_string_search)

# Logging-related
logger = logging.getLogger()
loginfo = logger.info
logdebug = logger.debug
logwarn = logger.warning
logerr = logger.error

def get_game_files(games_str: str, data_dir_path: str) -> list:
    """
    Get list of game files (file-names only).

    :param games_str: comma-separated list of game files (that exist in
                      the data directory) with or without a .jsonlines
                      suffix (or "all" for all game files) (Note: if
                      "sample"/"sample.jsonlines" is included it will be
                      filtered out)
    :type games_str: str
    :param data_dir_path: path to data directory
    :type data_dir_path: str

    :returns: list of games
    :rtype: list

    :raises ValueError: no games were included in the list of games (or
                        `games_str` only includes
                        "sample"/"sample.jsonlines")
    :raises FileNotFoundError: if file(s) corresponding to games in the
                               input cannot be found
    """

    if not games_str or games_str in ['sample', 'sample.jsonlines']:
        raise ValueError('No files passed in via --game_files argument were '
                         'found: {}.'.format(', '.join(games_str.split(','))))
    elif games_str == "all":
        game_files = [f for f in listdir(data_dir_path) if f.endswith('.jsonlines')]

        # Remove the sample game file from the list
        del game_files[game_files.index('sample.jsonlines')]
    else:
        game_files = []
        for f in games_str.split(','):
            f_path = join(data_dir_path,
                          f if f.endswith('.jsonlines')
                          else '{0}.jsonlines'.format(f))
            if not exists(f_path):
                raise FileNotFoundError('{0} does not exist (input string: {1}).'
                                        .format(f_path, games_str))
            game_files.append(f if f.endswith('.jsonlines')
                              else '{0}.jsonlines'.format(f))

    return game_files


def get_review_data_for_game(appid: str,
                             time_out: float = 10.0,
                             limit: int = -1,
                             wait: float = 10.0) -> dict:
    """
    Generate dictionaries for each review for a given game.

    The dictionaries will contain keys for the review text, the
    reviewer ID, the reviewer's user-name, the number of friends the
    reviewer has, the the number of reviews the reviewer has written,
    and much more.

    :param appid: ID corresponding to a given game
    :type appid: str
    :param time_out: amount of time allowed to go by without hearing
                     response while using `requests.get` method
    :type time_out: float
    :param limit: the maximum number of reviews to collect (defaults to
                  -1, which signifies no limit)
    :type limit: int
    :param wait: amount of time to wait between requesting different
                 pages on the Steam website
    :type wait: float

    :yields: dictionary with keys for various pieces of data related to
             a single review, including the review itself, the number
             of hours the reviewer has played the game, etc.
    :ytype: dict

    :raises ValueError: encounterd connection error, any of various
                        types of errors while trying to parse HTML,
                        etc.
    """

    # Codecs for use with UnicodeDammit
    codecs = ["windows-1252", "utf8", "ascii", "cp500", "cp850", "cp852", "cp858",
              "cp1140", "cp1250", "iso-8859-1", "iso8859_2", "iso8859_15",
              "iso8859_16", "mac_roman", "mac_latin2", "utf32", "utf16"]

    # Base URL for pages of reviews
    BASE_URL = 'http://steamcommunity.com/app/{2}/homecontent/?userreviewsoffset={0}&p=1&itemspage={1}&screenshotspage={1}&videospage={1}&artpage={1}&allguidepage={1}&webguidepage={1}&integratedguidepage={1}&discussionspage={1}&appid={2}&appHubSubSection=10&appHubSubSection=10&l=english&browsefilter=toprated&filterLanguage=default&searchText=&forceanon=1'

    NO_RATINGS = 'No ratings yet'

    loginfo('Collecting review data for {0} ({1})...'
            .format([x[0] for x in APPID_DICT.items() if x[1] == appid][0], appid))
    loginfo('TIME_OUT = {0} seconds'.format(time_out))
    loginfo('LIMIT = {0} reviews'.format(limit))
    loginfo('SLEEP = {0} seconds'.format(wait))

    cdef int reviews_count = 0
    cdef int range_begin = 0
    cdef int i = 1
    cdef int breaks = 0
    while True and breaks < 100:

        # Get unique URL for values of range_begin and i
        url = BASE_URL.format(range_begin, i, appid)
        loginfo('Collecting review data from the following URL: {0}'.format(url))

        # Get the URL content
        base_page = None
        sleep(wait)

        # Get the HTML page; if there's a timeout error, then catch it
        # and exit out of the loop, effectively ending the function.
        try:
            base_page = rget(url, timeout=(0.1, time_out))
        except (Timeout,
                ConnectionError) as e:
            if type(e) == type(Timeout()):
                logerr('There was a Timeout error. Skipping to next review.')
                breaks += 1
                continue
            else:
                logdebug('ConnectionError encountered. Will try to wait for a'
                         ' bit before trying to reconnect again.')
                sleep(1000)
                try:
                    base_page = rget(url, timeout=(0.1, time_out))
                except (Timeout, ConnectionError) as e:
                    if type(e) == type(Timeout()):
                        logerr('There was a Timeout error. Skipping to next '
                               'review.')
                        breaks += 1
                        continue
                    else:
                        error_msg = \
                            ('Encountered second ConnectionError in a row. '
                             'Gracefully exiting program. Here are the '
                             'details for continuing:\nReviews count = {0}\n'
                             'Range start = {1}\nindex = {2}\nURL = {3}'
                             .format(reviews_count, range_begin, i, url))
                        raise ValueError(error_msg)

        """
        If there's nothing at this URL, page might have no value at
        all, in which case we should skip the URL.
        Another situation where we'd want to skip is if page.text
        contains only an empty string or a string that has only a
        sequence of one or more spaces.
        """
        if not base_page:
            breaks += 1
            continue
        elif not base_page.text.strip():
            breaks += 1
            continue

        # Preprocess the HTML source, getting rid of "<br>" tags and
        # replacing any sequence of one or more carriage returns or
        # whitespace characters with a single space
        base_html = space_sub(r' ', breaks_sub(r' ', base_page.text.strip()))

        # Try to decode the HTML to unicode and then re-encode the text
        # with ASCII, ignoring any characters that can't be represented
        # with ASCII
        base_html = (UnicodeDammit(base_html, codecs)
                     .unicode_markup.encode('ascii', 'ignore'))

        # Parse the source HTML with BeautifulSoup
        source_soup = BeautifulSoup(base_html, 'lxml')
        reviews = source_soup.find_all('div', 'apphub_CardContentMain')
        link_blocks = source_soup.findAll('div',
                                          'apphub_Card modalContentLink interactable')

        # Iterate over the reviews/link blocks in the source HTML and
        # find data for each review, yielding a dictionary
        for (review, link_block) in zip(reviews, link_blocks):

            # Get links to review URL, profile URL, Steam ID number
            review_url = link_block.attrs['data-modal-content-url'].strip()
            review_url_split = review_url.split('/')
            profile_url = '/'.join(review_url_split[:5]).strip()

            # Get other data within the base reviews page
            stripped_strings = list(review.stripped_strings)

            if not (review_url.startswith('http://')
                    or len(review_url_split) > 4):
                logerr('Found review with invalid review_url: {0}\nThe rest '
                       'of the review\'s contents: {1}\nReview URL: {2}\n'
                       'Continuing on to next review.'
                       .format(review_url, stripped_strings, url))
                continue
            try:
                steam_id_number = review_url_split[4]
                if not steam_id_number:
                    raise ValueError
            except ValueError as e:
                logerr('Empty steam_id_number. URL: {0}\nstripped_strings: '
                       '{1}\nContinuing on to next review.'
                       .format(url, stripped_strings))
                continue

            """
            Parsing the stripped_strings attribute depends on its
            containing the following items (not all will be collected
            here, however):
                 1) number of people who found the review helpful
                    and/or funny
                 2) the "recommended" value
                 3) the hours on record
                 4) the date posted (and updated, if present)
                 5+) the review, which can stretch over several
                     elements
            All that we are really interested in collecting from here
            are 1 and 5+
            If the first element in stripped_strings has the value "No
            ratings yet", then 0s will be assumed for all the values
            related to ratings (except found_helpful_percentage, which
            will be set to None).
            And, if the first element in stripped_strings is not
            related to the number of ratings by other users, then we
            can simply add "No ratings yet" to the beginning of the
            list
            """
            helpful_funny_str = stripped_strings[0].strip()
            if not (helpful_or_funny_search(helpful_funny_str)
                    or helpful_funny_str == NO_RATINGS):
                stripped_strings = [NO_RATINGS] + stripped_strings
            if len(stripped_strings) >= 5:
                helpful_and_funny_list = stripped_strings[0].strip().split()

                """
                Extracting data from the text that supplies the number
                of users who found the review helpful and/or funny
                depends on a couple of facts about what is in this
                particular string.
                """
                num_found_helpful = 0
                num_voted_helpfulness = 0
                num_found_unhelpful = 0
                found_helpful_percentage = None
                num_found_funny = 0

                """
                If there are exactly 15 elements in the split string,
                then the string appears to conform to the canonical
                format, i.e., data about how many users found the
                review both helpful and funny is supplied; if the
                string contains exactly 6 or 9 elements, then only data
                related to the users who found the review funny or
                helpful, respectively, is supplied; if the string is
                equal to the value "No ratings yet", then the values
                defined above can be kept as they are; finally, if the
                string does not meet any of the conditions above, then
                it is an aberrant string and cannot be processed.
                """
                if (len(helpful_and_funny_list) == 15
                    or len(helpful_and_funny_list) == 9
                    or len(helpful_and_funny_list) == 6):
                    if len(helpful_and_funny_list) == 15:
                        helpful = helpful_and_funny_list[:9]
                        funny = helpful_and_funny_list[9:]
                    elif len(helpful_and_funny_list) == 9:
                        helpful = helpful_and_funny_list
                        funny = None
                    else:
                        helpful = None
                        funny = helpful_and_funny_list

                    # Extract the number of people who found the review
                    # helpful
                    if helpful:
                        num_found_helpful = comma_sub(r'', helpful[0])
                        try:
                            num_found_helpful = int(num_found_helpful)
                        except ValueError:
                            logerr('Could not cast num_found_helpful value to'
                                   ' an int: {0}\nRest of review: {1}\nURL: '
                                   '{2}\nContinuing on to next review.'
                                   .format(num_found_helpful, stripped_strings, url))
                            continue

                        # Extract the number of people who voted on
                        # whether or not the review was helpful
                        # (whether it was or wasn't)
                        num_voted_helpfulness = comma_sub(r'', helpful[2])
                        try:
                            num_voted_helpfulness = int(num_voted_helpfulness)
                        except ValueError:
                            logerr('Could not cast num_voted_helpfulness '
                                   'value to an int: {0}\nRest of review: '
                                   '{1}\nURL: {2}\nContinuing on to next '
                                   'review.'.format(num_voted_helpfulness,
                                                    stripped_strings, url))
                            continue

                        # Calculate the number of people who must have
                        # found the review NOT helpful and the
                        # percentage of people who found the review
                        # helpful
                        num_found_unhelpful = \
                            num_voted_helpfulness - num_found_helpful
                        found_helpful_percentage = \
                            float(num_found_helpful)/num_voted_helpfulness

                    # Extract the number of people who found the review
                    # funny
                    if funny:
                        num_found_funny = comma_sub(r'', funny[0])
                        try:
                            num_found_funny = int(num_found_funny)
                        except ValueError:
                            logerr('Could not cast num_found_funny value to '
                                   'an int: {0}\nRest of review: {1}\nURL: '
                                   '{2}\nContinuing on to next review.'
                                   .format(num_found_funny, stripped_strings, url))
                            continue
                elif stripped_strings[0].strip() == NO_RATINGS:
                    pass
                else:
                    logerr("Found review with a helpful/funny string that "
                           "does not conform to the expected format: {0}\n"
                           "Rest of the review's contents: {1}\nURL: {2}\n"
                           "Continuing on to next review."
                           .format(stripped_strings[0], stripped_strings[1:], url))
                    continue

                # The review text should be located at index 4, but, in
                # some cases, the review text is split over several
                # indices
                review_text = ' '.join(stripped_strings[4:]).strip()

                # Skip reviews that don't have any characters
                if not review_text:
                    logdebug('Found review with apparently no review text: '
                             '{0}\nURL: {1}\nContinuing on to next review.'
                             .format(stripped_strings, url))
                    continue

                # Skip review if it is not recognized as English
                try:
                    if not detect(review_text) == 'en':
                        continue
                except LangDetectException:
                    logdebug('Found review whose review text content was not '
                             'recognized as English: {0}\nRest of review: {1}'
                             '\nURL: {2}\nContinuing on to next review.'
                             .format(review_text, stripped_strings, url))
                    continue
            else:
                logerr('Found incorrect number of "stripped_strings" in '
                       'review HTML element. The review\'s stripped_strings: '
                       '{0}\nURL: {1}\nContinuing on to next review.'
                       .format(stripped_strings, url))
                continue

            # Make dictionary for holding all the data related to the
            # review
            review_dict = dict(review_url=review_url,
                               review=review_text,
                               num_found_helpful=num_found_helpful,
                               num_found_unhelpful=num_found_unhelpful,
                               num_voted_helpfulness=num_voted_helpfulness,
                               found_helpful_percentage=found_helpful_percentage,
                               num_found_funny=num_found_funny,
                               steam_id_number=steam_id_number,
                               profile_url=profile_url,
                               orig_url=url)

            # Follow links to profile and review pages and collect data
            # from there
            sleep(wait)
            try:
                review_page = rget(review_dict['review_url'],
                                   timeout=(0.1, time_out))
            except ConnectionError:
                logdebug('ConnectionError encountered. Will try to wait for a'
                         ' bit before trying to reconnect again.')
                sleep(1000)
                try:
                    review_page = rget(review_dict['review_url'],
                                       timeout=(0.1, time_out))
                except ConnectionError:
                    error_msg = ('Encountered second ConnectionError in a '
                                 'row. Gracefully exiting program. Here are '
                                 'the details for continuing:\nReviews count '
                                 '= {0}\nRange start = {1}\nIndex = {2}\nURL '
                                 '= {3}\nReview URL = {4}'
                                 .format(reviews_count, range_begin, i, url,
                                         review_dict['review_url']))
                    raise ValueError(error_msg)
            sleep(wait)
            try:
                profile_page = rget(review_dict['profile_url'],
                                    timeout=(0.1, time_out))
            except ConnectionError:
                logdebug('ConnectionError encountered. Will try to wait for a'
                         ' bit before trying to reconnect again.')
                sleep(1000)
                try:
                    profile_page = rget(review_dict['profile_url'],
                                        timeout=(0.1, time_out))
                except ConnectionError:
                    error_msg = ('Encountered second ConnectionError in a '
                                 'row. Gracefully exiting program. Here are '
                                 'the details for continuing:\nReviews count '
                                 '= {0}\nRange start = {1}\nIndex = {2}\nURL '
                                 '= {3}\nProfile URL = {4}'
                                 .format(reviews_count, range_begin, i, url,
                                         review_dict['profile_url']))
                    raise ValueError(error_msg)
            review_page_html = review_page.text.strip()
            profile_page_html = profile_page.text.strip()

            # Preprocess HTML and try to decode the HTML to unicode and
            # then re-encode the text with ASCII, ignoring any
            # characters that can't be represented with ASCII
            review_page_html = space_sub(r' ', breaks_sub(r' ', review_page_html))
            review_page_html = (UnicodeDammit(review_page_html, codecs)
                                .unicode_markup.encode('ascii', 'ignore')).strip()
            profile_page_html = space_sub(r' ', breaks_sub(r' ', profile_page_html))
            profile_page_html = (UnicodeDammit(profile_page_html, codecs)
                                 .unicode_markup.encode('ascii', 'ignore')).strip()

            # Now use BeautifulSoup to parse the HTML
            review_soup = BeautifulSoup(review_page_html, 'lxml')
            profile_soup = BeautifulSoup(profile_page_html, 'lxml')
            _find = profile_soup.find

            """
            Get some information about the review, such as when it was
            posted, what the rating summary is (i.e., is the game
            recommended or not?), the number of hours the reviewer has
            played the game, and the number of hours the reviewer has
            played the game in the previous two weeks.
            """
            rating_summary_block = review_soup.find('div', 'ratingSummaryBlock')
            try:
                rating_summary_block_list = list(rating_summary_block)
                # Get rating
                review_dict['rating'] = rating_summary_block_list[3].string.strip()

                # Get date posted string, which will include an
                # original posted date and maybe an updated date as
                # well (if the post was updated)
                date_updated = None
                date_str_orig = rating_summary_block_list[7].string.strip()
                date_strs = space_sub(' ', date_str_orig[8:]).split(' Updated: ')
                if len(date_strs) == 1:
                    date_str = date_strs[0]
                    date, time = tuple([x.strip() for x in date_str.split('@')])
                    time = time.upper()
                    if date_end_with_year_string_search(date):
                        date_posted = '{0}, {1}'.format(date, time)
                    else:
                        date_posted = '{0}, 2015, {1}'.format(date, time)
                elif len(date_strs) == 2:

                    # Get original posted date
                    date_str = date_strs[0]
                    date_orig, time_orig = tuple([x.strip() for x
                                                  in date_str.split('@')])
                    time_orig = time_orig.upper()
                    if date_end_with_year_string_search(date_orig):
                        date_posted = '{0}, {1}'.format(date_orig, time_orig)
                    else:
                        date_posted = '{0}, 2015, {1}'.format(date_orig, time_orig)

                    # Get date when review was updated
                    date_str_updated = date_strs[1]
                    date_updated, time_updated = \
                        tuple([x.strip() for x in date_str_updated.split('@')])
                    time_updated = time_updated.upper()
                    if date_end_with_year_string_search(date_updated):
                        date_updated = '{0}, {1}'.format(date_updated, time_updated)
                    else:
                        date_updated = '{0}, 2015, {1}'.format(date_updated,
                                                               time_updated)
                else:
                    logerr('Found date posted string with unexpected format: '
                           '{0}\nReview URL: {1}\nContinuing on to next review.'
                           .format(date_str_orig, review_dict['review_url']))
                review_dict['date_posted'] = date_posted
                review_dict['date_updated'] = date_updated if date_updated else None

                # Get number of hours the reviewer has played the game
                hours_str = rating_summary_block_list[5].string.strip()
                (hours_last_two_weeks_str,
                 hours_total_str) = tuple([x.strip() for x in hours_str.split('/')])
                hours_total = hours_total_str.split()[0]
                review_dict['total_game_hours'] = float(comma_sub(r'', hours_total))
                hours_last_two_weeks = hours_last_two_weeks_str.split()[0]
                review_dict['total_game_hours_last_two_weeks'] = \
                    float(comma_sub(r'', hours_last_two_weeks))
            except (AttributeError, ValueError, IndexError, TypeError) as e:
                logerr('Found unexpected ratingSummaryBlock div element in '
                       'review HTML. Review URL: {0}\nContinuing on to next '
                       'review.'.format(review_url))
                continue

            # Get the user-name from the review page
            try:
                username = (review_soup
                            .find('span', 'profile_small_header_name')
                            .string
                            .strip())
                if not username:
                    raise ValueError
                review_dict['username'] = username
            except (AttributeError, ValueError) as e:
                logerr('Could not identify username from review page.\nError '
                       'output: {0}\nReview URL: {1}\nContinuing on to next '
                       'review.'.format(str(e), review_dict['review_url']))
                continue

            # Get the number of comments users made on the review (if
            # any)
            try:
                comment_match_1 = comment_re_1_search(review_page.text.strip())
                num_comments = 0
                comment_match_2 = None
                if comment_match_1:
                    comment_match_2 = comment_re_2_search(comment_match_1.group())
                if comment_match_2:
                    num_comments = comment_match_2.groups()[0].strip()
                if num_comments:
                    review_dict['num_comments'] = int(comma_sub(r'', num_comments))
                else:
                    review_dict['num_comments'] = 0
            except (AttributeError, ValueError, IndexError, TypeError) as e:
                logerr('Could not determine the number of comments.\nError '
                       'output: {0}\nReview URL: {1}\nContinuing on to next '
                       'review.'.format(str(e), review_dict['review_url']))
                continue

            # Get the reviewer's "level" (friend player level)
            try:
                friend_player_level = _find('div', 'friendPlayerLevel')
                if friend_player_level:
                    review_dict['friend_player_level'] = \
                        int(comma_sub(r'', friend_player_level.string.strip()))
                else:
                    review_dict['friend_player_level'] = None
            except (AttributeError, ValueError, IndexError, TypeError) as e:
                try:
                    logerr('Got unexpected value for the friendPlayerLevel '
                           'div element: {0}\nError output: {1}\nProfile URL:'
                           ' {2}\nContinuing on to next review.'
                           .format(_find('div', 'friendPlayerLevel').string,
                                   str(e), review_dict['profile_url']))
                except AttributeError:
                    logerr('Got unexpected value for the friendPlayerLevel '
                           'div element: {0}\nProfile URL: {1}\nContinuing on'
                           ' to next review.'
                           .format(_find('div', 'friendPlayerLevel'),
                                   review_dict['profile_url']))
                continue

            # Get the game achievements summary data
            try:
                achievements = _find('span', 'game_info_achievement_summary')
                if achievements:
                    achievements_list = \
                        list(achievements.stripped_strings)[1].split()
                    attained = comma_sub(r'', achievements_list[0].strip())
                    possible = comma_sub(r'', achievements_list[2].strip())
                    review_dict['achievement_progress'] = \
                            dict(num_achievements_attained=int(attained),
                                 num_achievements_possible=int(possible),
                                 num_achievements_percentage=(float(attained)/
                                                              float(possible)))
                else:
                    review_dict['achievement_progress'] = \
                            dict(num_achievements_attained=None,
                                 num_achievements_possible=None,
                                 num_achievements_percentage=None)
            except (AttributeError, ValueError, IndexError, TypeError) as e:
                try:
                    logerr('Got unexpected value for the '
                           'game_info_achievement_summary div element: {0}\n'
                           'Error output: {1}\nProfile URL: {2}\nContinuing '
                           'on to next review.'
                           .format(list(_find('div',
                                              'game_info_achievement_summary')
                                        .stripped_strings),
                                   str(e), review_dict['profile_url']))
                except AttributeError:
                    logerr('Got unexpected value for the '
                           'game_info_achievement_summary div element: {0}\n'
                           'Profile URL: {1}\nContinuing on to next review.'
                           .format(_find('div', 'game_info_achievement_summary'),
                                   review_dict['profile_url']))
                continue

            # Get the number of badges the reviewer has earned on the
            # site
            try:
                badges = _find('div', 'profile_badges')
                if badges:
                    review_dict['num_badges'] = \
                        int(comma_sub(r'', list(badges.stripped_strings)[1].strip()))
                else:
                    review_dict['num_badges'] = None
            except (AttributeError, ValueError, IndexError, TypeError) as e:
                try:
                    logerr('Got unexpected value for the profile_badges div '
                           'element: {0}\nError output: {1}\nProfile URL: {2}'
                           '\n\nContinuing on to next review.'
                           .format(list(_find('div', 'profile_badges')
                                        .stripped_strings),
                                   str(e), review_dict['profile_url']))
                except AttributeError:
                    logerr('Got unexpected value for the profile_badges div '
                           'element: {0}\nProfile URL: {1}\nContinuing on to '
                           'next review.'
                           .format(_find('div', 'profile_badges'),
                                   review_dict['profile_url']))
                continue

            # Get the number of reviews the reviewer has written,
            # screenshots he/she has taken, guides written, etc.
            try:
                profile_items = _find('div', 'profile_item_links')
                if profile_items:

                    # Try to parse stripped_strings by getting each
                    # pair of sequential strings
                    profile_items_strings_iter = \
                        iter(profile_items.stripped_strings)
                    profile_items_strings_dict = \
                        dict(zip(profile_items_strings_iter,
                                 profile_items_strings_iter))
                    profile_items_get = profile_items_strings_dict.get

                    # Get the number of games the reviewer owns if it
                    # exists
                    review_dict['num_games_owned'] = \
                        int(comma_sub(r'', profile_items_get('Games', '0')))

                    # Get the number of screenshots if it exists
                    review_dict['num_screenshots'] = \
                        int(comma_sub(r'', profile_items_get('Screenshots', '0')))

                    # Get the number of reviews if it exists
                    review_dict['num_reviews'] = \
                        int(comma_sub(r'', profile_items_get('Reviews', '0')))

                    # Get the number of guides if it exists
                    review_dict['num_guides'] = \
                        int(comma_sub(r'', profile_items_get('Guides', '0')))

                    # Get the number of workship items if it exists
                    review_dict['num_workshop_items'] = \
                        int(comma_sub(r'', profile_items_get('Workshop Items', '0')))
                else:
                    review_dict['num_games_owned'] = 0
                    review_dict['num_screenshots'] = 0
                    review_dict['num_reviews'] = 0
                    review_dict['num_guides'] = 0
                    review_dict['num_workshop_items'] = 0
            except (AttributeError, ValueError, IndexError, TypeError) as e:
                try:
                    logerr('Got unexpected value for the profile_item_links '
                           'div element: {0}\nError output: {1}\nProfile URL:'
                           ' {2}\nContinuing on to next review.'
                           .format(list(_find('div', 'profile_item_links')
                                        .stripped_strings),
                                   str(e), review_dict['profile_url']))
                except AttributeError:
                    logerr('Got unexpected value for the profile_item_links '
                           'div element: {0}\nProfile URL: {1}\nContinuing on'
                           ' to next review.'
                           .format(_find('div', 'profile_item_links'),
                                   review_dict['profile_url']))
                continue

            # Get the number of groups the reviewer is part of on the
            # site
            try:
                groups = _find('div', 'profile_group_links')
                if groups:
                    review_dict['num_groups'] = \
                        int(comma_sub(r'', list(groups.stripped_strings)[1].strip()))
                else:
                    review_dict['num_groups'] = None
            except (AttributeError, ValueError, IndexError, TypeError) as e:
                try:
                    logerr('Got unexpected value for the profile_group_links '
                           'div element: {0}\nError output: {1}\nProfile URL: '
                           '{2}\nContinuing on to next review.'
                           .format(list(_find('div', 'profile_group_links')
                                        .stripped_strings),
                                   str(e), review_dict['profile_url']))
                except AttributeError:
                    logerr('Got unexpected value for the profile_group_links '
                           'div element: {0}\nProfile URL: {1}\nContinuing on'
                           ' to next review.'
                           .format(_find('div', 'profile_group_links'),
                                   review_dict['profile_url']))
                continue

            # Get the number of friends the reviwer has on the site
            try:
                friends = _find('div', 'profile_friend_links')
                if friends:
                    review_dict['num_friends'] = \
                        int(comma_sub(r'', list(friends.stripped_strings)[1].strip()))
                else:
                    review_dict['num_friends'] = None
            except (AttributeError, ValueError, IndexError, TypeError) as e:
                try:
                    logerr('Got unexpected value for the profile_friend_links'
                           ' div element: {0}\nError output: {1}\nProfile '
                           'URL: {2}\nContinuing on to next review.'
                           .format(list(_find('div', 'profile_friend_links')
                                        .stripped_strings),
                                   str(e), review_dict['profile_url']))
                except AttributeError:
                    logerr('Got unexpected value for the profile_friend_links'
                           ' div element: {0}\nProfile URL: {1}\nContinuing '
                           'on to next review.'
                           .format(_find('div', 'profile_friend_links'),
                                   review_dict['profile_url']))
                continue
            yield review_dict

            reviews_count += 1
            if reviews_count == limit:
                break
        if reviews_count == limit:
            break

        # Increment the range_begin and i variables, which will be used
        # in the generation of the next page of reviews
        range_begin += 10
        i += 1


def get_and_describe_dataset(file_path: str, report: bool = True,
                             reports_dir: str = None) -> dict:
    """
    Return list of review dictionaries; also produce a report with some
    descriptive statistics and graphs.

    :param file_path: path to game reviews .jsonlines file
    :type file_path: str
    :param report: make a report describing the data-set (defaults to
                   True)
    :type report: bool
    :param reports_dir: path to directory where reports should be
                        stored
    :type reports_dir: str

    :returns: list of review dictionaries
    :rtype: list
    """

    # Get list of review dictionaries
    reviews = [loads(json_line) for json_line in open(file_path)]

    if report:

        # Get path to reports directory and open report file
        reports_dir = (reports_dir if reports_dir
                       else join(dirname(dirname(realpath(__file__))), 'reports'))
        game = splitext(basename(file_path))[0]
        output_path = join(reports_dir, '{0}_report.txt'.format(game))
        output = open(output_path, 'w')

        # Initialize seaborn-related stuff
        sns.set_palette('deep', desat=.6)
        sns.set_context(rc={'figure.figsize': (14, 7)})

        # Write header of report
        output.write('Descriptive Report for {0}\n==========================='
                     '====================================================\n'
                     '\n'.format(underscore_sub(r' ', game)))
        output.write('Number of English-language reviews: {}\n\n'
                     .format(len(reviews)))

        # Write length distribution information to report
        lengths = np.array([len(review['review']) for review in reviews])
        output.write('Review Lengths Distribution\n\n')
        output.write('Average review length: {0}\n'.format(lengths.mean()))
        output.write('Minimum review length = {0}\n'.format(min(lengths)))
        output.write('Maximum review length = {0}\n'.format(max(lengths)))
        output.write('Standard deviation = {0}\n\n\n'.format(lengths.std()))

        # Generate length histogram
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(pd.Series(lengths))
        ax.set_label(game)
        ax.set_xlabel('Review length (in characters)')
        ax.set_ylabel('Total reviews')
        fig.savefig(join(reports_dir, '{0}_length_histogram'.format(game)))
        plt.close(fig)

        # Write hours played distribution information to report
        hours = np.array([review['total_game_hours'] for review in reviews])
        output.write('Review Experience Distribution\n\n')
        output.write('Average game experience (in hours played): {0}\n'
                     .format(hours.mean()))
        output.write('Minimum experience = {0}\n'.format(min(hours)))
        output.write('Maximum experience = {0}\n'.format(max(hours)))
        output.write('Standard deviation = {0}\n\n\n'.format(hours.std()))

        # Generate experience histogram
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(pd.Series(hours))
        ax.set_label(game)
        ax.set_xlabel('Game experience (in hours played)')
        ax.set_ylabel('Total reviews')
        fig.savefig(join(reports_dir, '{0}_experience_histogram'.format(game)))
        plt.close(fig)
        output.close()

    return [r for r in reviews if type(r['total_game_hours']) in [float, int]]


def get_bin_ranges(float _min,
                   float _max,
                   int nbins=5,
                   float factor=1.0) -> list:
    """
    Return list of floating point number ranges (in increments of 0.1)
    that correspond to each bin in the distribution.

    If the bin sizes should be weighted so that they become larger as
    they get toward the end of the scale, specify a factor.

    This function is hacky and needs to be fixed, but it is a more
    difficult problem than it might at first appear.

    :param _min: minimum value of the distribution
    :type _min: float
    :param _max: maximum value of the distribution
    :type _max: float
    :param nbins: number of bins into which the distribution is being
                  sub-divided (default: 5)
    :type nbins: int
    :param factor: factor by which to multiply the bin sizes, must be
                   positive, non-zero value (default: 1.0)
    :type factor: float

    :returns: list of tuples representing the minimum and maximum
              values of each bin
    :rtype: list

    :raises ValueError: if `factor` is not a non-zero, positive value,
                        bin range validation fails, or there is an
                        overflow problem
    """

    if factor <= 0.0:
        raise ValueError('"factor" must be positive, non-zero value.')

    # Test if `_min` and `_max` are equal or if `_min` is greater
    equal = False
    try:
        np.testing.assert_almost_equal(abs(_min - _max), 0.0, decimal=1)
        equal = True
    except AssertionError:
        pass
    if equal or _min > _max:
        raise ValueError('Either "_min" is greater than "_max" or they are '
                         'equal.')

    """
    Make a list of range units, one for each bin.

    For example, `_min` is 0.0 and `_max` is 5.0 and `nbins` and
    `factor` are 5 and 1.0, then `range_parts` will simply be a list of
    1.0 values, i.e., each bin will be equal to one part of the range,
    or 1/5th in this case. If `factor` is 1.5, however, `range_parts`
    will then be [1.0, 1.5, 2.25, 3.375, 5.0625] and the first bin
    would be equal to the range divided by the sum of `range_parts`, or
    1/13th of the range, while the last bin would be equal to about
    5/13ths of the range.
    """
    cdef float i = 1.0
    range_parts = [i]
    for _ in list(range(nbins))[1:]:
        i *= factor
        range_parts.append(i)

    # Generate a list of range tuples
    _max = round(_max, 1)
    cdef float range_unit = round(_max - _min)/sum(range_parts)
    bin_ranges = []
    a, b = _min, None
    first_time = True
    for range_part in range_parts:
        current_range = range_unit*range_part
        if first_time:
            b = round(a + current_range, 1)
            first_time = False
        else:
            b = round(a + current_range - 0.1, 1)
        _bin = a if a <= _max else _max, b if b <= _max else _max

        # Try to detect overflow issues
        for val in _bin:
            if val == val + 0.1:
                raise ValueError('Bin values are too big: {0} == {0} + 0.1'
                                 .format(repr(val)))

        bin_ranges.append(_bin)
        a = round(b + 0.1, 1)

    # Ensure that the end value of the last bin is actually the given
    # `_max`
    bin_ranges[-1] = (bin_ranges[-1][0], _max)

    try:
        validate_bin_ranges(bin_ranges)
    except ValueError as e:
        error_msg = ('"bin_ranges" could not be validated: {0}'
                     .format(repr(bin_ranges)))
        logerr(error_msg)
        raise e

    return bin_ranges


def get_bin_ranges_helper(db: collection,
                          games: list,
                          label: str,
                          int nbins,
                          float factor=1.0,
                          lognormal: bool = False,
                          power_transform: float = None) -> list:
    """
    Get bin ranges given a set of games, a label, the desired number of
    bins, and the factor by which the bin sizes will be multiplied as
    the index of the bins increase.

    A set of raw data transformations can be specified as well.
    `lognormal` can be set to True to transform raw values with the
    natural log and `power_transform` can be specified as a positive,
    non-zero float value to transform raw values such that
    `x**power_transform` is used.

    :param db: MongoDB collection
    :type db: collection
    :param games: list of games
    :type games: list
    :param label: prediction label
    :type label: str
    :param nbins: number of bins into which the distribution is being
                  sub-divided
    :type nbins: int
    :param factor: factor by which to multiply the bin sizes (default:
                   1.0)
    :type factor: float
    :param lognormal: transform raw label values using `ln` (default:
                      False)
    :type lognormal: bool
    :param power_transform: power by which to transform raw label
                            values (default: None)
    :type power_transform: float or None

    :returns: list of tuples representing the minimum and maximum
              values of each bin or None if `nbins` is 0
    :rtype: list

    :raises ValueError: if `nbins` is not greater than 1 or both
                        `lognormal` and `power_transform` were
                        specified
    """

    # Return None if `nbins` is 0
    if not nbins or nbins < 2:
        raise ValueError('"nbins" must be positive integer greater than 1.')

    # Validate transformer parameters
    if lognormal and power_transform:
        raise ValueError('Both "lognormal" and "power_transform" were '
                         'specified simultaneously.')

    # Get label values
    values = np.array(get_label_values(db,
                                       games,
                                       label,
                                       lognormal=lognormal,
                                       power_transform=power_transform))

    # Divide up the distribution of label values
    _min = np.floor(values.min())
    _max = np.ceil(values.max())
    try:
        bin_ranges = get_bin_ranges(_min,
                                    _max,
                                    nbins,
                                    factor)
    except ValueError as e:
        error_msg = ('Encountered ValueError at call to get_bin_ranges: {0}\n'
                     'Min: {1}\nMax: {2}\nN bins: {3}\nFactor: {4}'
                     .format(e, _min, _max, nbins, factor))
        logerr(error_msg)
        raise ValueError(e)

    return bin_ranges


def validate_bin_ranges(bin_ranges: list) -> bool:
    """
    Validate a list of tuples representing bins that make up a
    continuous range.

    :param bin_ranges: list of ranges that define each bin, where each
                       bin should be represented as a tuple with the
                       first value, a float that is precise to one
                       decimal place, as the lower bound and the
                       second, also a float with the same type of
                       precision, the upper bound, but both limits are
                       technically soft since label values will be
                       compared to see if they are equal at the same
                       precision and so they can end up being
                       larger/smaller and still be in a given bin;
                       the bins should also make up a continuous range
                       such that every first bin value should be
                       less than the second bin value and every bin's
                       values should be less than the succeeding bin's
                       values
    :type bin_ranges: list of tuples representing the minimum and
                      maximum values of a range of values

    :returns: None
    :rtype: None

    :raises ValueError: if validation fails
    """

    # Raise error if there's only one bin
    if len(bin_ranges) == 1:
        error_msg = 'Only one bin: {}'.format(repr(bin_ranges))
        logerr(error_msg)
        raise ValueError(error_msg)

    cdef int i
    current_value = None
    for i, bin_range in enumerate(bin_ranges):

        # Make sure that each value is a float
        for end_point in bin_range:
            if (not isinstance(end_point, float)
                or not test_float_decimal_places(repr(end_point))):
                raise ValueError('"bin_ranges" includes bins that have '
                                 'non-float values or whose values are more '
                                 'precise than one-decimal place: {0}'
                                 .format(repr(bin_ranges)))

        # Make sure that each bin's values make sense internally and in
        # terms of the succeeding bin's values
        if (bin_range[0] > bin_range[1]
            or abs(bin_range[1] - bin_range[0]) < 0.01):
            error_msg = ('Found bin range whose values are either equal '
                         'or the left-hand value is greater than the '
                         'right-hand value: {0}, {1}'
                         .format(repr(bin_range[0]), repr(bin_range[1])))
            logerr(error_msg)
            raise ValueError(error_msg)
        try:
            diff = abs(bin_ranges[i + 1][0] - bin_range[1])
        except IndexError:
            return
        try:
            np.testing.assert_almost_equal(diff, 0.1, decimal=1)
        except AssertionError:
            if diff < 0.1:
                error_msg = ('Found adjacent end-points in successive bins '
                             'that have less than a 0.1 absolute difference '
                             'between them: {0} and {1}'
                             .format(repr(bin_range[1]),
                                     repr(bin_ranges[i + 1][0])))
                logerr(error_msg)
                raise ValueError(error_msg)
            else:
                error_msg = ('Found adjacent end-points in successive bins '
                             'that do not represent a continuous range: {0} '
                             'is almost equal to {1}'
                             .format(repr(bin_range[1]),
                                     repr(bin_ranges[i + 1][0])))
                logerr(error_msg)
                raise ValueError(error_msg)


def get_bin(bin_ranges: list, float val) -> int:
    """
    Return the index of the bin range in which the value falls.

    :param bin_ranges: list of ranges that define each bin, where each
                       bin should be represented as a tuple with the
                       first value, a float that is precise to one
                       decimal place, as the lower bound and the
                       second, also a float with the same type of
                       precision, the upper bound, but both limits are
                       technically soft since label values will be
                       compared to see if they are equal at the same
                       precision and so they can end up being
                       larger/smaller and still be in a given bin;
                       the bins should also make up a continuous range
                       such that every first bin value should be
                       less than the second bin value and every bin's
                       values should be less than the succeeding bin's
                       values
    :type bin_ranges: list of tuples representing the minimum and
                      maximum values of a range of values
    :param val: value
    :type val: float

    :returns: int (-1 if val not in any of the bin ranges)
    :rtype: int
    """

    cdef int i
    for i, bin_range in enumerate(bin_ranges):
        # Test if val is almost equal to the beginning or end of the
        # range
        try:
            np.testing.assert_almost_equal(val, bin_range[0], decimal=1)
            almost_equal_begin = True
        except AssertionError:
            almost_equal_begin = False
        try:
            np.testing.assert_almost_equal(val, bin_range[1], decimal=1)
            almost_equal_end = True
        except AssertionError:
            almost_equal_end = False

        if ((val > bin_range[0] or almost_equal_begin)
            and (val < bin_range[1] or almost_equal_end)):
            return i + 1

    return -1


def get_label_values(db: collection,
                     games: list,
                     label: str,
                     lognormal: bool = False,
                     power_transform: float = None) -> list:
    """
    Get all of the values for the given label in the data for the
    given list of game(s). Optionally transform raw values with
    `np.log` if lognormal is set to True (False by default).

    A set of raw data transformations can be specified as well.
    `lognormal` can be set to True to transform raw values with the
    natural log and `power_transform` can be specified as a positive,
    non-zero float value to transform raw values such that
    `x**power_transform` is used.

    The raw values of either `num_achievements_percentage` or
    `found_helpful_percentage` labels (i.e., that have percentage
    values between 0.0 and 1.0, inclusive) will be multiplied by 100
    before doing any other transformation (if any).

    :param db: MongoDB collection
    :type db: collection
    :param games: list of games
    :type games: list
    :param label: feature label
    :type label: str
    :param lognormal: transform raw label values using `ln` (default:
                      False)
    :type lognormal: bool
    :param power_transform: power by which to transform raw label
                            values (default: None)
    :type power_transform: float or None

    :returns: list of label values that are not equal to None or an
              empty string or `lognormal` and `power_transform` were
              specified
    :rtype: list

    :raises ValueError: if `lognormal` and `power_transform` were both
                        specified
    """

    # Validate transformer parameters
    if lognormal and power_transform:
        raise ValueError('Both "lognormal" and "power_transform" were '
                         'specified simultaneously.')

    # Make a cursor across all reviews from the given set of games
    cursor = db.find({'game': {'$in': list(games)}},
                     {'_id': False, 'nlp_features': False})

    # Collect all of the values from the MongoDB collection for the
    # given label/set of games and drop NaNs
    cdef int column = 0
    cdef int axis = 1
    if label in ACHIEVEMENTS_LABELS:
        label_values = \
            [compute_label_value(doc.get('achievement_progress', {}).get(label),
                                 label,
                                 lognormal=lognormal,
                                 power_transform=power_transform)
             for doc in cursor]
    else:
        label_values = [compute_label_value(doc.get(label),
                                            label,
                                            lognormal=lognormal,
                                            power_transform=power_transform)
                        for doc in cursor]
    return list(filter(lambda x: not x == None, label_values))


def compute_label_value(value,
                        label: str,
                        lognormal: bool = False,
                        power_transform: float = None,
                        bin_ranges: list = None):
    """
    Compute the value and apply any transformations specified via
    `lognormal` or `power_transform`.

    :param value: label value (always a positive value)
    :type value: int/float/None
    :param label: feature label
    :type label: str
    :param lognormal: transform raw label values using `ln` (default:
                      False)
    :type lognormal: bool
    :param power_transform: power by which to transform raw label
                            values (default: None)
    :type power_transform: float or None
    :param bin_ranges: list of ranges that define each bin, where each
                       bin should be represented as a tuple with the
                       first value, a float that is precise to one
                       decimal place, as the lower bound and the
                       second, also a float with the same type of
                       precision, the upper bound, but both limits are
                       technically soft since label values will be
                       compared to see if they are equal at the same
                       precision and so they can end up being
                       larger/smaller and still be in a given bin;
                       the bins should also make up a continuous range
                       such that every first bin value should be
                       less than the second bin value and every bin's
                       values should be less than the succeeding bin's
                       values
    :type bin_ranges: list of tuples representing the minimum and
                      maximum values of a range of values (or None)

    :returns: the value of the label, after applying transformations
              (if any)
    :rtype: float or None

    :raises ValueError: if `value` is not positive or both `lognormal`
                        and `power_transform` were specified
    """

    # Validate transformer parameters
    if lognormal and power_transform:
        raise ValueError('Both "lognormal" and "power_transform" were '
                         'specified simultaneously.')

    # Return None if value is None
    if value == None:
        return

    value = float(value)

    # Check if value is positive
    if value < 0.0:
        raise ValueError('Received invalid "value" value: {}. "value" should '
                         'be positive.'.format(repr(value)))

    # If the label has percentage values, i.e., values between 0.0 and
    # 1.0 (inclusive), multiply the value by 100 before doing anything
    # else
    if label in LABELS_WITH_PCT_VALUES:
        value *= 100.0

    # Apply natural log transformation to values above greater than or
    # equal to 1 (positive values less than 1 are very small, so they
    # are just converted to 0)
    if lognormal:
        if value > 1.0:
            value = np.log(value)
    elif power_transform:
        value = value**power_transform

    # Convert value to bin-transformed value
    if bin_ranges:
        return get_bin(bin_ranges, value)
    else:
        return value


def write_arff_file(dest_path: str,
                    file_names: list,
                    reviews: list = None,
                    db: collection = None,
                    make_train_test: bool = False,
                    bins=False) -> None:
    """
    Write .arff file either for a list of reviews read in from a file
    or list of files or for both the training and test partitions in
    the MongoDB database.

    :param reviews: list of dicts with hours/review keys-value mappings
                    representing each data-point (defaults to None)
    :type reviews: list
    :param db: MongoDB reviews collection (None by default)
    :type db: collection
    :param dest_path: path for .arff output file
    :type dest_path: str
    :param file_names: list of extension-less game file-names
    :type file_names: list
    :param make_train_test: if True, use MongoDB collection to find
                            reviews that are from the training and test
                            partitions and make files for them instead
                            of making one big file (defaults to False)
    :type make_train_test: boolean
    :param bins: if True or a list of bin range tuples, use collapsed
                 hours played values (if `make_train_test` was also
                 True, then the pre-computed collapsed hours values
                 will be used (even if a list of ranges is passed in
                 for some reason, i.e., the bin ranges will be
                 ignored); if not, the passed-in value must be a list
                 of 2-tuples representing the floating-point number
                 ranges of the bins); if False, the original,
                 unmodified hours played values will be used (default:
                 False)
    :type bins: boolean or list

    :returns: None
    :rtype: None

    :raises ValueError: if the arguments conflict with each other, if
                        arguments are missing, or if various other
                        types of issues occur
    """

    # Make sure that the passed-in keyword arguments make sense
    if make_train_test and (reviews or not db):
        raise ValueError('The make_train_test keyword argument was set to '
                         'True and either the `db` keyword was left '
                         'unspecified or the reviews keyword was specified '
                         '(or both). If the make_train_test keyword is used, '
                         'it is expected that training/test reviews will be '
                         'retrieved from the MongoDB database rather than a '
                         'list of reviews passed in via the reviews keyword.')

    if not make_train_test and db:
        if reviews:
            logwarn('Ignoring passed-in `db` keyword value. Reason: If a '
                    'list of reviews is passed in via the reviews keyword '
                    'argument, then the `db` keyword argument should not be '
                    'used at all since it will not be needed.')
        else:
            raise ValueError('A list of review dictionaries was not '
                             'specified.')
    if bins:
        if make_train_test and type(bins) == list:
            logwarn('The write_arff_file method was called with '
                    '\'make_train_test\' set to True and \'bins\' set to a '
                    'list of bin ranges ({0}). Because the bin values in the '
                    'database were precomputed, the passed-in list of bin '
                    'ranges will be ignored.'.format(repr(bins)))
        if reviews and type(bins) == bool:
            raise ValueError('The write_arff_file method was called with a '
                             'list of review dictionaries and \'bins\' set to'
                             ' True. If the hours played values are to be '
                             'collapsed and precomputed values (as from the '
                             'database, for example) are not being used, then'
                             ' the bin ranges must be specified.')
    # ARFF file template
    ARFF_BASE = """% Generated on {}
% This ARFF file was generated with review data from the following game(s): {}
% It is useful only for trying out machine learning algorithms on the bag-of-words representation of the reviews.
@relation reviewer_experience
@attribute string_attribute string
@attribute numeric_attribute numeric

@data"""
    TIMEF = '%A, %d. %B %Y %I:%M%p'

    # Replace underscores with spaces in game names and make
    # comma-separated list of games
    _file_names = str([underscore_sub(r' ', f) for f in file_names])

    # Write ARFF file(s)
    if make_train_test:

        # Make an ARFF file for each partition
        for partition in ['training', 'test']:

            # Make empty list of lines to populate with ARFF-style
            # lines, one per review
            reviews_lines = []

            # Get reviews for the given partition from all of the games
            game_docs = db.find({'partition': partition,
                                 'game': {'$in': file_names}})
            if game_docs.count() == 0:
                raise ValueError('No matching documents were found in the '
                                 'MongoDB collection for the {0} partition '
                                 'and the following games:\n\n{1}'
                                 .format(partition, file_names))
            for game_doc in game_docs:

                # Remove single/double quotes from the reviews first...
                review = quotes_sub(r'', game_doc['review'].lower())

                # Get rid of backslashes since they only make things
                # confusing
                review = backslash_sub(r'', review)
                hours = (game_doc['total_game_hours_bin'] if bins
                         else game_doc['total_game_hours'])
                reviews_lines.append('"{0}",{1}'.format(review, hours))

            # Modify file-path by adding suffix(es)
            suffix = 'train' if partition.startswith('train') else 'test'
            replacement = '{0}.arff'.format(suffix)
            if bins:
                replacement = 'bins.{0}'.format(replacement)
            _dest_path = '{0}{1}'.format(dest_path[:-4], replacement)

            # Write to file
            with open(_dest_path, 'w') as out:
                out.write('{0}\n{1}'
                          .format(ARFF_BASE.format(strftime(TIMEF), _file_names),
                                  '\n'.join(reviews_lines)))
    else:

        if not reviews:
            raise ValueError('Empty list of reviews passed in to the '
                             'write_arff_file method.')

        # Make empty list of lines to populate with ARFF-style lines,
        # one per review
        reviews_lines = []
        for rd in reviews:

            # Remove single/double quotes from the reviews first
            review = quotes_sub(r'', rd['review'].lower())

            # Get rid of backslashes since they only make things
            # confusing
            review = backslash_sub(r'', review)
            if bins:
                hours = get_bin(bins, rd['total_game_hours'])
                if hours < 0:
                    raise ValueError('The given hours played value ({0}) was '
                                     'not found in the list of possible bin '
                                     'ranges ({1}).'
                                     .format(rd['total_game_hours'], bins))
            else:
                hours = rd['total_game_hours']
            reviews_lines.append('"{0}",{1}'.format(review, hours))

        # Modify file-path by adding partition suffix
        if bins:
            dest_path = '{0}bins.arff'.format(dest_path[:-4])

        # Write to file
        with open(dest_path, 'w') as out:
            out.write('{0}\n{1}'
                      .format(ARFF_BASE.format(strftime(TIMEF), _file_names),
                                               '\n'.join(reviews_lines)))
