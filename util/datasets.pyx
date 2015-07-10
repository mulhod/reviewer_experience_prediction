'''
:author: Matt Mulholland
:date: May 5, 2015

Cython extension for functions that collect review data from Steam, read
review data from raw text files, describe review data in terms of descriptive
statistics, write ARFF format files for use with Weka, convert raw hours
played values to a scale of a given number of values, etc.
'''
import logging
from sys import exit
from re import (sub,
                compile as recompile,
                search)
from time import (strftime,
                  sleep)


def get_review_data_for_game(appid, time_out=0.5, limit=-1, wait=10):
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
    :type limit: int (default: -1, which signifies no limit)
    :param wait: amount of time to wait between requesting different pages on
                 the Steam website
    :type wait: int/float
    :yields: dictionary with keys for various pieces of data related to a
             single review, including the review itself, the number of hours
             the reviewer has played the game, etc.
    '''

    # Connect to get_review_data logger
    logger = logging.getLogger('get_review_data')
    loginfo = logger.info
    logdebug = logger.debug
    logwarn = logger.warning
    logerr = logger.error

    # Define a couple useful regular expressions
    SPACE = recompile(r'[\s]+')
    BREAKS_REGEX = recompile(r'\<br\>')
    COMMA = recompile(r',')
    DATE_END_WITH_YEAR_STRING = recompile(r', \d{4}$')
    COMMENT_RE_1 = recompile(r'<span id="commentthread[^<]+')
    COMMENT_RE_2 = recompile(r'>(\d*)$')

    # Codecs for use with UnicodeDammit
    codecs = ["windows-1252", "utf8", "ascii", "cp500", "cp850", "cp852",
              "cp858", "cp1140", "cp1250", "iso-8859-1", "iso8859_2",
              "iso8859_15", "iso8859_16", "mac_roman", "mac_latin2", "utf32",
              "utf16"]

    # Base URL for pages of reviews
    BASE_URL = 'http://steamcommunity.com/app/{2}/homecontent/?userreviewso' \
               'ffset={0}&p=1&itemspage={1}&screenshotspage={1}&videospage=' \
               '{1}&artpage={1}&allguidepage={1}&webguidepage={1}&integrate' \
               'dguidepage={1}&discussionspage={1}&appid={2}&appHubSubSecti' \
               'on=10&appHubSubSection=10&l=english&browsefilter=toprated&f' \
               'ilterLanguage=default&searchText=&forceanon=1'

    # Imports
    import requests
    from data import APPID_DICT
    from langdetect import detect
    from bs4 import (BeautifulSoup,
                     UnicodeDammit)
    from langdetect.lang_detect_exception import LangDetectException

    loginfo('Collecting review data for {} ({})...'
            .format([x[0] for x in APPID_DICT.items()
                     if x[1] == appid][0],
                    appid))
    loginfo('TIME_OUT = {} seconds'.format(time_out))
    loginfo('LIMIT = {} reviews'.format(limit))
    loginfo('SLEEP = {} seconds'.format(wait))

    cdef int reviews_count = 0
    cdef int range_begin = 0
    cdef int i = 1
    cdef int breaks = 0
    while True and breaks < 100:
        # Get unique URL for values of range_begin and i
        url = BASE_URL.format(range_begin,
                              i,
                              appid)
        loginfo('Collecting review data from the following URL: {}'
                .format(url))
        # Get the URL content
        base_page = None
        sleep(wait)
        # Get the HTML page; if there's a timeout error, then catch it and
        # exit out of the loop, effectively ending the function.
        try:
            base_page = requests.get(url,
                                     timeout=time_out)
        except requests.exceptions.Timeout as e:
            logerr('There was a Timeout error...')
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
        reviews = source_soup.find_all('div',
                                       'apphub_CardContentMain')
        link_blocks = \
            source_soup.findAll('div',
                                'apphub_Card modalContentLink interactable')

        # Iterate over the reviews/link blocks in the source HTML and find
        # data for each review, yielding a dictionary
        for (review,
             link_block) in zip(reviews,
                                link_blocks):

            # Get links to review URL, profile URL, Steam ID number
            review_url = link_block.attrs['data-modal-content-url'].strip()
            review_url_split = review_url.split('/')
            profile_url = '/'.join(review_url_split[:5]).strip()

            # Get other data within the base reviews page
            stripped_strings = list(review.stripped_strings)

            if not (review_url.startswith('http://')
                    or len(review_url_split) > 4):
                logerr('Found review with invalid review_url: {}\nThe rest of'
                       ' the review\'s contents: {}\nReview URL: {}\n'
                       'Continuing on to next review.'
                       .format(review_url,
                               stripped_strings,
                               url))
                continue
            try:
                steam_id_number = review_url_split[4]
                if not steam_id_number:
                    raise ValueError
            except ValueError as e:
                logerr('Empty steam_id_number. URL: {}\nstripped_strings: {}'
                       '\nContinuing on to next review.'
                       .format(url,
                               stripped_strings))
                continue

            # Parsing the HTML in this way (with the stripped_strings
            # attribute) depends on a length of at least 5 (one list element
            # for the following items (not all will be collected, however):
            #     1) number of people who found the review helpful/funny
            #     2) "recommended" value
            #     3) hours on record
            #     4) date posted
            #     5+) the review, which can stretch over several elements
            # All that we are really interested in collecting from here are 1
            # and 5+
            if len(stripped_strings) >= 5:
                helpful_and_funny_list = stripped_strings[0].strip().split()
                # Extracting data from the text that supplies the number of
                # users who found the review helpful and/or funny depends on a
                # couple facts about how what is in this particular string
                num_found_helpful = 0
                num_voted_helpfulness = 0
                num_found_unhelpful = 0
                found_helpful_percentage = 0
                num_found_funny = 0
                if len(helpful_and_funny_list) == 15:
                    helpful = helpful_and_funny_list[:9]
                    funny = helpful_and_funny_list[9:]
                elif (len(helpful_and_funny_list) == 9
                      or len(helpful_and_funny_list) == 6):
                    # Get the parts of the string that have to do with the
                    # review being helpful, funny
                    if len(helpful_and_funny_list) == 9:
                        helpful = helpful_and_funny_list
                        funny = None
                    else:
                        helpful = None
                        funny = helpful_and_funny_list

                    # Extract the number of people who found the review
                    # helpful
                    if helpful:
                        num_found_helpful = COMMA.sub(r'',
                                                      helpful[0])
                        try:
                            num_found_helpful = int(num_found_helpful)
                        except ValueError:
                            logerr('Could not cast num_found_helpful value to'
                                   ' an int: {}\nRest of review: {}\nURL: {}'
                                   '\nContinuing on to next review.'
                                   .format(num_found_helpful,
                                           stripped_strings,
                                           url))
                            continue

                        # Extract the number of people who voted on whether or
                        # not the review was helpful (whether it was or
                        # wasn't)
                        num_voted_helpfulness = COMMA.sub(r'',
                                                          helpful[2])
                        try:
                            num_voted_helpfulness = int(num_voted_helpfulness)
                        except ValueError:
                            logerr('Could not cast num_voted_helpfulness '
                                   'value to an int: {}\nRest of review: {}\n'
                                   'URL: {}\nContinuing on to next review.'
                                   .format(num_voted_helpfulness,
                                           stripped_strings,
                                           url))
                            continue

                        # Calculate the number of people who must have found
                        # the review NOT helpful and the percentage of people
                        # who found the review helpful
                        num_found_unhelpful = \
                            num_voted_helpfulness - num_found_helpful
                        found_helpful_percentage = \
                            float(num_found_helpful)/num_voted_helpfulness

                    # Extract the number of people who found the review funny
                    if funny:
                        num_found_funny = COMMA.sub(r'',
                                                    funny[0])
                        try:
                            num_found_funny = int(num_found_funny)
                        except ValueError:
                            logerr('Could not cast num_found_funny value to '
                                   'an int: {}\nRest of review: {}\nURL: {}\n'
                                   'Continuing on to next review.'
                                   .format(num_found_funny,
                                           stripped_strings,
                                           url))
                            continue

                else:
                    logerr('Found review with a helpful/funny string that '
                           'does not conform to the expected format: {}\nRest'
                           ' of the review\'s contents: {}\nURL: {}\n'
                           'Continuing on to next review.'
                           .format(stripped_strings[0],
                                   stripped_strings[1:],
                                   url))
                    continue

                # The review text should be located at index 4, but, in some
                # cases, the review text is split over several indices
                review_text = ' '.join(stripped_strings[4:]).strip()

                # Skip reviews that don't have any characters
                if not review_text:
                    logdebug('Found review with apparently no review text: {}'
                             '\nURL: {}\nContinuing on to next review.'
                             .format(stripped_strings,
                                     url))
                    continue

                # Skip review if it is not recognized as English
                try:
                    if not detect(review_text) == 'en':
                        continue
                except LangDetectException:
                    logdebug('Found review whose review text content was not '
                             'recognized as English: {}\nRest of review: {}\n'
                             'URL: {}\nContinuing on to next review.'
                             .format(review_text,
                                     stripped_strings,
                                     url))
                    continue
            else:
                logerr('Found incorrect number of "stripped_strings" in '
                       'review HTML element. The review\'s stripped_strings: '
                       '{}\nURL: {}\nContinuing on to next review.'
                       .format(stripped_strings,
                               url))
                continue

            # Make dictionary for holding all the data related to the
            # review
            review_dict = \
                dict(review_url=review_url,
                     review=review_text,
                     num_found_helpful=num_found_helpful,
                     num_found_unhelpful=num_found_unhelpful,
                     num_voted_helpfulness=num_voted_helpfulness,
                     found_helpful_percentage=found_helpful_percentage,
                     num_found_funny=num_found_funny,
                     steam_id_number=steam_id_number,
                     profile_url=profile_url,
                     orig_url=url)

            # Follow links to profile and review pages and collect data from
            # there
            sleep(wait)
            review_page = requests.get(review_dict['review_url'])
            sleep(wait)
            profile_page = requests.get(review_dict['profile_url'])
            review_page_html = review_page.text.strip()
            profile_page_html = profile_page.text.strip()

            # Preprocess HTML and try to decode the HTML to unicode and then
            # re-encode the text with ASCII, ignoring any characters that
            # can't be represented with ASCII
            review_page_html = \
                SPACE.sub(r' ',
                          BREAKS_REGEX.sub(r' ',
                                           review_page_html))
            review_page_html = \
                UnicodeDammit(review_page_html,
                              codecs).unicode_markup.encode('ascii',
                                                            'ignore').strip()
            profile_page_html = \
                SPACE.sub(r' ',
                          BREAKS_REGEX.sub(r' ',
                                           profile_page_html))
            profile_page_html = \
                UnicodeDammit(profile_page_html,
                              codecs).unicode_markup.encode('ascii',
                                                            'ignore').strip()

            # Now use BeautifulSoup to parse the HTML
            review_soup = BeautifulSoup(review_page_html,
                                        'lxml')
            profile_soup = BeautifulSoup(profile_page_html,
                                         'lxml')

            # Get some information about the review, such as when it was
            # posted, what the rating summary is (i.e., is the game
            # recommended or not?), the number of hours the reviewer has
            # played the game, and the number of hours the reviewer has
            # played the game in the previous 2 weeks
            rating_summary_block = (review_soup
                                    .find('div',
                                          'ratingSummaryBlock'))
            try:
                rating_summary_block_list = list(rating_summary_block)

                # Get rating
                review_dict['rating'] = (rating_summary_block_list[3]
                                         .string
                                         .strip())

                # Get date posted string, which will include an original
                # posted date and maybe an updated date as well (if the post
                # was updated)
                date_updated = None
                date_str_orig = (rating_summary_block_list[7]
                                 .string
                                 .strip())
                date_strs = \
                    SPACE.sub(' ',
                              date_str_orig[8:]).split(' Updated: ')
                if len(date_strs) == 1:
                    date_str = date_strs[0]
                    date, time = tuple([x.strip() for x
                                        in date_str.split('@')])
                    time = time.upper()
                    if DATE_END_WITH_YEAR_STRING.search(date):
                        date_posted = '{}, {}'.format(date,
                                                      time)
                    else:
                        date_posted = '{}, 2015, {}'.format(date,
                                                            time)
                elif len(date_strs) == 2:
                    # Get original posted date
                    date_str = date_strs[0]
                    date_orig, time_orig = \
                        tuple([x.strip() for x
                               in date_str.split('@')])
                    time_orig = time_orig.upper()
                    if DATE_END_WITH_YEAR_STRING.search(date_orig):
                        date_posted = '{}, {}'.format(date_orig,
                                                      time_orig)
                    else:
                        date_posted = '{}, 2015, {}'.format(date_orig,
                                                            time_orig)

                    # Get date when review was updated
                    date_str_updated = date_strs[1]
                    date_updated, time_updated = \
                        tuple([x.strip() for x
                               in date_str_updated.split('@')])
                    time_updated = time_updated.upper()
                    if DATE_END_WITH_YEAR_STRING.search(date_updated):
                        date_updated = '{}, {}'.format(date_updated,
                                                       time_updated)
                    else:
                        date_updated = '{}, 2015, {}'.format(date_updated,
                                                             time_updated)
                else:
                    logerr('Found date posted string with unexpected format: '
                           '{}\nReview URL: {}\nContinuing on to next review.'
                           .format(date_str_orig,
                                   review_dict['review_url']))
                review_dict['date_posted'] = date_posted
                review_dict['date_updated'] = (date_updated if date_updated
                                                            else None)

                # Get number of hours the reviewer has played the game
                hours_str = rating_summary_block_list[5].string.strip()
                (hours_last_two_weeks_str,
                 hours_total_str) = tuple([x.strip() for x
                                           in hours_str.split('/')])
                hours_total = hours_total_str.split()[0]
                review_dict['total_game_hours'] = \
                    float(COMMA.sub(r'',
                                    hours_total))
                hours_last_two_weeks = hours_last_two_weeks_str.split()[0]
                review_dict['total_game_hours_last_two_weeks'] = \
                        float(COMMA.sub(r'',
                                        hours_last_two_weeks))

            except (AttributeError,
                    ValueError,
                    IndexError,
                    TypeError) as e:
                logerr('Found unexpected ratingSummaryBlock div element in '
                       'review HTML. Review URL: {}\nContinuing on to next '
                       'review.'.format(review_url))
                continue

            # Get the user-name from the review page
            try:
                username = (review_soup
                            .find('span',
                                  'profile_small_header_name')
                            .string
                            .strip())
                if not username:
                    raise ValueError
                review_dict['username'] = username
            except (AttributeError,
                    ValueError) as e:
                logerr('Could not identify username from review page.\nError '
                       'output: {}\nReview URL: {}\nContinuing on to next '
                       'review.'
                       .format(str(e),
                               review_dict['review_url']))
                continue

            # Get the number of comments users made on the review (if any)
            try:
                comment_match_1 = COMMENT_RE_1.search(review_page
                                                      .text
                                                      .strip())
                num_comments = 0
                comment_match_2 = None
                if comment_match_1:
                    comment_match_2 = COMMENT_RE_2.search(comment_match_1
                                                          .group())
                if comment_match_2:
                    num_comments = comment_match_2.groups()[0].strip()
                if num_comments:
                    review_dict['num_comments'] = \
                        int(COMMA.sub(r'',
                                      num_comments))
                else:
                    review_dict['num_comments'] = 0
            except (AttributeError,
                    ValueError,
                    IndexError,
                    TypeError) as e:
                logerr('Could not determine the number of comments.\nError '
                       'output: {}\nReview URL: {}\nContinuing on to next '
                       'review.'.format(str(e),
                                        review_dict['review_url']))
                continue

            # Get the reviewer's "level" (friend player level)
            try:
                friend_player_level = profile_soup.find('div',
                                                        'friendPlayerLevel')
                if friend_player_level:
                    review_dict['friend_player_level'] = \
                        int(COMMA.sub(r'',
                                      friend_player_level
                                      .string
                                      .strip()))
                else:
                    review_dict['friend_player_level'] = None
            except (AttributeError,
                    ValueError,
                    IndexError,
                    TypeError) as e:
                try:
                    logerr('Got unexpected value for the friendPlayerLevel '
                           'div element: {}\nError output: {}\nProfile URL: '
                           '{}\nContinuing on to next review.'
                           .format(profile_soup
                                   .find('div',
                                         'friendPlayerLevel')
                                   .string,
                                   str(e),
                                   review_dict['profile_url']))
                except AttributeError:
                    logerr('Got unexpected value for the friendPlayerLevel '
                           'div element: {}\nProfile URL: {}\nContinuing on '
                           'to next review.'
                           .format(profile_soup
                                   .find('div',
                                         'friendPlayerLevel'),
                                   review_dict['profile_url']))
                continue

            # Get the game achievements summary data
            try:
                achievements = \
                    profile_soup.find('span',
                                      'game_info_achievement_summary')
                if achievements:
                    achievements_list = list(achievements
                                             .stripped_strings)[1].split()
                    attained = COMMA.sub(r'',
                                         achievements_list[0].strip())
                    possible = COMMA.sub(r'',
                                         achievements_list[2].strip())
                    review_dict['achievement_progress'] = \
                            dict(
                        num_achievements_attained=int(attained),
                        num_achievements_possible=int(possible),
                        num_achievements_percentage=(float(attained)
                                                     /float(possible)))
                else:
                    review_dict['achievement_progress'] = \
                            dict(num_achievements_attained=None,
                                 num_achievements_possible=None,
                                 num_achievements_percentage=None)
            except (AttributeError,
                    ValueError,
                    IndexError,
                    TypeError) as e:
                try:
                    logerr('Got unexpected value for the '
                           'game_info_achievement_summary div element: {}\n'
                           'Error output: {}\nProfile URL: {}\nContinuing on '
                           'to next review.'
                           .format(list(profile_soup
                                        .find('div',
                                              'game_info_achievement_summary')
                                        .stripped_strings),
                                   str(e),
                                   review_dict['profile_url']))
                except AttributeError:
                    logerr('Got unexpected value for the '
                           'game_info_achievement_summary div element: {}\n'
                           'Profile URL: {}\nContinuing on to next review.'
                           .format(profile_soup
                                   .find('div',
                                         'game_info_achievement_summary'),
                                   review_dict['profile_url']))
                continue

            # Get the number of badges the reviewer has earned on the site
            try:
                badges = profile_soup.find('div',
                                           'profile_badges')
                if badges:
                    review_dict['num_badges'] = \
                        int(COMMA.sub(r'',
                                      list(badges
                                           .stripped_strings)[1]
                                      .strip()))
                else:
                    review_dict['num_badges'] = None
            except (AttributeError,
                    ValueError,
                    IndexError,
                    TypeError) as e:
                try:
                    logerr('Got unexpected value for the profile_badges div '
                           'element: {}\nError output: {}\nProfile URL: {}\n'
                           '\nContinuing on to next review.'
                           .format(list(profile_soup
                                        .find('div',
                                              'profile_badges')
                                        .stripped_strings),
                                   str(e),
                                   review_dict['profile_url']))
                except AttributeError:
                    logerr('Got unexpected value for the profile_badges div '
                           'element: {}\nProfile URL: {}\nContinuing on to '
                           'next review.'
                           .format(profile_soup
                                   .find('div',
                                         'profile_badges'),
                                   review_dict['profile_url']))
                continue

            # Get the number of reviews the reviewer has written, screenshots
            # he/she has taken, guides written, etc.
            try:
                profile_items = profile_soup.find('div',
                                                  'profile_item_links')
                if profile_items:
                    # Try to parse stripped_strings by getting each pair of
                    # sequential strings
                    profile_items_strings_iter = iter(profile_items
                                                      .stripped_strings)
                    profile_items_strings_dict = \
                        dict(zip(profile_items_strings_iter,
                                 profile_items_strings_iter))

                    # Get the number of games the reviewer owns if it exists
                    review_dict['num_games_owned'] = \
                        int(COMMA.sub(r'',
                                      profile_items_strings_dict
                                      .get('Games',
                                           '0')))

                    # Get the number of screenshots if it exists
                    review_dict['num_screenshots'] = \
                        int(COMMA.sub(r'',
                                      profile_items_strings_dict
                                      .get('Screenshots',
                                           '0')))

                    # Get the number of reviews if it exists
                    review_dict['num_reviews'] = \
                        int(COMMA.sub(r'',
                                      profile_items_strings_dict
                                      .get('Reviews',
                                           '0')))

                    # Get the number of guides if it exists
                    review_dict['num_guides'] = \
                        int(COMMA.sub(r'',
                                      profile_items_strings_dict
                                      .get('Guides',
                                           '0')))

                    # Get the number of workship items if it exists
                    review_dict['num_workshop_items'] = \
                        int(COMMA.sub(r'',
                                      profile_items_strings_dict
                                      .get('Workshop Items',
                                           '0')))
                else:
                    review_dict['num_games_owned'] = 0
                    review_dict['num_screenshots'] = 0
                    review_dict['num_reviews'] = 0
                    review_dict['num_guides'] = 0
                    review_dict['num_workshop_items'] = 0
            except (AttributeError,
                    ValueError,
                    IndexError,
                    TypeError) as e:
                try:
                    logerr('Got unexpected value for the profile_item_links '
                           'div element: {}\nError output: {}\nProfile URL: '
                           '{}\nContinuing on to next review.'
                           .format(list(profile_soup
                                        .find('div',
                                              'profile_item_links')
                                        .stripped_strings),
                                   str(e),
                                   review_dict['profile_url']))
                except AttributeError:
                    logerr('Got unexpected value for the profile_item_links '
                           'div element: {}\nProfile URL: {}\nContinuing on'
                           ' to next review.'
                           .format(profile_soup
                                   .find('div',
                                         'profile_item_links'),
                                   review_dict['profile_url']))
                continue

            # Get the number of groups the reviewer is part of on the site
            try:
                groups = profile_soup.find('div',
                                           'profile_group_links')
                if groups:
                    review_dict['num_groups'] = \
                        int(COMMA.sub(r'',
                                      list(groups
                                           .stripped_strings)[1]
                                      .strip()))
                else:
                    review_dict['num_groups'] = None
            except (AttributeError,
                    ValueError,
                    IndexError,
                    TypeError) as e:
                try:
                    logerr('Got unexpected value for the profile_group_links '
                           'div element: {}\nError output: {}\nProfile URL: '
                           '{}\nContinuing on to next review.'
                           .format(list(profile_soup
                                        .find('div',
                                              'profile_group_links')
                                        .stripped_strings),
                                   str(e),
                                   review_dict['profile_url']))
                except AttributeError:
                    logerr('Got unexpected value for the profile_group_links '
                           'div element: {}\nProfile URL: {}\nContinuing on '
                           'to next review.'
                           .format(profile_soup
                                   .find('div',
                                         'profile_group_links'),
                                   review_dict['profile_url']))
                continue

            # Get the number of friends the reviwer has on Steam
            try:
                friends = profile_soup.find('div',
                                            'profile_friend_links')
                if friends:
                    review_dict['num_friends'] = \
                        int(COMMA.sub(r'',
                                      list(friends
                                           .stripped_strings)[1]
                                      .strip()))
                else:
                    review_dict['num_friends'] = None
            except (AttributeError,
                    ValueError,
                    IndexError,
                    TypeError) as e:
                try:
                    logerr('Got unexpected value for the profile_friend_links'
                           ' div element: {}\nError output: {}\nProfile URL: '
                           '{}\nContinuing on to next review.'
                           .format(list(profile_soup
                                        .find('div',
                                              'profile_friend_links')
                                        .stripped_strings),
                                   str(e),
                                   review_dict['profile_url']))
                except AttributeError:
                    logerr('Got unexpected value for the profile_friend_links'
                           ' div element: {}\nProfile URL: {}\nContinuing on '
                           'to next review.'
                           .format(profile_soup
                                   .find('div',
                                         'profile_friend_links'),
                                   review_dict['profile_url']))
                continue

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


def parse_appids(appids, logger_name=None):
    '''
    Parse the command-line argument passed in with the --appids flag, exiting
    if any of the resulting IDs do not map to games in APPID_DICT.

    :param appids: game IDs
    :type appids: str
    :param logger_name: name of logger initialized in driver program (default:
                        None)
    :type logger_name: str
    :returns: list of game IDs
    '''

    if logger_name:
        logger = logging.getLogger(logger_name)

    # Import APPID_DICT, containing a mapping between the appid strings and
    # the names of the video games
    from data import APPID_DICT

    appids = appids.split(',')
    for appid in appids:
        if not appid in APPID_DICT.values():
            logger.error('{} not found in APPID_DICT. Exiting.'.format(appid))
            exit(1)
    return appids


cdef read_reviews_from_game_file(file_path):
    '''
    Generate list of review dictionaries from a single game's .jsonlines file.

    :param file_path: path to reviews file
    :type file_path: str
    :returns: list of dict
    '''

    from json import JSONDecoder
    json_decode = JSONDecoder().decode

    reviews = []
    for json_line in open(file_path).readlines():
        reviews.append(json_decode(json_line))

    return reviews


def get_and_describe_dataset(file_path, report=True):
    '''
    Return dictionary with a list of review dictionaries (filtered in terms of
    the values for maximum/minimum review length and minimum/maximum hours
    played values) and the number of original, English-language reviews
    (before filtering); also produce a report with some descriptive statistics
    and graphs.

    :param file_path: path to game reviews .jsonlines file
    :type file_path: str
    :param report: make a report describing the data-set (defaults to True)
    :type report: boolean
    :returns: dict containing a 'reviews' key mapped to the list of read-in
              review dictionaries and int values mapped to keys for MAXLEN,
              MINLEN, MAXHOURS, and MINHOURS
    '''

    # Imports
    import numpy as np
    from os.path import (basename,
                         dirname,
                         realpath,
                         join)

    if report:
        # Imports
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

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
    reviews = list(read_reviews_from_game_file(file_path))

    if report:
        # Write header of report
        output.write('Descriptive Report for {}\n============================'
                     '===================================================\n\n'
                     .format(sub(r'_',
                                 r' ',
                                 game)))
        output.write('Number of English-language reviews: {}\n\n'
                     .format(len(reviews)))

    # Look at review lengths to figure out what should be filtered out
    lengths = np.array([len(review['review']) for review in reviews])
    cdef float meanl = lengths.mean()
    cdef float stdl = lengths.std()
    if report:
        # Write length distribution information to report
        output.write('Review Lengths Distribution\n\n')
        output.write('Average review length: {}\n'.format(meanl))
        output.write('Minimum review length = {}\n'.format(min(lengths)))
        output.write('Maximum review length = {}\n'.format(max(lengths)))
        output.write('Standard deviation = {}\n\n\n'.format(stdl))

    # Use the standard deviation to define the range of acceptable reviews
    # (in terms of the length only) as within 4 standard deviations of the
    # mean (but with the added caveat that the reviews be at least 50
    # characters
    cdef float minl = 50.0 if (meanl - 4*stdl) < 50 else (meanl - 4*stdl)
    cdef float maxl = meanl + 4*stdl

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
    hours = np.array([review['total_game_hours'] for review in reviews])
    cdef float meanh = hours.mean()
    cdef float stdh = hours.std()
    if report:
        # Write hours played distribution information to report
        output.write('Review Experience Distribution\n\n')
        output.write('Average game experience (in hours played): {}'
                     '\n'.format(meanh))
        output.write('Minimum experience = {}\n'.format(min(hours)))
        output.write('Maximum experience = {}\n'.format(max(hours)))
        output.write('Standard deviation = {}\n\n\n'.format(stdh))

    # Use the standard deviation to define the range of acceptable reviews
    # (in terms of experience) as within 4 standard deviations of the mean
    # (starting from zero, actually)
    cdef float minh = 0.0
    cdef float maxh = meanh + 4*stdh

    # Write MAXLEN, MINLEN, etc. values to report
    if report:
        # Write information about the hours played filtering
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


def get_bin_ranges(float _min, float _max, int nbins=5, float factor=1.0):
    '''
    Return list of floating point number ranges (in increments of 0.1) that
    correspond to each bin in the distribution.

    If the bin sizes should be weighted so that they become larger as they get
    toward the end of the scale, specify a factor.

    :param _min: minimum value of the distribution
    :type _min: float
    :param _max: maximum value of the distribution
    :type _max: float
    :param nbins: number of bins into which the distribution is being
                  sub-divided (default: 5)
    :type nbins: int
    :param factor: factor by which to multiply the bin sizes (default: 1.0)
    :type factor: float
    :returns: list of tuples representing the minimum and maximum values of
              each bin
    '''

    # Make a list of range units, one for each bin
    # For example, if nbins = 5 and factor = 1.0, then range_parts will simply
    # be a list of 1.0 values, i.e., each bin will be equal to one part of the
    # range, or 1/5th in this case.
    # If factor = 1.5, however, range_parts will then be [1.0, 1.5, 2.25,
    # 3.375, 5.0625] and the first bin would be equal to the range divided by
    # the sum of range_parts, or 1/13th of the range, while the last bin would
    # be equal to about 5/13ths of the range.
    cdef float i = 1.0
    range_parts = [i]
    for _ in list(range(nbins))[1:]:
        i *= factor
        range_parts.append(i)

    # Generate a list of range tuples
    cdef float range_unit = round(_max - _min)/sum(range_parts)
    bin_ranges = []
    cdef float current_min = _min
    for range_part in range_parts:
        _range = (range_unit*range_part)
        bin_ranges.append((round(current_min + 0.1, 1),
                           round(current_min + _range, 1)))
        current_min += _range

    # Subtract 0.1 from the beginning of the first range tuple since 0.1 was
    # artifically added to every range beginning value to ensure that the
    # range tuples did not overlap in values
    bin_ranges[0] = (bin_ranges[0][0] - 0.1,
                     bin_ranges[0][1])

    return bin_ranges


def get_bin(bin_ranges, float val):
    '''
    Return the index of the bin range in which the value falls.

    :param bin_ranges: list of ranges that define each bin
    :type bin_ranges: list of tuples representing the minimum and maximum
                      values of a range of values
    :param val: value
    :type val: float
    :returns: int (-1 if val not in any of the bin ranges)
    '''

    cdef int i
    for i, bin_range in enumerate(bin_ranges):
        if (val >= bin_range[0]
            and val <= bin_range[1]):
            return i + 1
    return -1


def write_arff_file(dest_path, file_names, reviews=None, reviewdb=None,
                    make_train_test=False, bins=False):
    '''
    Write .arff file either for a list of reviews read in from a file or list
    of files or for both the training and test partitions in the MongoDB
    database.

    :param reviews: list of dicts with hours/review keys-value mappings
                    representing each data-point (defaults to None)
    :type reviews: list of dict
    :param reviewdb: MongoDB reviews collection
    :type reviewdb: pymongo.MongoClient object (None by default)
    :param dest_path: path for .arff output file
    :type dest_path: str
    :param file_names: list of extension-less game file-names
    :type file_names: list of str
    :param make_train_test: if True, use MongoDB collection to find reviews
                            that are from the training and test partitions and
                            make files for them instead of making one big file
                            (defaults to False)
    :type make_train_test: boolean
    :param bins: if True or a list of bin range tuples, use collapsed hours
                 played values (if make_train_test was also True, then the
                 pre-computed collapsed hours values will be used (even if a
                 list of ranges is passed in for some reason, i.e., the bin
                 ranges will be ignored); if not, the passed-in value must be
                 a list of 2-tuples representing the floating-point number
                 ranges of the bins); if False, the original, unmodified hours
                 played values will be used (default: False)
    :type bins: boolean or list of 2-tuples of floats
    :returns: None
    '''

    # Connect to logger from driver progam
    logger = logging.getLogger('make_arff_files')
    logerr = logger.error
    logwarn = logger.warning

    # Make sure that the passed-in keyword arguments make sense
    if (make_train_test
        and (reviews
             or not reviewdb)):
        logerr('The make_train_test keyword argument was set to True and '
               'either the reviewdb keyword was left unspecified or the '
               'reviews keyword was specified (or both). If the '
               'make_train_test keyword is used, it is expected that training'
               '/test reviews will be retrieved from the MongoDB database '
               'rather than a list of reviews passed in via the reviews '
               'keyword. Exiting.')
        exit(1)
    if (not make_train_test
        and reviewdb):
        if reviews:
            logwarn('Ignoring passed-in reviewdb keyword value. Reason: If a '
                    'list of reviews is passed in via the reviews keyword '
                    'argument, then the reviewdb keyword argument should not '
                    'be used at all since it will not be needed.')
        else:
            logerr('A list of review dictionaries was not specified. '
                   'Exiting.')
            exit(1)
    if bins:
        if (make_train_test
            and type(bins) == list):
            logwarn('The write_arff_file method was called with '
                    '\'make_train_test\' set to True and \'bins\' set to a '
                    'list of bin ranges ({}). Because the bin values in the '
                    'database were precomputed, the passed-in list of bin '
                    'ranges will be ignored.'.format(repr(bins)))
        if (reviews
            and type(bins) == bool):
            logerr('The write_arff_file method was called with a list of '
                   'review dictionaries and \'bins\' set to True. If the '
                   'hours played values are to be collapsed and precomputed '
                   'values (as from the database, for example) are not being '
                   'used, then the bin ranges must be specified. Exiting.')
            exit(1)

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
                logerr('No matching documents were found in the MongoDB '
                       'collection for the {} partition and the following '
                       'games:\n\n{}\n\nExiting.'.format(partition,
                                                         file_names))
                exit(1)

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
            logerr('Empty list of reviews passed in to the write_arff_file '
                   'method. Exiting.')
            exit(1)

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
                    logerr('The given hours played value ({}) was not found '
                           'in the list of possible bin ranges ({}). Exiting.'
                           .format(rd['hours'],
                                   bins))
                    exit(1)
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
