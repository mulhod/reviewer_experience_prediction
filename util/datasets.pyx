'''
@author Matt Mulholland
@date 05/05/2015

Module of code that reads review data from raw text files and returns a list of files, describes the data, etc.
'''
import numpy as np
from re import sub
import pandas as pd
import seaborn as sns
from langdetect import detect
import matplotlib.pyplot as plt
from os.path import abspath, basename, dirname, realpath, join
from langdetect.lang_detect_exception import LangDetectException


def read_reviews_from_game_file(file_path):
    '''
    Get list of reviews from a single game file.

    :param file_path: path to reviews file
    :type file_path: str
    :returns: list of dicts
    '''

    reviews = []
    lines = open(abspath(file_path)).readlines()
    i = 0
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
        sns.set_palette("deep", desat=.6)
        sns.set_context(rc={"figure.figsize": (8, 4)})

    # Get list of review dictionaries
    reviews = read_reviews_from_game_file(file_path)

    if report:
        output.write('Descriptive Report for {}\n======================' \
                     '=================================================' \
                     '========\n\n'.format(sub(r'_',
                                               r' ',
                                               game)))
        output.write('Number of English-language reviews: {}\n' \
                     '\n'.format(len(reviews)))

    # Look at review lengths to figure out what should be filtered out
    lengths = np.array([len(review['review']) for review in reviews])
    mean = lengths.mean()
    std = lengths.std()
    if report:
        output.write('Review Lengths Distribution\n\n')
        output.write('Average review length: {}\n'.format(mean))
        output.write('Minimum review length = {}\n'.format(min(lengths)))
        output.write('Maximum review length = {}\n'.format(max(lengths)))
        output.write('Standard deviation = {}\n\n\n'.format(std))

    # Use the standard deviation to define the range of acceptable reviews
    # (in terms of the length only) as within 2 standard deviations of the
    # mean (but with the added caveat that the reviews be at least 50
    # characters
    MINLEN = 50 if (mean - 2.0*std) < 50 else (mean - 2.0*std)
    MAXLEN = mean + 2.0*std

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

    # Look at hours played values in the same way as above for length
    hours = np.array([review['hours'] for review in reviews])
    mean = hours.mean()
    std = hours.std()
    if report:
        output.write('Review Experience Distribution\n\n')
        output.write('Average game experience (in hours played): {}' \
                     '\n'.format(mean))
        output.write('Minimum experience = {}\n'.format(min(hours)))
        output.write('Maximum experience = {}\n'.format(max(hours)))
        output.write('Standard deviation = {}\n\n\n'.format(std))

    # Use the standard deviation to define the range of acceptable reviews
    # (in terms of experience) as within 2 standard deviations of the mean
    # (starting from zero, actually)
    MINHOURS = 0
    MAXHOURS = mean + 2.0*std

    # Write MAXLEN, MINLEN, etc. values to report
    if report:
        output.write('Filter Values\nMINLEN = {}\nMAXLEN = {}\nMINHOURS ' \
                     '= {}\nMAXHOURS = {}'.format(MINLEN,
                                                  MAXLEN,
                                                  MINHOURS,
                                                  MAXHOURS))

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

    if report:
        output.close()
    orig_total_reviews=len(reviews)
    reviews = [r for r in reviews if len(r['review']) <= MAXLEN
                                  and len(r['review']) >= MINLEN
                                  and r['hours'] <= MAXHOURS]
    return dict(reviews=reviews,
                MINLEN=MINLEN,
                MAXLEN=MAXLEN,
                MINHOURS=MINHOURS,
                MAXHOURS=MAXHOURS,
                orig_total_reviews=orig_total_reviews)