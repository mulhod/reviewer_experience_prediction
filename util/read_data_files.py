import sys
import os
from langdetect import detect
from os.path import dirname, realpath, join

# This is just a way of getting the path to this script
my_path = dirname(realpath(__file__))
# This is the path to the 'data' directory, which will store the files
# in the same format as the files in the 'test' directory
data_dir = join(my_path, 'data')

def get_reviews_for_game(file_name):
    '''
    Get list of reviews in a single game file.

    :param file_name: name of file
    :type file_name: str
    :returns: list of dicts
    '''

    reviews = []
    lines = open(join(data_dir, file_name)).readlines()
    i = 0
    while i + 1 < len(lines): # We need to get every 2-line couplet
        # Just assume that the next two lines of code correctly extract the
        # two pieces of info we want, the number of hours played and the review
        # itself
        h = float(lines[i].split()[1].strip())
        r = lines[i + 1].split(' ', 1)[1].strip()
        if not detect(r) == 'en':
            i += 2
            continue
        # Now we append the 2-key dict to the end of reviews
        reviews.append(dict(hours=h,
                            review=r))
        i += 2 # Increment i by 2 since we need to go to the next
            # 2-line couplet
    return reviews
