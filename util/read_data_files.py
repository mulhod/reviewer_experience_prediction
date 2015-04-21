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
        # Extract the hours value and the review text from each 2-line
        # sequence
        try:
            h = float(lines[i].split()[1].strip())
            r = lines[i + 1].split(' ', 1)[1].strip()
        except (ValueError, IndexError) as e:
            i += 2
            continue
        if not detect(r) == 'en':
            i += 2
            continue
        # Skip reviews that don't have at least 50 characters
        if len(r) <= 50:
            i += 2
            continue
        # Skip reviews with more than 1000 characters (?)
        # Insert code here
        # Skip if number of hours is greater than 5000
        if h > 5000:
            i += 2
            continue
        # Contraction rules
        # wont ==> won't
        r = re.sub(r"\bwont\b", r"won't", r, re.IGNORECASE)
        # dont ==> don't
        r = re.sub(r"\bdont\b", r"don't", r, re.IGNORECASE)
        # wasnt ==> wasn't
        r = re.sub(r"\bwasnt\b", r"wasn't", r, re.IGNORECASE)
        # werent ==> weren't
        r = re.sub(r"\bwerent\b", r"weren't", r, re.IGNORECASE)
        # Get the hang of it now?
        # Insert more contraction code here
        # Now we append the 2-key dict to the end of reviews
        reviews.append(dict(hours=h,
                            review=r))
        i += 2 # Increment i by 2 since we need to go to the next
            # 2-line couplet
    return reviews
