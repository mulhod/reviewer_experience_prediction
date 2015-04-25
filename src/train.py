'''
@author: Matt Mulholland, Janette Martinez, Emily Olshefski
@date: 3/18/15

Script used to train a model on a given data-set (or multiple data-sets combined).
'''
import sys
import re
import pymongo
import argparse
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer


class Review(object):
    '''
    Class for objects representing Reviews.
    '''

    # Attributes
    orig = None # original text
    hours_played = None # length of time played
    game_id = None # game ID (corresponds to Steam's 'appid')
    norm = None # str representing normalized review text
    tok_sents = None # list of tokenized sentences (lists of str)
    stem_sents = None # list of stemmed sentences (lists of str)
    # Note: we could decide to lemmatize the text instead
    pos_sents = None # list of POS-tagged sentences (lists of
        # tuples containing tokens (str) and POS tags (str))
    parsed_sents = None # list of parsed sentences
    ngrams = Counter() # frequency distribution of token n-grams
    lc_ngrams = Counter() # frequency distribution of lower-cased
        # token n-grams
    char_ngrams = Counter() # frequency distribution of character
        # n-grams
    lc_char_ngrams = Counter # frequency distribution of lower-cased
        # character n-grams
    stem_ngrams = Counter() # frequency distribution of stemmed
        # token n-grams
    dep = None # frequency distribution of syntactic dependencies
    #suffix_tree = None # suffix tree representing text

    def __init__(self, review_text, hours_played, game_id):
        '''
        Initialization method.

        :param review_text: review text
        :type review_text: str
        :param hours_played: length of time author spent playing game
        :type hours_played: int
        :param game_id: game ID
        :type game_id: str
        '''

        # Make sure time is a float (or can be interpreted as an int, at least)
        try:
            hours_played = float(hours_played)
        except ValueError:
            sys.exit('ERROR: The \"hours_played\" parameter that was passed in, {},'
                     'could not be typecast as a float.\n\n'
                     'Exiting.\n'.format(hours_played))

        self.orig = review_text
        self.hours_played = hours_played
        self.game_id = game_id

        # Generate attribute values
        self.norm = normalize()
        self.tok_sents = tokenize()
        self.stem_sents = stem()
        self.pos_sents = pos_tag()
        self.parsed_sents = parse()

        # Feature frequency distributions
        self.ngrams = Counter()
        self.char_ngrams = Counter()
        self.dep = Counter()


    @staticmethod
    def normalize():
        '''
        Perform text preprocessing, i.e., lower-casing, etc., to generate the norm attribute.
        '''

        # Collapse all sequences of one or more whitespace characters, strip
        # whitespace off the ends of the string, and lower-case all characters
        r = re.sub(r'[\n\t ]+',
                   r' ',
                   self.orig.strip().lower())
        # Hand-crafted contraction-fixing rules
        # wont ==> won't
        r = re.sub(r"\bwont\b", r"won't", r, re.IGNORECASE)
        # dont ==> don't
        r = re.sub(r"\bdont\b", r"don't", r, re.IGNORECASE)
        # wasnt ==> wasn't
        r = re.sub(r"\bwasnt\b", r"wasn't", r, re.IGNORECASE)
        # werent ==> weren't
        r = re.sub(r"\bwerent\b", r"weren't", r, re.IGNORECASE)
        # aint ==> am not
        r = re.sub(r"\baint\b", r"am not", r, re.IGNORECASE)
        # arent ==> are not
        r = re.sub(r"\barent\b", r"are not", r, re.IGNORECASE)
        # cant ==> can not
        r = re.sub(r"\bcant\b", r"can not", r, re.IGNORECASE)
        # didnt ==> does not
        r = re.sub(r"\bdidnt\b", r"did not", r, re.IGNORECASE)
        # havent ==> have not
        r = re.sub(r"\bhavent\b", r"have not", r, re.IGNORECASE)
        # ive ==> I have
        r = re.sub(r"\bive\b", r"I have", r, re.IGNORECASE)
        # isnt ==> is not
        r = re.sub(r"\bisnt\b", r"is not", r, re.IGNORECASE)
        # theyll ==> they will
        r = re.sub(r"\btheyll\b", r"they will", r, re.IGNORECASE)
        # thats ==> that's
        r = re.sub(r"\bthatsl\b", r"that's", r, re.IGNORECASE)
        # whats ==> what's
        r = re.sub(r"\bwhats\b", r"what's", r, re.IGNORECASE)
        # wouldnt ==> would not
        r = re.sub(r"\bwouldnt\b", r"would not", r, re.IGNORECASE)
        # im ==> I am
        r = re.sub(r"\bim\b", r"I am", r, re.IGNORECASE)
        # youre ==> you are
        r = re.sub(r"\byoure\b", r"you are", r, re.IGNORECASE)
        # youve ==> you have
        r = re.sub(r"\byouve\b", r"you have", r, re.IGNORECASE)
        # ill ==> i will
        r = re.sub(r"\bill\b", r"i will", r, re.IGNORECASE)
        return r


    @staticmethod
    def tokenize():
        '''
        Perform tokenization using NLTK's sentence/word tokenizers.
        '''

        sents = sent_tokenize(self.norm)
        return [word_tokenize(sent) for sent in sents]


    @staticmethod
    def stem():
        '''
        Perform stemming
        '''

        sb = SnowballStemmer("english")
        stemmed_sents = []
        for sent in self.tok_sents:
            stemmed_sents.append([sb.stem(tok) for tok in sent])
        return stemmed_sents


    @staticmethod
    def pos_tag():
        '''
        Run NLTK POS-tagger on text.
        '''

        raise NotImplementedError


    @staticmethod
    def parse():
        '''
        Parse sentences.
        '''

        raise NotImplementedError


def generate_ngram_fdist(sents, _min=1, _max=3, lower=True):
    '''
    Generate frequency distribution for the tokens in the text.

    :param sents: list of sentence-corresponding lists of tokens that can be chopped up into n-grams.
    :type sents: list of lists/strs
    :param _min: minimum value of n for n-gram extraction
    :type _min: int
    :param _max: maximum value of n for n-gram extraction
    :type _max: int
    :param lower: whether or not to lower-case the text (True by default)
    :type lower: boolean
    :returns: Counter
    '''

    raise NotImplementedError


def generate_cngram_fdist(sents, _min=2, _max=5, lower=False):
    '''
    Generate frequency distribution for the characters in the text.

    :param sents: list of sentence-corresponding lists of characters that can be chopped up into n-grams.
    :type sents: list of lists/strs
    :param _min: minimum value of n for n-gram extraction
    :type _min: int
    :param _max: maximum value of n for n-gram extraction
    :type _max: int
    :param lower: whether or not to lower-case the text (False by default)
    :type lower: boolean
    :returns: Counter
    '''

    raise NotImplementedError


def generate_suffix_tree(self, max_depth=5):
    '''
    Generate suffix tree of a specified maximum depth (defaults to 5).

    :param max_depth: 
    :type max_depth: int
    '''

    raise NotImplementedError


if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage='python train.py',
        description='Build a machine learning model based on the ' \
                    'features that are extracted from a set of reviews ' \
                    'relating to a specific game or a set of games.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game_files',
        help='comma-separated list of file-names or "all" for all of the ' \
             'files (the game files should reside in the "data" directory)',
        type=str,
        required=True)
    args = parser.parse_args()

    # Establish connection to MongoDB database
    connection = pymongo.MongoClient('mongodb://localhost:27017')
    db = connection['reviews_project']
    reviewdb = db['reviews']

    