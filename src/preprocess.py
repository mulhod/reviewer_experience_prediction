'''
@author: Matt Mulholland
@date: 3/18/15

Module for Review objects, which include attributes representing the original text, the preprocessed version of the text, processed representations of the text, time author spent playing video game, author, game, etc.
'''
import sys
import re
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
        #self.dep = 
        #self.ngrams = 
        #self.lc_ngrams = 
        #self.char_ngrams = 
        #self.lc_char_ngrams = 
        #self.stem_ngrams = 
        #self.generate_suffix_tree()

    @staticmethod
    def normalize():
        '''
        Perform text preprocessing, i.e., lower-casing, etc., to generate the norm_text attribute.
        '''

        # Collapse all sequences of one or more whitespace characters, strip
        # whitespace off the ends of the string, and lower-case all characters
        return re.sub(r'[\n\t ]+',
                      r' ',
                      self.orig.strip().lower())


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


    @staticmethod
    def generate_ngram_fdist(sents, min=1, max=5):
        '''
        Generate frequency distribution for the tokens in the text (and possibly also for the lemmas or stems).

        :param sents: list of sentence-corresponding lists (of characters, tokens, etc.) that can be chopped up into n-grams.
        :type sents: list of lists/strs
        :param min: minimum value of n for n-gram extraction
        :type min: int
        :param max: maximum value of n for n-gram extraction
        :type max: int
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
