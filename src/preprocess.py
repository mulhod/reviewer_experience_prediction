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

    def __init__(self, text, time, author, game):
        '''
        Initialization method.

        :param text: review text
        :type text: str
        :param time: amount of time author spent playing game (in hours)
        :type time: int
        :param author: author name
        :type author: str
        :param game: game name
        :type game: str
        '''

        # Make sure time is an int (or can be interpreted as an int, at least)
        try:
            int(time)
        except ValueError:
            sys.exit('ERROR: The \"time\" parameter that was passed in, {},'
                     'could not be typecast as an int.\n\n'
                     'Exiting.\n'.format(time))

        # Initialization attributes
        self.orig_text = text
        self.time = int(time)
        self.author = author
        self.game = game

        # Text-related attributes
        self.norm_text = None # str representing normalize text
        self.tok_text = None # list of tokenized sentences (lists of str)
        self.stem_text = None # list of stemmed sentences (lists of str)
        # Note: we could decide to lemmatize the text instead
        self.pos_text = None # list of POS-tagged sentences (lists of tuples
            # containing tokens (str) and POS tags (str))
        self.dep_text = None # list of syntactic parse trees
        self.tok_fdist = Counter() # frequency distribution of tokens
        self.stem_fdist = Counter() # frequency distribution of stems
        self.suffix_tree = None # suffix tree representing text

        # Generate attribute values
        self.normalize_text()
        self.tokenize_text()
        self.stem_text()
        self.pos_tag_text()
        self.parse_text()
        self.generate_fdist()
        self.generate_suffix_tree()


    def normalize_text(self):
        '''
        Perform text preprocessing, i.e., lower-casing, etc., to generate the norm_text attribute.
        '''

        # Collapse all sequences of one or more whitespace characters, strip
        # whitespace off the ends of the string, and lower-case all characters
        self.norm_text = re.sub(r'[\n\t ]+',
                                r' ',
                                self.orig_text.strip().lower())


    def tokenize_text(self):
        '''
        Perform tokenization using NLTK's sentence/word tokenizers.
        '''

        sents = sent_tokenize(self.norm_text)
        self.tok_text = [word_tokenize(sent) for sent in sents]


    def stem_text(self):
        '''
        Perform stemming
        '''

        stemmer = SnowballStemmer("english")
        stemmed_sents = []
        for sent in self.tok_text:
            stemmed_sents.append([stemmer.stem(tok) for tok in sent])
        self.stem_text = stemmed_sents


    def pos_tag_text(self):
        '''
        Run NLTK POS-tagger on text.
        '''

        raise NotImplementedError


    def parse_text(self):
        '''
        Parse sentences.
        '''

        raise NotImplementedError


    def generate_fdist(self):
        '''
        Generate frequency distribution for the tokens in the text (and possibly also for the lemmas or stems).
        '''

        [self.tok_fdist.update(sent) for sent in self.tok_text]


    def generate_stem_fdist(self):
        '''
        Generate frequency distribution for the tokens in the text (and possibly also for the lemmas or stems).
        '''

        [self.stem_fdist.update(sent) for sent in self.stem_text]


    def generate_suffix_tree(self, max_depth=5):
        '''
        Generate suffix tree of a specified maximum depth (defaults to 5).

        :param max_depth: 
        :type max_depth: int
        '''

        raise NotImplementedError