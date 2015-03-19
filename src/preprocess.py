'''
@author: Matt Mulholland
@date: 3/18/15

Module for Review objects, which include attributes representing the original text, the preprocessed version of the text, processed representations of the text, time author spent playing video game, author, game, etc.
'''
import sys
import os
from collections import Counter


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

        # Initialization attributes
        self.orig_text = text
        self.time = time
        self.author = author
        self.game = game

        # Text-related attributes
        self.norm_text = None # str representing normalize text
        self.tok_text = None # list of tokenized sentences (lists of str)
        #self.lem_text = None # list of lemmatized (or stemmed) sentences
        #    # (lists of str)
        self.pos_text = None # list of POS-tagged sentences (lists of tuples
            # containing tokens (str) and POS tags (str))
        self.dep_text = None # list of syntactic parse trees
        self.tok_fdist = Counter() # frequency distribution of tokens
        #self.lem_fdist = Counter() # frequency distribution of lemmas (or
        #    # stems)
        self.suffix_tree = None # suffix tree representing text

        # Generate attribute values
        self.normalize_text()
        self.tokenize_text()
        self.lemmatize_text()
        self.pos_tag_text()
        self.parse_text()
        self.generate_fdist()
        self.generate_suffix_tree()


    def normalize_text(self):
        '''
        Perform text preprocessing, i.e., lower-casing, etc., to generate values for the norm_text and tok_text (and possibly lem_text as well) attributes.
        '''

        raise NotImplementedError


    def tokenize_text(self):
        '''
        Perform tokenization using NLTK's sentence/word tokenizers.
        '''

        raise NotImplementedError


    def lemmatize_text(self):
        '''
        Perform sentence lemmatization (or stemming).
        '''

        raise NotImplementedError


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
        Generate frequency distributions for the tokens in the text (and possibly also for the lemmas or stems).
        '''

        raise NotImplementedError


    def generate_suffix_tree(self, max_depth=5):
        '''
        Generate suffix tree of a specified maximum depth (defaults to 5).

        :param max_depth: 
        :type max_depth: int
        '''

        raise NotImplementedError