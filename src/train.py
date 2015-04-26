'''
@author: Matt Mulholland, Janette Martinez, Emily Olshefski
@date: 3/18/15

Script used to train a model on a given data-set (or multiple data-sets combined).
'''
import sys
import re
import pymongo
import argparse
from os import listdir
from numpy import log2
from data import APPID_DICT
from spacy.en import English
from collections import Counter
from skll import run_configuration
from nltk.stem import SnowballStemmer
from os.path import join, dirname, realpath, abspath


class Review(object):
    '''
    Class for objects representing Reviews.
    '''

    def __init__(self, review_text, hours_played, game, appid, spaCy_nlp):
        '''
        Initialization method.

        :param review_text: review text
        :type review_text: str
        :param hours_played: length of time author spent playing game
        :type hours_played: float
        :param game: name of game
        :type game: str
        :param appid: appid string (usually a number of up to 6-7 digits)
        :type appid: str
        :param spaCy_nlp: spaCy English analyzer
        :type spaCy_nlp: spaCy.en.English
        '''

        # Make sure time is a float (or can be interpreted as an int, at least)
        try:
            hours_played = float(hours_played)
        except ValueError:
            sys.exit('ERROR: The \"hours_played\" parameter that was passed in, {},'
                     'could not be typecast as a float.\n\n'
                     'Exiting.\n'.format(hours_played))

        self.orig = review_text
        try:
            self.hours_played = hours_played
        except ValueError:
            sys.exit('ERROR: hours_played value not castable to type float: {}' \
                     '\nExiting.\n'.format(hours_played))
        self.appid = appid
        # Attribute representing the normalized text (str)
        self.norm = None
        # Attributes representing the word- and sentence-tokenized representations
        # of self.norm, both consisting of a list of elements corresponding to the
        # identified sentences, which in turn consist of list of elements
        # corresponding to the identified tokens, tags, lemmas, etc.
        self.tokens = []
        self.tags = []
        self.lemmas = []

        # Atrribute representing the named entities in the review
        self.entities = []

        # Attribute representing the dependency labels for each token
        self.dep = []

        # Attribute representing the syntactic heads of each token
        self.heads = []

        # Attribute representing the syntactic child(ren) of each token (if any),
        # which will be represented as a Counter mapping a token and its children
        # to frequencies
        self.children = Counter()

        # Cluster IDs and repvec (representation vectors) corresponding to tokens
        # Maybe it would be a good idea to make a frequency distribution of the
        # cluster IDs...
        # self.cluster_ids = []
        # self.repvecs = []

        # Generate attribute values
        self.length = log2(len(review_text)) # Get base-2 log of the length of the
            # original version of the review text, not the normalized version
        self.normalize()
        # Use spaCy to analyze the normalized version of the review text
        spaCy_annotations = self.spaCy_nlp(self.norm,
                                           tag=True,
                                           parse=True)
        self.get_tokens_from_spaCy()
        self.get_entities_from_spaCy()


    @staticmethod
    def normalize(self, lower=True):
        '''
        Perform text preprocessing, i.e., lower-casing, etc., to generate the norm attribute.
        '''

        # Collapse all sequences of one or more whitespace characters, strip
        # whitespace off the ends of the string, and lower-case all characters
        if lower:
            r = re.sub(r'[\n\t ]+',
                       r' ',
                       self.orig.strip().lower())
        else:
            r = re.sub(r'[\n\t ]+',
                       r' ',
                       self.orig.strip())
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
        self.norm = str(r)


    def get_tokens_from_spaCy(self):
        '''
        Get tokens/tags from spaCy's text annotations.
        '''

        for sent in self.spaCy_annotations.sents:
            # Get tokens
            self.tokens.append([t.orth_ for t in sent])
            # Get tags
            self.tags.append([t.tag_ for t in sent])
            # Get lemmas
            self.lemmas.append([t.lemma_ for t in sent])
            # Get syntactic heads
            self.heads.append([t.head.orth_ for t in sent])
            # Get syntactic children
            for t in sent:
                #children = [c for c in t.children]
                #if children:
                #    for c in children:
                #        c = {"children##{0.orth_}:{1.orth_}".format(t, c): 1}
                #        self.children.update(c)
                if t.n_lefts + t.n_rights:
                    fstring = "children##{0.orth_}:{1.orth_}"
                    [self.children.update({"fstring".format(t, c): 1}) for c in
                     t.children]


    def get_entities_from_spaCy(self):
        '''
        Get named entities from spaCy's text annotations.
        '''

        for entity in self.spaCy_annotations.ents:
            self.entities.append(dict(entity=entity.orth_,
                                      label=entity.label_))


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
    parser.add_argument('--combine',
        help='combine all game files together to make one big model',
        action='store_true',
        required=False)
    parser.add_argument('--combined_model_prefix',
        help='prefix to use when naming the combined model',
        type=str,
        required=False)
    args = parser.parse_args()

    # Get paths to the project and data directories
    project_dir = dirname(dirname(abspath(realpath(__file__))))
    data_dir = join(project_dir,
                    'data')

    # Make sure that, if --combine is being used, there is also a file
    # prefix being passed in via --combined_model_prefix for the combined
    # model
    if args.combine and not args.combined_model_prefix:
        sys.exit('ERROR: When using the --combine flag, you must also ' \
                 'specify a model prefix, which can be passed in via ' \
                 'the --combined_model_prefix option argument. ' \
                 'Exiting.\n')

    # Establish connection to MongoDB database
    connection_string = 'mongodb://localhost:27017'
    try:
        connection = pymongo.MongoClient(connection_string)
    except pymongo.errors.ConnectionFailure as e:
        sys.exit('ERROR: Unable to connecto to Mongo server at ' \
                 '{}'.format(connection_string))
    db = connection['reviews_project']
    reviewdb = db['reviews']

    # Initialize an English-language NLP spaCy analyzer
    spaCy_nlp = English()

    # Get list of games
    game_files = []
    if args.game_files == "all":
        game_files = [f for f in listdir(data_dir) if f.endswith('.txt')]
    else:
        game_files = args.game_files.split(',')

    if args.combine:
        # Initialize empty list for holding all of the feature dictionaries
        # from each review in each game
        feature_dicts = []
        for game_file in game_files:
            # Get the training reviews for this game from the Mongo
            # database
            game = game_file[:-4]
            appid = APPID_DICT[game]
            game_docs = list(reviewdb.find({'game': game,
                                            'partition': 'training'}))
            for game_doc in game_docs:
                _id = str(game_doc['_id'])
                hours = game_doc['hours']
                _Review = Review(game_doc['review'],
                                 hours,
                                 game,
                                 appid,
                                 spaCy_nlp)
                # Extract features from the review text
                game_features = Counter()
                ngrams_counter = generate_ngram_fdist(_Review.tok_sents)
                cngrams_counter = generate_cngram_fdist(_Review.norm)
                length_feature = {'length##{}'.format(_Review.length): 1}
                game_features.update(ngrams_counter)
                game_features.update(cngrams_counter)
                game_features.update(length_feature)
                feature_dicts.append({'id': _id,
                                      'y': hours,
                                      'x': game_features})
                # Optional: update Mongo database with a new key, which will
                # be mapped to feature_dicts
        # Write .jsonlines file, create the config file, and then run the
        # configuration, producing a model file
    else:
        for game_file in game_files:
            