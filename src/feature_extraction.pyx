'''
@author Matt Mulholland
@date 05/05/2015

Module of functions/classes related to feature extraction, model-building, ARFF file generation, etc.
'''
import sys
import time
#from ctypes import *
from math import ceil
from numpy import log2
from nltk.util import ngrams
from string import punctuation
from re import sub, IGNORECASE
from collections import Counter
from configparser import ConfigParser

cdef class Review(object):
    '''
    Class for objects representing Reviews.
    '''

    # Original review text
    #orig = None
    # Normalized review text
    norm = None
    # Number of hours the reviewer has played the game (float)
    cdef float hours_played
    # appid of the game (string ID code that Steam uses to represent the
    # game
    #appid = None
    # Attribute whose value determines whether or not the review text will
    # be lower-cased as part of the normalization step
    #lower = None
    # Length of the original text (base-2 log)
    cdef float length
    # Attributes representing the word- and sentence-tokenized
    # representations of self.norm, consisting of a list of elements
    # corresponding to the identified sentences, which in turn consist of
    # a list of elements corresponding to the identified tokens, tags,
    # lemmas, respectively
    tokens = []
    tags = []
    #lemmas = []
    # Attributes representing the spaCy text annotations
    spaCy_annotations = None
    spaCy_sents = None
    #spaCy_ents = None
    # Atrribute representing the named entities in the review
    #entities = []
    # Attribute representing the syntactic heads of each token
    #heads = []
    # Attribute representing the syntactic child(ren) of each token (if
    # any), which will be represented as a Counter mapping a token and its
    # children to frequencies
    #children = Counter()
    # Attributes representing the cluster IDs and repvecs (representation
    # vectors) corresponding to tokens
    # Maybe it would be a good idea to make a frequency distribution of
    # the cluster IDs...
    #cluster_ids = []
    #repvecs = []


    def __init__(self, review_text, float hours_played, game, appid,
                 spaCy_nlp, lower=True):
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
        :param lower: include lower-casing as part of the review text normalization step
        :type lower: boolean
        '''

        self.orig = review_text
        self.hours_played = hours_played
        self.appid = appid
        self.lower = lower

        # Generate attribute values
        self.length = ceil(log2(len(self.orig))) # Get base-2 log of th
            # length of the original version of the review text, not the
            # normalized version
        self.normalize()
        # Use spaCy to analyze the normalized version of the review text
        self.spaCy_annotations = spaCy_nlp(self.norm,
                                           tag=True,
                                           parse=True)
        self.spaCy_sents = [list(sent) for sent in
                            self.spaCy_annotations.sents]
        #self.spaCy_ents = [list(ent) for ent in self.spaCy_annotations.ents]
        self.get_token_features_from_spaCy()
        #self.get_entities_from_spaCy()


    def normalize(self):
        '''
        Perform text preprocessing, i.e., lower-casing, etc., to generate the norm attribute.
        '''

        # Lower-case text if self.lower is Trues
        if self.lower:
            r = self.orig.lower()
        else:
            r = self.orig

        # Collapse all sequences of one or more whitespace characters, strip
        # whitespace off the ends of the string, and lower-case all characters
        r = sub(r'[\n\t ]+',
                r' ',
                r.strip())
        # Hand-crafted contraction-fixing rules
        # wont ==> won't
        r = sub(r"\bwont\b", r"won't", r, IGNORECASE)
        # dont ==> don't
        r = sub(r"\bdont\b", r"don't", r, IGNORECASE)
        # wasnt ==> wasn't
        r = sub(r"\bwasnt\b", r"wasn't", r, IGNORECASE)
        # werent ==> weren't
        r = sub(r"\bwerent\b", r"weren't", r, IGNORECASE)
        # aint ==> am not
        r = sub(r"\baint\b", r"am not", r, IGNORECASE)
        # arent ==> are not
        r = sub(r"\barent\b", r"are not", r, IGNORECASE)
        # cant ==> can not
        r = sub(r"\bcant\b", r"can not", r, IGNORECASE)
        # didnt ==> does not
        r = sub(r"\bdidnt\b", r"did not", r, IGNORECASE)
        # havent ==> have not
        r = sub(r"\bhavent\b", r"have not", r, IGNORECASE)
        # ive ==> I have
        r = sub(r"\bive\b", r"I have", r, IGNORECASE)
        # isnt ==> is not
        r = sub(r"\bisnt\b", r"is not", r, IGNORECASE)
        # theyll ==> they will
        r = sub(r"\btheyll\b", r"they will", r, IGNORECASE)
        # thats ==> that's
        r = sub(r"\bthatsl\b", r"that's", r, IGNORECASE)
        # whats ==> what's
        r = sub(r"\bwhats\b", r"what's", r, IGNORECASE)
        # wouldnt ==> would not
        r = sub(r"\bwouldnt\b", r"would not", r, IGNORECASE)
        # im ==> I am
        r = sub(r"\bim\b", r"I am", r, IGNORECASE)
        # youre ==> you are
        r = sub(r"\byoure\b", r"you are", r, IGNORECASE)
        # youve ==> you have
        r = sub(r"\byouve\b", r"you have", r, IGNORECASE)
        # ill ==> i will
        r = sub(r"\bill\b", r"i will", r, IGNORECASE)
        self.norm = r


    def get_token_features_from_spaCy(self):
        '''
        Get tokens-related features from spaCy's text annotations.
        '''

        for sent in self.spaCy_sents:
            # Get tokens
            self.tokens.append([t.orth_ for t in sent])
            # Get tags
            self.tags.append([t.tag_ for t in sent])
            # Get lemmas
            #self.lemmas.append([t.lemma_ for t in sent])
            # Get syntactic heads
            #self.heads.append([t.head.orth_ for t in sent])


    def get_entities_from_spaCy(self):
        '''
        Get named entities from spaCy's text annotations.
        '''

        for entity in self.spaCy_ents:
            self.entities.append(dict(entity=entity.orth_,
                                      label=entity.label_))


def extract_features_from_review(_review, lowercase_cngrams=False):
    '''
    Extract word/character n-gram features, length, and syntactic dependency features from a Review object and return as dictionary where each feature ("wngrams" for word n-grams, "cngrams" for character n-grams, "length" for length, and "dep" for syntactic dependency features) is represented as a key:value mapping in which the key is a string with the name of the feature class followed by two hashes and then the string representation of the feature (e.g. "the dog" for an example n-gram feature, "th" for an example character n-gram feature, or "step:forward" for an example syntactic dependency feature) and the value is the frequency with which that feature occurred in the review.

    :param _review: object representing the review
    :type _review: Review object
    :param lowercase_cngrams: whether or not to lower-case the review text before extracting character n-grams
    :type lowercase_cngrams: boolean (False by default)
    :returns: dict
    '''

    def generate_ngram_fdist(sents, _min=1, _max=2):
        '''
        Generate frequency distribution for the tokens in the text.

        :param sents: list of lists of tokens that can be chopped up into n-grams.
        :type sents: list of lists/strs
        :param _min: minimum value of n for n-gram extraction
        :type _min: int
        :param _max: maximum value of n for n-gram extraction
        :type _max: int
        :returns: Counter
        '''

        # Make emtpy Counter
        ngram_counter = Counter()

        # Count up all n-grams
        for sent in sents:
            for i in range(_min, _max + 1):
                ngram_counter.update(list(ngrams(sent, i)))

        # Re-represent keys as string representations of specific features
        # of the feature class "ngrams"
        for ngram in list(ngram_counter):
            ngram_counter['ngrams##{}'.format(' '.join(ngram))] = \
                ngram_counter[ngram]
            del ngram_counter[ngram]

        return ngram_counter


    def generate_cngram_fdist(text, _min=2, _max=5):
        '''
        Generate frequency distribution for the characters in the text.

        :param text: review text
        :type text: str
        :param _min: minimum value of n for character n-gram extraction
        :type _min: int
        :param _max: maximum value of n for character n-gram extraction
        :type _max: int
        :returns: Counter
        '''

        # Make emtpy Counter
        cngram_counter = Counter()

        # Count up all character n-grams
        for i in range(_min, _max + 1):
            cngram_counter.update(list(ngrams(text, i)))

        # Re-represent keys as string representations of specific features
        # of the feature class "cngrams" (and set all values to 1 if binarize
        # is True)
        for cngram in list(cngram_counter):
            cngram_counter['cngrams##{}'.format(''.join(cngram))] = \
                cngram_counter[cngram]
            del cngram_counter[cngram]

        return cngram_counter


    def generate_dep_features(spaCy_annotations):
        '''
        Generate syntactic dependency features from spaCy text annotations.

        :param spaCy_annotations: spaCy English text analysis object
        :type spaCy_annotations: spacy.en.English instance
        :returns: Counter object representing a frequency distribution of syntactic dependency features
        '''

        # Make emtpy Counter
        dep_counter = Counter()

        # Iterate through spaCy annotations for each sentence and then for
        # each token
        for s in spaCy_annotations.sents:
            for t in s:
                # If the number of children to the left and to the right
                # of the token add up to a value that is not zero, then
                # get the children and make dependency features with
                # them
                if t.n_lefts + t.n_rights:
                    fstr = "dep##{0.orth_}:{1.orth_}"
                    [dep_counter.update({fstr.format(t, c): 1})
                     for c in t.children if not c.tag_ in punctuation]

        return dep_counter

    # Extract features
    features = Counter()

    # Get the length feature
    # Note: This feature will always be mapped to a frequency of 1 since
    # it exists for every single review and, thus, a review of this length
    # being mapped to the hours played value that it is mapped to has
    # occurred once.
    features.update({'length##{}'.format(_review.length): 1})

    # Extract n-gram features
    features.update(generate_ngram_fdist(_review.tokens))

    # Extract character n-gram features
    orig = _review.orig.lower() if lowercase_cngrams else _review.orig
    features.update(generate_cngram_fdist(orig))

    # Generate the syntactic dependency features
    features.update(generate_dep_features(_review.spaCy_annotations))

    return dict(features)


def write_config_file(config_dict, path):
    '''
    This Creates a configparser config file from a dict and writes it to a file that can be read in by SKLL.  The dict should map keys for the SKLL config sections to dictionaries of key-value pairs for each section.

    :param config_dict: configuration dictionary
    :type config_dict: dict
    :param path: destination path to configuration file
    :type path: str
    :returns: None
    '''

    cfg = ConfigParser()
    for section_name, section_dict in config_dict.items():
        cfg.add_section(section_name)
        for key, val in section_dict.items():
            cfg.set(section_name, key, val)
    with open(path, 'w') as config_file:
        cfg.write(config_file)


def write_arff_file(dest_path, file_names, reviews=None, reviewdb=None,
                    make_train_test=False):
    '''
    Write .arff file either for a list of reviews read in from a file or list of files or for both the training and test partitions in the MongoDB database.

    :param reviews: list of dicts with hours/review keys-value mappings representing each data-point (defaults to None)
    :type reviews: list of dict
    :param reviewdb: MongoDB reviews collection
    :type reviewdb: pymongo.MongoClient object (None by default)
    :param dest_path: path for .arff output file
    :type dest_path: str
    :param file_names: list of extension-less game file-names
    :type file_names: list of str
    :param make_train_test: if True, use MongoDB collection to find reviews that are from the training and test partitions and make files for them instead of making one big file (defaults to False)
    :type make_train_test: boolean
    :returns: None
    '''

    # Make sure that the passed-in keyword arguments make sense
    if make_train_test and (reviews or not reviewdb):
        sys.exit('ERROR: The make_train_test keyword argument was set to ' \
                 'True and either the reviewdb keyword was left unspecified' \
                 ' or the reviews keyword was specified (or both). If the ' \
                 'make_train_test keyword is used, it is expected that ' \
                 'training/test reviews will be retrieved from the MongoDB ' \
                 'database rather than a list of reviews passed in via the ' \
                 'reviews keyword. Exiting.\n')
    if not make_train_test and reviewdb:
        if reviews:
            sys.stderr.write('WARNING: Ignoring passed-in reviewdb keyword ' \
                             'value. Reason: If a list of reviews is passed' \
                             ' in via the reviews keyword argument, then ' \
                             'the reviewdb keyword argument should not be ' \
                             'used at all since it will not be needed.\n')
        else:
            sys.exit('ERROR: A list of review dictionaries was not ' \
                     'specified. Exiting.\n')

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

            # Modify file-path by adding partition suffix
            suffix = 'train' if partition.startswith('train') else 'test'
            replacement = '.{}.arff'.format(suffix)
            _dest_path = sub(r'\.arff$',
                             replacement,
                             dest_path)

            # Get reviews for the given partition from all of the games
            game_docs_cursor = reviewdb.find({'partition': partition,
                                              'game': {'$in': file_names}})
            if game_docs_cursor.count() == 0:
                sys.exit('ERROR: No matching documents were found in the ' \
                         'MongoDB collection for the {} partition and the' \
                         ' following games:\n\n{}\nExiting.' \
                         '\n'.format(partition,
                                     file_names))

            game_docs = list(game_docs_cursor)
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
                reviews_lines.append('"{}",{}'.format(review,
                                                      game_doc['hours']))
            with open(_dest_path,
                      'w') as out:
                t = time.strftime(TIMEF)
                out.write('{}\n{}'.format(ARFF_BASE.format(t,
                                                           _file_names),
                                          '\n'.join(reviews_lines)))
    else:

        if not reviews:
            sys.exit('ERROR: Empty list of reviews passed in to the ' \
                     'write_arff_file method. Exiting.\n')

        # Make empty list of lines to populate with ARFF-style lines,
        # one per review
        reviews_lines = []

        for review_dict in reviews:
            # Remove single/double quotes from the reviews first...
            review = sub(r'\'|"',
                         r'',
                         review_dict['review'].lower())
            # Get rid of backslashes since they only make things confusing
            review = sub(r'\\',
                         r'',
                         review)
            reviews_lines.append('"{}",{}'.format(review,
                                                  review_dict['hours']))
        with open(dest_path,
                      'w') as out:
            t = time.strftime(TIMEF)
            out.write('{}\n{}'.format(ARFF_BASE.format(t,
                                                       _file_names),
                                      '\n'.join(reviews_lines)))