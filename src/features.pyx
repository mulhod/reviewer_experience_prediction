"""
:authors: Matt Mulholland (mulhodm@gmail.com), Janette Martinez,
          Emily Olshefski
:date: 05/05/2015

Module of functions/classes related to feature extraction, ARFF file
generation, etc.
"""
import logging
from math import ceil
from time import sleep
from json import dumps
from os.path import join
from string import punctuation
from collections import Counter
from itertools import combinations
from re import (IGNORECASE,
                compile as recompile)

import numpy as np
from bson import BSON
from typing import (Dict,
                    Union)
from nltk.util import ngrams
from spacy.en import English
from pymongo.cursor import Cursor
from skll.metrics import (kappa,
                          pearson)
from bson.objectid import ObjectId
from pymongo.collection import Collection

from src import Numeric

bson_decode = BSON.decode
spaCy_nlp = English()

# Logging-related
logger = logging.getLogger(__name__)
logwarn = logger.warning
logerr = logger.error

# Review preprocessing-related regular expressions
SPACES = recompile(r'[\n\t ]+')
spaces_sub = SPACES.sub
WONT = recompile(r"\bwont\b", IGNORECASE)
wont_sub = WONT.sub
DONT = recompile(r"\bdont\b", IGNORECASE)
dont_sub = DONT.sub
WASNT = recompile(r"\bwasnt\b", IGNORECASE)
wasnt_sub = WASNT.sub
WERENT = recompile(r"\bwerent\b", IGNORECASE)
werent_sub = WERENT.sub
AINT = recompile(r"\baint\b", IGNORECASE)
aint_sub = AINT.sub
ARENT = recompile(r"\barent\b", IGNORECASE)
arent_sub = ARENT.sub
CANT = recompile(r"\bcant\b", IGNORECASE)
cant_sub = CANT.sub
DIDNT = recompile(r"\bdidnt\b", IGNORECASE)
didnt_sub = DIDNT.sub
HAVENT = recompile(r"\bhavent\b", IGNORECASE)
havent_sub = HAVENT.sub
IVE = recompile(r"\bive\b", IGNORECASE)
ive_sub = IVE.sub
ISNT = recompile(r"\bisnt\b", IGNORECASE)
isnt_sub = ISNT.sub
THEYLL = recompile(r"\btheyll\b", IGNORECASE)
theyll_sub = THEYLL.sub
THATS = recompile(r"\bthats\b", IGNORECASE)
thats_sub = THATS.sub
WHATS = recompile(r"\bwhats\b", IGNORECASE)
whats_sub = WHATS.sub
WOULDNT = recompile(r"\bwouldnt\b", IGNORECASE)
wouldnt_sub = WOULDNT.sub
IM = recompile(r"\bim\b", IGNORECASE)
im_sub = IM.sub
YOURE = recompile(r"\byoure\b", IGNORECASE)
youre_sub = YOURE.sub
YOUVE = recompile(r"\byouve\b", IGNORECASE)
youve_sub = YOUVE.sub
ILL = recompile(r"\bill\b", IGNORECASE)
ill_sub = ILL.sub
preprocessing_regex_funcs = [lambda x: spaces_sub(r' ', x),
                             lambda x: wont_sub(r"won't", x),
                             lambda x: dont_sub(r"don't", x),
                             lambda x: wasnt_sub(r"wasn't", x),
                             lambda x: werent_sub(r"weren't", x),
                             lambda x: aint_sub(r"am not", x),
                             lambda x: arent_sub(r"are not", x),
                             lambda x: cant_sub(r"can not", x),
                             lambda x: didnt_sub(r"did not", x),
                             lambda x: havent_sub(r"have not", x),
                             lambda x: ive_sub(r"I have", x),
                             lambda x: isnt_sub(r"is not", x),
                             lambda x: theyll_sub(r"they will", x),
                             lambda x: thats_sub(r"that's", x),
                             lambda x: whats_sub(r"what's", x),
                             lambda x: wouldnt_sub(r"would not", x),
                             lambda x: im_sub(r"I am", x),
                             lambda x: youre_sub(r"you are", x),
                             lambda x: youve_sub(r"you have", x),
                             lambda x: ill_sub(r"i will", x)]


class Review(object):
    """
    Class for objects representing review texts and NLP features.
    """

    def __init__(self, review_text: str, lower: bool = True) -> 'Review':
        """
        Initialization method.

        :param review_text: review text
        :type review_text: str
        :param game: name of game
        :type game: str
        :param lower: include lower-casing as part of the review text
                      normalization step
        :type lower: bool
        """

        # Get review text and lower-casing attributes
        self.orig = review_text
        self._lower = lower

        # Get base-2 log of the length of the original (read: before
        # normalization) version of the review text
        self.length = ceil(np.log2(len(self.orig)))
        self._normalize()

        # Use spaCy to analyze the normalized version of the review
        # text
        self.spaCy_annotations = spaCy_nlp(self.norm, tag=True, parse=True)
        self.spaCy_sents = []
        spaCy_sents_append = self.spaCy_sents.append
        for _range in self.spaCy_annotations.sents:
            spaCy_sents_append([self.spaCy_annotations[i] for i in range(*_range)])
        self._get_token_features_from_spaCy()

    def _normalize(self) -> None:
        """
        Perform text preprocessing, i.e., lower-casing, etc., to
        generate the norm attribute.

        :returns: None
        :rtype: None
        """

        # Lower-case text if self.lower is True
        r = self.orig.lower() if self._lower else self.orig

        # Collapse all sequences of one or more whitespace characters,
        # strip whitespace off the ends of the string, and lower-case
        # all characters and apply the hand-crafted contraction-fixing
        # rules
        r = r.strip()
        for regex_sub in preprocessing_regex_funcs:
            r = regex_sub(r)
        self.norm = r

    def _get_token_features_from_spaCy(self):
        """
        Get tokens-related features from spaCy's text annotations,
        including Brown corpus cluster IDs.

        :returns: None
        :rtype: None
        """

        self.tokens = []
        tokens_append = self.tokens.append
        cluster_ids = []
        cluster_ids_extend = cluster_ids.extend
        for sent in self.spaCy_sents:
            # Get tokens and clusters
            tokens_append([t.orth_ for t in sent])
            cluster_ids_extend([t.cluster for t in sent])
        self.cluster_id_counter = dict(Counter(cluster_ids))


def extract_features(review: Review,
                     lowercase_cngrams: bool = False) -> Dict[str, int]:
    """
    Extract word/character n-gram, length, Brown corpus cluster ID,
    and syntactic dependency features from a Review object and return
    as dictionary where each feature is represented as a key:value
    mapping in which the key is a string representation of the feature
    (e.g. "the dog" for an example n-gram feature, "th" for an example
    character n-gram feature, "c667" for an example cluster feature,
    and "step:VMOD:forward" for an example syntactic dependency
    feature) and the value is the frequency with which that feature
    occurred in the review.

    :param review: object representing the review
    :type review: Review object
    :param lowercase_cngrams: whether or not to lower-case the review
                              text before extracting character n-grams
                              (False by default)
    :type lowercase_cngrams: bool

    :returns: feature dictionary
    :rtype: dict
    """

    def generate_ngram_fdist(_min: int = 1, _max: int = 2) -> Counter:
        """
        Generate frequency distribution for the tokens in the text.

        :param _min: minimum value of n for n-gram extraction
        :type _min: int
        :param _max: maximum value of n for n-gram extraction
        :type _max: int

        :returns: frequency distribution of word n-grams
        :rtype: Counter
        """

        # Make emtpy Counter
        ngram_counter = Counter()
        ngram_counter_update = ngram_counter.update

        # Count up all n-grams
        for sent in review.tokens:
            for i in range(_min, _max + 1):
                ngram_counter_update(list(ngrams(sent, i)))

        # Re-represent keys as string representations of specific
        # features of the feature class "ngrams"
        for ngram in list(ngram_counter):
            ngram_counter[' '.join(ngram)] = ngram_counter[ngram]
            del ngram_counter[ngram]

        return ngram_counter

    def generate_cngram_fdist(_min: int = 2, _max: int = 5) -> Counter:
        """
        Generate frequency distribution for the characters in the text.

        :param _min: minimum value of n for character n-gram extraction
        :type _min: int
        :param _max: maximum value of n for character n-gram extraction
        :type _max: int

        :returns: frequency distribution of character n-grams
        :rtype: Counter
        """

        # Make emtpy Counter
        cngram_counter = Counter()
        cngram_counter_update = cngram_counter.update

        # Count up all character n-grams
        for i in range(_min, _max + 1):
            cngram_counter_update(list(ngrams(review.orig.lower()
                                              if lowercase_cngrams
                                              else review.orig, i)))

        # Re-represent keys as string representations of specific
        # features of the feature class "cngrams" (and set all values
        # to 1 if binarize is True)
        for cngram in list(cngram_counter):
            cngram_counter[''.join(cngram)] = cngram_counter[cngram]
            del cngram_counter[cngram]

        return cngram_counter

    def generate_cluster_fdist() -> Counter:
        """
        Convert Brown corpus cluster ID frequency distribution to a
        frequency distribution where the keys are strings representing
        "cluster" features (rather than just a number, the cluster ID).

        :returns: frequency distribution of cluster IDs
        :rtype: Counter
        """

        cluster_fdist = review.cluster_id_counter
        for cluster_id, freq in list(cluster_fdist.items()):
            del cluster_fdist[cluster_id]
            cluster_fdist['cluster{0}'.format(cluster_id)] = freq

        return cluster_fdist

    def generate_dep_features() -> Counter:
        """
        Generate syntactic dependency features from spaCy text
        annotations and represent the features as token (lemma) +
        dependency type + child token (lemma).

        :returns: frequency distribution of syntactic dependency
                  features
        :rtype: Counter
        """

        # Make emtpy Counter
        dep_counter = Counter()
        dep_counter_update = dep_counter.update

        # Iterate through spaCy annotations for each sentence and then
        # for each token
        for sent in review.spaCy_sents:
            for t in sent:
                # If the number of children to the left and to the
                # right of the token add up to a value that is not
                # zero, then get the children and make dependency
                # features with them
                if t.n_lefts + t.n_rights:
                    [dep_counter_update({'{0.lemma_}:{0.dep_}:{1.lemma_}'
                                         .format(t, c): 1})
                     for c in t.children if not c.tag_ in punctuation]

        return dep_counter

    # Extract features
    feats = {}

    feats_update = feats.update
    # Get the length feature
    feats_update({str(review.length): 1})

    # Extract n-gram features
    feats_update(generate_ngram_fdist())

    # Extract character n-gram features
    feats_update(generate_cngram_fdist())

    # Convert cluster ID values into useable features
    feats_update(generate_cluster_fdist())

    # Generate the syntactic dependency features
    feats_update(generate_dep_features())

    return feats


def get_nlp_features_from_db(db: Collection, _id: ObjectId) -> Dict[str, int]:
    """
    Collect the NLP features from the Mongo database collection for a
    given review and return the decoded value.

    :param db: MongoDB collection
    :type db: Collection
    :param _id: MongoDB document's ObjectId
    :type _id: ObjectId

    :returns: dictionary of features if features were found; otherwise,
              an empty dictionary
    :rtype: dict
    """

    nlp_feats_doc = db.find_one({'_id': _id}, {'_id': 0, 'nlp_features': 1})
    return bson_decode(nlp_feats_doc.get('nlp_features')) if nlp_feats_doc else {}


def get_steam_features_from_db(get_feat) -> Dict[str, Union[Numeric, bool]]:
    """
    Get features collected from Steam (i.e., the non-NLP features).

    :param get_feat: built-in method get of dictionary object
                     representing a single Mongo database document
    :type get_feat: method/function

    :returns: non-NLP features/review attributes
    :rtype: dict
    """

    achievements = get_feat('achievement_progress')
    steam_feats = {'total_game_hours': get_feat('total_game_hours'),
                   'total_game_hours_bin': get_feat('total_game_hours_bin'),
                   'total_game_hours_last_two_weeks':
                       get_feat('total_game_hours_last_two_weeks'),
                   'num_found_funny': get_feat('num_found_funny'),
                   'num_found_helpful': get_feat('num_found_helpful'),
                   'found_helpful_percentage':
                       get_feat('found_helpful_percentage'),
                   'num_friends': get_feat('num_friends'),
                   'friend_player_level': get_feat('friend_player_level'),
                   'num_groups': get_feat('num_groups'),
                   'num_screenshots': get_feat('num_screenshots'),
                   'num_workshop_items': get_feat('num_workshop_items'),
                   'num_comments': get_feat('num_comments'),
                   'num_games_owned': get_feat('num_games_owned'),
                   'num_reviews': get_feat('num_reviews'),
                   'num_guides': get_feat('num_guides'),
                   'num_badges': get_feat('num_badges'),
                   'updated': 1 if get_feat('date_updated') else 0,
                   'num_achievements_attained':
                       achievements.get('num_achievements_attained'),
                   'num_achievements_percentage':
                       achievements.get('num_achievements_percentage'),
                   'rating': get_feat('rating')}
    return steam_feats


def binarize_nlp_features(nlp_features: Dict[str, int]) -> Dict[str, int]:
    """
    Binarize the NLP features.

    :param nlp_features: feature dictionary
    :type nlp_features: dict

    :returns: dictionary of features
    :rtype: dict
    """

    return dict(Counter(list(nlp_features)))


def bulk_extract_features(db: Collection,
                          game_cursor: Cursor,
                          reuse_nlp_feats: bool = True,
                          use_binarized_nlp_feats: bool = True,
                          lowercase_text: bool = True,
                          lowercase_cngrams: bool = False) -> Dict[str, int]:
    """
    Extract NLP features from reviews in the MongoDB database for a
    particular game/data partition in bulk and generate the feature
    dictionaries and other information for use in updating the
    database.

    :param db: MongoDB collection
    :type db: Collection
    :param game_cursor: a cursor on a MongoDB collection
    :type game_cursor: Cursor
    :param reuse_nlp_feats: reuse NLP features from database instead of
                            extracting them all over again (default:
                            True)
    :type reuse_nlp_feats: bool
    :param use_binarized_nlp_feats: use binarized NLP features
                                    (default: True)
    :type use_binarized_nlp_feats: bool
    :param lowercase_text: whether or not to lower-case the review
                           text (default: True)
    :type lowercase_text: bool
    :param lowercase_cngrams: whether or not to lower-case the
                              character n-grams (default: False)
    :type lowercase_cngrams: bool

    :yields: dictionary containing a feature dictionary and an
             `ObjectId` string to insert
    :ytype: dict
    """

    for game_doc in game_cursor:
        nlp_feats = None
        game_doc_get = game_doc.get
        review_text = game_doc_get('review')
        binarized_nlp_feats = game_doc_get('nlp_features_binarized', False)
        _id = game_doc_get('_id')

        # Extract NLP features by querying the database (if they are
        # available and the --reuse_features option was used or the ID
        # is in the list of IDs for reviews already collected);
        # otherwise, extract features from the review text directly
        # (and try to update the database)
        found_nlp_feats = False
        if (reuse_nlp_feats & ((use_binarized_nlp_feats & binarized_nlp_feats)
                                | (use_binarized_nlp_feats & (not binarized_nlp_feats)))):
            nlp_feats = get_nlp_features_from_db(db, _id)
            found_nlp_feats = True if nlp_feats else False

        extracted_anew = False
        if not found_nlp_feats:
            nlp_feats = extract_features(Review(review_text, lower=lowercase_text),
                                         lowercase_cngrams=lowercase_cngrams)
            extracted_anew = True

        # Make sure features get binarized if need be
        if (use_binarized_nlp_feats
            & (((not reuse_nlp_feats) | (not binarized_nlp_feats)) | extracted_anew)):
            nlp_feats = binarize_nlp_features(nlp_feats)

        # Yield a dictionary containing the data to be (possibly
        # altered and then) inserted
        if ((not found_nlp_feats) | (use_binarized_nlp_feats ^ binarized_nlp_feats)):
            yield dict(features=nlp_feats, _id=_id)
