'''
@author Matt Mulholland, Janette Martinez, Emily Olshefski
@date 05/05/2015

Module of functions/classes related to feature extraction, model-building,
ARFF file generation, etc.
'''
import numpy as np
from math import ceil
from json import dumps
from os.path import join
from nltk.util import ngrams
from string import punctuation
from re import (sub,
                IGNORECASE)
from collections import Counter
from configparser import ConfigParser

class Review(object):
    '''
    Class for objects representing Reviews.
    '''

    # Normalized review text
    norm = None
    # appid of the game (string ID code that Steam uses to represent the
    # game
    appid = None
    # Attribute whose value determines whether or not the review text will
    # be lower-cased as part of the normalization step
    lower = None
    # Attribute consisting of the identified sentences, which, in turn
    # consist of the identified tokens
    tokens = []
    # Attributes representing the spaCy text annotations
    spaCy_annotations = None
    spaCy_sents = None
    # Attributes representing the cluster IDs, "repvecs" (representation
    # vectors), and "probs" (log probabilities) corresponding to tokens
    cluster_id_counter = Counter()
    repvecs = []
    probs = []


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
        :param lower: include lower-casing as part of the review text
                      normalization step
        :type lower: boolean
        '''

        self.orig = review_text
        self.hours_played = hours_played
        self.appid = appid
        self.lower = lower

        # Generate attribute values
        self.length = ceil(np.log2(len(self.orig))) # Get base-2 log of the
            # length of the original version of the review text, not the
            # normalized version
        self.normalize()
        # Use spaCy to analyze the normalized version of the review text
        self.spaCy_annotations = spaCy_nlp(self.norm,
                                           tag=True,
                                           parse=True)
        self.spaCy_sents = []
        for _range in self.spaCy_annotations.sents:
            self.spaCy_sents.append([self.spaCy_annotations[i]
                                     for i in range(*_range)])
        self.get_token_features_from_spaCy()


    def normalize(self):
        '''
        Perform text preprocessing, i.e., lower-casing, etc., to generate the
        norm attribute.
        '''

        # Lower-case text if self.lower is True
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
            # Get clusters
            self.cluster_id_counter.update([t.cluster for t in sent])
            # Get "probs"
            self.probs.append([t.prob_ for t in sent])
            # Get repvecs
            self.repvecs.append([t.repvec for t in sent])


def extract_features_from_review(_review, lowercase_cngrams=False):
    '''
    Extract word/character n-gram, length, cluster ID, average cosine
    similarity between word representation vectors, and syntactic dependency
    features from a Review object and return as dictionary where each feature
    is represented as a key:value mapping in which the key is a string
    representation of the feature (e.g. "the dog" for an example n-gram
    feature, "th" for an example character n-gram feature, "c667" for an
    example cluster feature, "mean_cos_sim" mapped to a float in the range 0
    to 1 for the average cosine similarity feature, and "step:VMOD:forward"
    for an example syntactic dependency feature) and the value is the
    frequency with which that feature occurred in the review.

    :param _review: object representing the review
    :type _review: Review object
    :param lowercase_cngrams: whether or not to lower-case the review text
                              before extracting character n-grams
    :type lowercase_cngrams: boolean (False by default)
    :returns: dict
    '''

    def generate_ngram_fdist(sents, _min=1, _max=2):
        '''
        Generate frequency distribution for the tokens in the text.

        :param sents: list of lists of tokens that can be chopped up into
                      n-grams.
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
            for i in range(_min,
                           _max + 1):
                ngram_counter.update(list(ngrams(sent, i)))

        # Re-represent keys as string representations of specific features
        # of the feature class "ngrams"
        for ngram in list(ngram_counter):
            ngram_counter[' '.join(ngram)] = ngram_counter[ngram]
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
        for i in range(_min,
                       _max + 1):
            cngram_counter.update(list(ngrams(text, i)))

        # Re-represent keys as string representations of specific features
        # of the feature class "cngrams" (and set all values to 1 if binarize
        # is True)
        for cngram in list(cngram_counter):
            cngram_counter[''.join(cngram)] = cngram_counter[cngram]
            del cngram_counter[cngram]

        return cngram_counter


    def generate_cluster_fdist(cluster_counter):
        '''
        Generate frequency distribution of the cluster IDs corresponding to
        the tokens in a text.

        :param text: frequency distribution of cluster IDs (int)
        :type text: Counter
        :returns: Counter
        '''

        for cluster_id, freq in cluster_counter.items():
            del cluster_counter[cluster_id]
            cluster_counter['c{}'
                            .format(cluster_id)] = freq

        return cluster_counter


    def calculate_mean_cos_sim(repvecs):
        '''
        Calcualte the mean cosine similarity between all pairwise cosine
        similarity metrics between two words.

        :param repvecs: list of 1-dimensinal representation vectors
        :type repvecs: list of 1-dimensional np.ndarray with dtype==np.float64
        '''

        cos_sims = []
        while i < len(repvecs):
            j = 0
            repvec = repvecs[i]
            if i == 0:
                rem_repvecs = repvecs[1:]
            elif i != len(repvecs) - 1:
                rem_repvecs = repvecs[:i] + repvecs[i + 1:]
            else:
                rem_repvecs = repvecs[:-1]
            while j < len(rem_repvecs):
                cos_sims.append(repvec.dot(rem_repvecs[j].T)
                                /np.linalg.norm(repvec)
                                /np.linalg.norm(rem_repvecs[j]))
                j += 1
            i += 1
        return {'mean_cos_sim': (np.sum([cos_sim for cos_sim in cos_sims
                                        if not np.isnan(cos_sim)])
                                 /len(cos_sims))}


    def generate_dep_features(spaCy_sents):
        '''
        Generate syntactic dependency features from spaCy text annotations and
        represent the features as token (lemma) + dependency type + child
        token (lemma).

        :param spaCy_annotations: lists of spaCy Token instances
        :type spaCy_annotations: list of list of spaCy.tokens.Tokens instances
        :returns: Counter object representing a frequency distribution of
                  syntactic dependency features
        '''

        # Make emtpy Counter
        dep_counter = Counter()

        # Iterate through spaCy annotations for each sentence and then for
        # each token
        for s in spaCy_sents:
            for t in s:
                # If the number of children to the left and to the right
                # of the token add up to a value that is not zero, then
                # get the children and make dependency features with
                # them
                if t.n_lefts + t.n_rights:
                    fstr = "{0.lemma_}:{0.dep_}:{1.lemma_}"
                    [dep_counter.update({fstr.format(t,
                                                     c): 1})
                     for c in t.children
                     if not c.tag_ in punctuation]

        return dep_counter

    # Extract features
    features = Counter()

    # Get the length feature
    # Note: This feature will always be mapped to a frequency of 1 since
    # it exists for every single review and, thus, a review of this length
    # being mapped to the hours played value that it is mapped to has
    # occurred once.
    features.update({str(_review.length): 1})

    # Extract n-gram features
    features.update(generate_ngram_fdist(_review.tokens))

    # Extract character n-gram features
    orig = _review.orig.lower() if lowercase_cngrams else _review.orig
    features.update(generate_cngram_fdist(orig))

    # Convert cluster ID values into useable features
    features.update(generate_cluster_fdist(_review.cluster_id_counter))

    # Calculate the mean cosine similarity across all word-pairs
    features.update(calculate_mean_cos_sim(_review.repvecs))

    # Generate the syntactic dependency features
    features.update(generate_dep_features(_review.spaCy_sents))

    return dict(features)


def generate_config_file(exp_name, feature_set_name, learner_name, obj_func,
                         project_dir_path, cfg_file_path):
    '''
    This Creates a configparser config file from a dict and writes it to a
    file that can be read in by SKLL.  The dict should map keys for the SKLL
    config sections to dictionaries of key-value pairs for each section.

    :param exp_name: name/ID associated with model/experiment
    :type exp_name: str
    :param feature_set_name: name of feature set (should be the name of the
                             corresponding JSONLINES file in the 'working'
                             directory minus the extension)
    :type feature_set_name: str
    :param learner_name: name of machine learning algorithm
    :type learner_name: str
    :param obj_func: name of objective function
    :type obj_func: str
    :param project_dir_path: path to main project directory
    :type project_dir_path: str
    :param cfg_file_path: path to configuration file
    :type cfg_filename: str
    :returns: None
    '''

    # Create base config file and then add specific attributes to it
    # afterwards
    cfg_dict = {'General': {},
                'Input': {'ids_to_floats': 'False',
                          'label_col': 'y',
                          'suffix': '.jsonlines'},
                'Tuning': {'feature_scaling': 'none',
                           'grid_search': 'True',
                           'min_feature_count': '1'},
                'Output': {'probability': 'False'}}

    cfg_dict['General']['task'] = 'train'
    cfg_dict['General']['experiment_name'] = exp_name
    cfg_dict['Output']['models'] = join(project_dir_path,
                                        'models')
    cfg_dict['Output']['log'] = join(project_dir_path,
                                     'logs',
                                     '{}.log'.format(exp_name))
    cfg_dict['Input']['train_directory'] = join(project_dir_path,
                                                'working')
    cfg_dict['Input']['featuresets'] = dumps([[feature_set_name]])
    cfg_dict['Input']['learners'] = dumps([learner_name])
    cfg_dict['Tuning']['objective'] = obj_func
    if learner_name == 'RescaledSVR':
        param_grid_list = [{'C': [10.0 ** x for x in range(-3, 4)]}]
        cfg_dict['Tuning']['param_grids'] = dumps([param_grid_list])

    # Create ConfigParser instance and populate it with values from
    # cfg_dict
    cfg = ConfigParser()
    for section_name, section_dict in cfg_dict.items():
        cfg.add_section(section_name)
        [cfg.set(section_name,
                 key,
                 val)
         for key, val in section_dict.items()]

    # Write the file to the provided destination path
    with open(cfg_file_path,
              'w') as config_file:
        cfg.write(config_file)


def make_confusion_matrix(x_true, y_pred, continuous=True):
    '''
    Return confusion matrix with n rows/columns where n is equal to the number
    of unique data-points (or points on a scale, if continuous).

    :param x_true: np.array of "true" labels
    :type x_true: 1-dimensional np.array with dtype=np.int32
    :param y_pred: np.array of predicted labels
    :type y_pred: 1-dimensional numpy.array with dtype=np.int32
    :param continuous: if data-points/labels form a continuous scale of
                       natural numbers
    :type continuous: boolean
    :returns: dictionary consisting of 1) a 'data' key mapped to the confusion
              matrix itself (a 2-dimensional np.array with
              dtype=np.int32) and 2) a 'string' key mapped to a string
              representation of the confusion matrix
    '''

    # Get the range of labels/data-points
    label_set = set(x_true)
    if continuous:
        _range = list(range(min(label_set),
                            max(label_set) + 1))
    else:
        _range = sorted(label_set)

    # Compute the confusion matrix
    rows = []
    cdef int i
    cdef int j
    for i, row_val in enumerate(_range):
        row = []
        for j, col_val in enumerate(_range):
            row.append(sum([1 for (x, y)
                            in zip(x_true,
                                   y_pred) if x == row_val
                                              and y == col_val]))
        rows.append(row)

    conf_matrix = np.array(rows,
                           dtype=np.int32)

    # Make string representations of the rows in the confusion matrix
    conf_matrix_rows = ['\t{}'
                        .format('\t'
                                .join(['_{}_'.format(val)
                                       for val in _range]))]
    cdef int k
    for k, row_val in enumerate(_range):
        conf_matrix_rows.append('_{}_\t{}'
                                .format(row_val,
                                        '\t'
                                        .join([str(val) for val
                                               in conf_matrix[k]])))

    return dict(data=conf_matrix,
                string='\n'.join(conf_matrix_rows))
