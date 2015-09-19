'''
@author Matt Mulholland, Janette Martinez, Emily Olshefski
@date 05/05/2015

Module of functions/classes related to feature extraction, model-building,
ARFF file generation, etc.
'''
import logging
logger = logging.getLogger()
logwarn = logger.warning
logerr = logger.error
import numpy as np
from sys import exit
from re import (sub,
                IGNORECASE)
from math import ceil
from time import sleep
from json import (dumps,
                  loads)
from os.path import join
from nltk.util import ngrams
from spacy.en import English
spaCy_nlp = English()
from string import punctuation
from collections import Counter
from skll.metrics import (kappa,
                          pearson)
from itertools import combinations
from util.mongodb import (update_db,
                          create_game_cursor)
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
    # Attribute representing the cluster IDs corresponding to tokens
    cluster_id_counter = None

    def __init__(self, review_text, lower=True):
        '''
        Initialization method.

        :param review_text: review text
        :type review_text: str
        :param game: name of game
        :type game: str
        :param lower: include lower-casing as part of the review text
                      normalization step
        :type lower: boolean
        '''

        # Get review text and lower-casing attributes
        self.orig = review_text
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
        spaCy_sents_append = self.spaCy_sents.append
        for _range in self.spaCy_annotations.sents:
            spaCy_sents_append([self.spaCy_annotations[i]
                                for i in range(*_range)])
        self.get_token_features_from_spaCy()

    def normalize(self):
        '''
        Perform text preprocessing, i.e., lower-casing, etc., to generate the
        norm attribute.
        '''

        # Lower-case text if self.lower is True
        r = self.orig.lower() if self.lower else self.orig

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

        lemma_set = set()
        cluster_ids = []
        tokens_append = self.tokens.append
        lemma_set_update = lemma_set.update
        cluster_ids_extend = cluster_ids.extend
        for sent in self.spaCy_sents:
            # Get tokens
            tokens_append([t.orth_ for t in sent])
            # Generate set of lemmas (for use in the average cosine similarity
            # calculation)
            lemma_set_update([t.lemma_ for t in sent])
            # Get clusters
            cluster_ids_extend([t.cluster for t in sent])
            # Get "probs"
            #self.probs.append([t.prob_ for t in sent])
        self.cluster_id_counter = dict(Counter(cluster_ids))


def extract_features_from_review(_review, lowercase_cngrams=False):
    '''
    Extract word/character n-gram, length, cluster ID, number of tokens
    corresponding to represenation vectors consisting entirely of zeroes,
    average cosine similarity between word representation vectors, and
    syntactic dependency features from a Review object and return as
    dictionary where each feature is represented as a key:value mapping in
    which the key is a string representation of the feature (e.g. "the dog"
    for an example n-gram feature, "th" for an example character n-gram
    feature, "c667" for an example cluster feature, "mean_cos_sim" mapped to a
    float in the range 0 to 1 for the average cosine similarity feature, and
    "step:VMOD:forward" for an example syntactic dependency feature) and the
    value is the frequency with which that feature occurred in the review.

    :param _review: object representing the review
    :type _review: Review object
    :param lowercase_cngrams: whether or not to lower-case the review text
                              before extracting character n-grams
    :type lowercase_cngrams: boolean (False by default)
    :returns: dict
    '''

    def generate_ngram_fdist(_min=1, _max=2):
        '''
        Generate frequency distribution for the tokens in the text.

        :param _min: minimum value of n for n-gram extraction
        :type _min: int
        :param _max: maximum value of n for n-gram extraction
        :type _max: int
        :returns: Counter
        '''

        # Make emtpy Counter
        ngram_counter = Counter()
        ngram_counter_update = ngram_counter.update

        # Count up all n-grams
        for sent in _review.tokens:
            for i in range(_min,
                           _max + 1):
                ngram_counter_update(list(ngrams(sent,
                                                 i)))

        # Re-represent keys as string representations of specific features
        # of the feature class "ngrams"
        for ngram in list(ngram_counter):
            ngram_counter[' '.join(ngram)] = ngram_counter[ngram]
            del ngram_counter[ngram]

        return ngram_counter


    def generate_cngram_fdist(_min=2, _max=5):
        '''
        Generate frequency distribution for the characters in the text.

        :param _min: minimum value of n for character n-gram extraction
        :type _min: int
        :param _max: maximum value of n for character n-gram extraction
        :type _max: int
        :returns: Counter
        '''

        # Make emtpy Counter
        cngram_counter = Counter()
        cngram_counter_update = cngram_counter.update

        # Count up all character n-grams
        for i in range(_min,
                       _max + 1):
            cngram_counter_update(list(ngrams(_review.orig.lower()
                                                  if lowercase_cngrams
                                                  else _review.orig,
                                              i)))

        # Re-represent keys as string representations of specific features
        # of the feature class "cngrams" (and set all values to 1 if binarize
        # is True)
        for cngram in list(cngram_counter):
            cngram_counter[''.join(cngram)] = cngram_counter[cngram]
            del cngram_counter[cngram]

        return cngram_counter


    def generate_cluster_fdist():
        '''
        Convert cluster ID frequency distribution to a frequency distribution
        where the keys are strings representing "cluster" features (rather
        than just a number, the cluster ID).

        :returns: Counter
        '''

        cluster_fdist = _review.cluster_id_counter
        for cluster_id, freq in list(cluster_fdist.items()):
            del cluster_fdist[cluster_id]
            cluster_fdist['cluster{}'.format(cluster_id)] = freq

        return cluster_fdist


    def generate_dep_features():
        '''
        Generate syntactic dependency features from spaCy text annotations and
        represent the features as token (lemma) + dependency type + child
        token (lemma).

        :returns: Counter
        '''

        # Make emtpy Counter
        dep_counter = Counter()
        dep_counter_update = dep_counter.update

        # Iterate through spaCy annotations for each sentence and then for
        # each token
        for sent in _review.spaCy_sents:
            for t in sent:
                # If the number of children to the left and to the right
                # of the token add up to a value that is not zero, then
                # get the children and make dependency features with
                # them
                if t.n_lefts + t.n_rights:
                    [dep_counter_update({'{0.lemma_}:{0.dep_}:{1.lemma_}'
                                         .format(t,
                                                 c): 1})
                     for c in t.children
                     if not c.tag_ in punctuation]

        return dep_counter

    # Extract features
    feats = {}

    feats_update = feats.update
    # Get the length feature
    feats_update({str(_review.length): 1})

    # Extract n-gram features
    feats_update(generate_ngram_fdist())

    # Extract character n-gram features
    feats_update(generate_cngram_fdist())

    # Convert cluster ID values into useable features
    feats_update(generate_cluster_fdist())

    # Generate the syntactic dependency features
    feats_update(generate_dep_features())

    return feats


def get_nlp_features_from_db(db, _id):
    '''
    Collect the NLP features from the Mongo database collection for a given
    review and return the decoded value.

    :param db: Mongo reviews collection
    :type db: pymongo.collection.Collection object
    :param _id: database document's Object ID
    :type _id: pymongo.bson.objectid.ObjectId
    :returns: dict if features were found; None otherwise
    '''

    nlp_feats_doc = db.find_one({'_id': _id},
                                {'_id': 0,
                                 'nlp_features': 1})
    return (loads(nlp_feats_doc.get('nlp_features')) if nlp_feats_doc
                                                     else None)


def get_steam_features_from_db(get_feat):
    '''
    Get features collected from Steam (i.e., the non-NLP features).

    :param get_feat: built-in method get of dictionary object representing a
                     single Mongo database document
    :type get_feat: method/function
    :returns: dict
    '''

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


def binarize_nlp_features(nlp_features):
    '''
    Binarize the NLP features.

    :param nlp_features: feature dictionary
    :type nlp_features: dict
    :returns: dict
    '''

    return dict(Counter(list(nlp_features)))


def extract_nlp_features_into_db(db, data_partition, game_id,
                                 reuse_nlp_feats=True,
                                 use_binarized_nlp_feats=True,
                                 lowercase_text=True,
                                 lowercase_cngrams=False):
    '''
    Extract NLP features from reviews in the Mongo database and write the
    features to the database if features weren't already added and
    reuse_nlp_feats is false).

    :param db: a Mongo DB collection client
    :type db: pymongo.collection.Collection
    :param data_partition: 'training', 'test', etc. (must be valid value for
                           'partition' key of review collection in Mongo
                           database); alternatively, can be the value "all"
                           for all partitions
    :type data_partition: str
    :param game_id: game ID
    :type game_id: str
    :param reuse_nlp_feats: reuse NLP features from database instead of
                            extracting them all over again
    :type reuse_nlp_feats: boolean
    :param use_binarized_nlp_feats: use binarized NLP features
    :type use_binarized_nlp_feats: boolean
    :param lowercase_text: whether or not to lower-case the review text
    :type lowercase_text: boolean
    :param lowercase_cngrams: whether or not to lower-case the character
                              n-grams
    :type lowercase_cngrams: boolean
    :returns: None
    '''

    db_update = db.update

    # Create cursor object and set batch_size to 1,000
    cdef int batch_size = 1000
    with create_game_cursor(db,
                            game_id,
                            data_partition,
                            batch_size) as game_cursor:
        for game_doc in game_cursor:
            game_doc_get = game_doc.get
            review_text = game_doc_get('review')
            binarized_nlp_feats = game_doc_get('nlp_features_binarized',
                                               False)
            _id = game_doc_get('_id')

            # Extract NLP features by querying the database (if they are
            # available and the --reuse_features option was used or the ID is
            # in the list of IDs for reviews already collected); otherwise,
            # extract features from the review text directly (and try to
            # update the database)
            found_nlp_feats = False
            if (reuse_nlp_feats
                & ((use_binarized_nlp_feats & binarized_nlp_feats)
                    | (use_binarized_nlp_feats & (not binarized_nlp_feats)))):
                nlp_feats = get_nlp_features_from_db(db,
                                                     _id)
                found_nlp_feats = True if nlp_feats else False

            extracted_anew = False
            if not found_nlp_feats:
                nlp_feats = extract_features_from_review(
                                Review(review_text
                                       lower=lowercase_text),
                                lowercase_cngrams=lowercase_cngrams)
                extracted_anew = True

            # Make sure features get binarized if need be
            if (use_binarized_nlp_feats
                & (((not reuse_nlp_feats)
                    | (not binarized_nlp_feats))
                   | extracted_anew)):
                nlp_feats = binarize_nlp_features(nlp_feats)

            # Update Mongo database game doc with new key "nlp_features",
            # update/create a "nlp_features_binarized" key to store a value
            # indicating whehter or not the NLP features were binarized or
            # not, and update/create an "id_string" key for storing the string
            # represenation of the ID
            if ((not found_nlp_feats)
                | (use_binarized_nlp_feats ^ binarized_nlp_feats)):
                update_db(db_update,
                          _id,
                          nlp_feats,
                          use_binarized_nlp_feats)


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


def write_predictions_to_file(path, game_id, model_id, preds_rows):
    '''
    Write review and predictions data to file.

    :param path: destination path for predictions file
    :type path: str
    :param game_id: game ID
    :type game_id: str
    :param model_id: model ID
    :type model_id: str
    :param preds_rows: list of ID, review text, hours, and prediction values
    :type preds_rows: list
    :returns: None
    '''

    import csv
    with open(join(path,
                   '{}.test_{}_predictions.csv'.format(game_id,
                                                       model_id)),
              'w') as preds_file:
        preds_file_csv = csv.writer(preds_file,
                                    delimiter=',')
        preds_file_csv_writerow = preds_file_csv.writerow
        preds_file_csv_writerow(['id',
                                 'review',
                                 'hours_played',
                                 'prediction'])
        [preds_file_csv_writerow([_id,
                                  review,
                                  hours_value,
                                  pred])
         for _id, review, hours_value, pred in preds_rows]


def write_results_file(results_dir, game_id, model_id, hours_vals, preds):
    '''
    Write evaluation report to file.

    :param results_dir: path to results directory
    :type results_dir: str
    :param game_id: game ID
    :type game_id: str
    :param model_id: model ID
    :type model_id: str
    :param hours_vals: hours played values
    :type hours_vals: list of float
    :param preds: prediction labels
    :type preds: list of int
    :returns: None
    '''

    with open(join(results_dir,
                   '{}.test_{}_results.txt'.format(game_id,
                                                   model_id)),
              'w') as results_file:
        results_file_write = results_file.write
        results_file_write('Results Summary\n\n')
        results_file_write('- Game: {}\n'.format(game_id))
        results_file_write('- Model: {}\n\n'.format(model_id))
        results_file_write('Evaluation Metrics\n\n')
        results_file_write('Kappa: {}\n'.format(kappa(hours_vals,
                                                      preds)))
        results_file_write('Kappa (allow off-by-one): {}\n'
                           .format(kappa(hours_vals,
                                         preds,
                                         allow_off_by_one=True)))
        results_file_write('Pearson: {}\n\n'
                           .format(pearson(hours_vals,
                                           preds)))
        results_file_write('Confusion Matrix\n')
        results_file_write('(predicted along top, actual along side)\n\n')
        results_file_write('{}\n'
                           .format(make_confusion_matrix(hours_vals,
                                                         preds)['string']))
