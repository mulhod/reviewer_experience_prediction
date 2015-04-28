'''
@author: Matt Mulholland, Janette Martinez, Emily Olshefski
@date: 3/18/15

Script used to train models on datasets (or multiple datasets combined).
'''
import sys
import re
import pymongo
import argparse
from json import dumps
from os import listdir
from numpy import log2
from data import APPID_DICT
from spacy.en import English
from nltk.util import ngrams
from string import punctuation
from collections import Counter
from skll import run_configuration
from nltk.stem import SnowballStemmer
from configparser import ConfigParser
from os.path import join, dirname, realpath, abspath


class Review(object):
    '''
    Class for objects representing Reviews.
    '''

    # Original review text
    orig = None
    # Normalized review text
    norm = None
    # Number of hours the reviewer has played the game (float)
    hours_played = None
    # appid of the game (string ID code that Steam uses to represent the
    # game
    appid = None
    # Length of the original text (base-2 log)
    length = None
    # Attributes representing the word- and sentence-tokenized
    # representations of self.norm, consisting of a list of elements
    # corresponding to the identified sentences, which in turn consist of
    # a list of elements corresponding to the identified tokens, tags,
    # lemmas, respectively
    tokens = []
    tags = []
    #lemmas = []
    # Attribute representing the spaCy text annotations
    spaCy_annotations = []
    # Atrribute representing the named entities in the review
    #entities = []
    # Attribute representing the dependency labels features
    dep = Counter()
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
    

    def __init__(self, review_text, hours_played, game, appid, spaCy_nlp,
                 lower=True):
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
        :param lower: include lower-casing as part of the review text normalization
        :type lower: boolean
        '''

        self.orig = review_text
        self.hours_played = hours_played
        self.appid = appid

        # Generate attribute values
        self.length = log2(len(self.orig)) # Get base-2 log of the length of
            # the original version of the review text, not the normalized
            # version
        self.normalize(lower=lower)
        # Use spaCy to analyze the normalized version of the review text
        self.spaCy_annotations = spaCy_nlp(self.norm,
                                           tag=True,
                                           parse=True)
        self.get_token_features_from_spaCy()
        self.get_entities_from_spaCy()


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
        self.norm = r


    def get_token_features_from_spaCy(self):
        '''
        Get tokens-related features from spaCy's text annotations.
        '''

        for sent in self.spaCy_annotations.sents:
            # Get tokens
            self.tokens.append([t.orth_ for t in sent])
            # Get tags
            self.tags.append([t.tag_ for t in sent])
            # Get lemmas
            #self.lemmas.append([t.lemma_ for t in sent])
            # Get syntactic heads
            #self.heads.append([t.head.orth_ for t in sent])
            # Get dependency features
            for t in sent:
                #children = [c for c in t.children]
                #if children:
                #    for c in children:
                #        c = {"dep##{0.orth_}:{1.orth_}".format(t, c): 1}
                #        self.children.update(c)
                if t.n_lefts + t.n_rights:
                    fstring = "dep##{0.orth_}:{1.orth_}"
                    [self.dep.update({fstring.format(t, c): 1}) for c in
                     t.children if not c.tag_ in punctuation]


    def get_entities_from_spaCy(self):
        '''
        Get named entities from spaCy's text annotations.
        '''

        for entity in self.spaCy_annotations.ents:
            self.entities.append(dict(entity=entity.orth_,
                                      label=entity.label_))


def generate_ngram_fdist(sents, _min=1, _max=3):
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

    ngram_counter = Counter()
    for sent in sents:
        for i in range(_min, _max + 1):
            ngram_counter.update(list(ngrams(sent, i)))
    # For each n-gram, restructure the key as a string representation of the
    # feature and assign the restructured key-value mapping and delete the old
    # key
    for ngram in ngram_counter:
        ngram_counter['ngrams##{}'.format(' '.join(ngram))] = \
            ngram_counter[ngram]
        del ngram_counter[ngram]
    return ngram_counter


def generate_cngram_fdist(text, _min=2, _max=5, lower=False):
    '''
    Generate frequency distribution for the characters in the text.

    :param text: review text
    :type text: str
    :param _min: minimum value of n for character n-gram extraction
    :type _min: int
    :param _max: maximum value of n for character n-gram extraction
    :type _max: int
    :param lower: whether or not to lower-case the text (False by default)
    :type lower: boolean
    :returns: Counter
    '''

    cngram_counter = Counter()
    for i in range(_min, _max + 1):
        if lower:
            cngram_counter.update(list(ngrams(text.lower(), i)))
        else:
            cngram_counter.update(list(ngrams(text, i)))
    # For each character n-gram, restructure the key as a string
    # representation of the feature and assign the restructured key-value
    # mapping and delete the old key
    for ngram in cngram_counter:
        cngram_counter['cngrams##{}'.format(' '.join(ngram))] = \
            cngram_counter[ngram]
        del cngram_counter[ngram]
    return cngram_counter


def write_config_file(config_dict, path):
    '''
    This Creates a configparser config file from a dict and writes it to a file that can be read in by SKLL.  The dict should maps keys for the SKLL config sections to dictionaries of key-value pairs for each section.

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage='python train.py',
        description='Build a machine learning model based on the features ' \
                    'that are extracted from a set of reviews relating to a' \
                    'specific game or set of games.',
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
    parser.add_argument('--lowercase_text',
        help='make lower-casing part of the review text normalization step,' \
             ' which affects word n-gram-related features (defaults to True)',
        action='store_true',
        default=True)
    parser.add_argument('--lowercase_cngrams',
        help='lower-case the review text before extracting character n-gram' \
             ' features (defaults to False)',
        action='store_true',
        default=False)
    args = parser.parse_args()

    # Get paths to the project, data, working, and models directories
    project_dir = dirname(dirname(abspath(realpath(__file__))))
    data_dir = join(project_dir,
                    'data')
    working_dir = join(project_dir,
                      'working')
    models_dir = join(project_dir,
                      'models')
    cfg_dir = join(project_dir,
                   'config')
    logs_dir = join(project_dir,
                   'logs')
    sys.stderr.write('project directory: {}\ndata directory: {}\nworking ' \
                     'directory: {}\nmodels directory: {}\nconfiguration ' \
                     'directory: {}\nlogs directory: {}\n'.format(project_dir,
                                                                  data_dir,
                                                                  working_dir,
                                                                  models_dir,
                                                                  cfg_dir,
                                                                  logs_dir))

    # Make sure that, if --combine is being used, there is also a file prefix
    # being passed in via --combined_model_prefix for the combined model
    if args.combine and not args.combined_model_prefix:
        sys.exit('ERROR: When using the --combine flag, you must also ' \
                 'specify a model prefix, which can be passed in via the ' \
                 '--combined_model_prefix option argument. Exiting.\n')

    # Establish connection to MongoDB database
    connection_string = 'mongodb://localhost:27017'
    try:
        connection = pymongo.MongoClient(connection_string)
    except pymongo.errors.ConnectionFailure as e:
        sys.exit('ERROR: Unable to connecto to Mongo server at ' \
                 '{}'.format(connection_string))
    db = connection['reviews_project']
    reviewdb = db['reviews']

    # Initialize an English-language spaCy NLP analyzer instance
    spaCy_nlp = English()

    # Get list of games
    game_files = []
    if args.game_files == "all":
        game_files = [f for f in listdir(data_dir) if f.endswith('.txt')]
    else:
        game_files = args.game_files.split(',')

    if args.combine:
        sys.stderr.write('Training a combined model with training data from' \
                         ' the following games: {}\n'.format(
                                                       ', '.join(game_files)))

        # Initialize empty list for holding all of the feature dictionaries
        # from each review in each game and then extract features from each
        # game's training data
        feature_dicts = []
        for game_file in game_files:
            # Get the training reviews for this game from the Mongo
            # database
            game = game_file[:-4]
            sys.stderr.write('Extracting features from the training data ' \
                             'for {}...\n'.format(game))
            appid = APPID_DICT[game]
            game_docs = list(reviewdb.find({'game': game,
                                            'partition': 'training'}))

            # Iterate over all training documents for the given game
            for game_doc in game_docs:

                # Get the game_doc ID, the hours played value, and the
                # original review text from the game_doc
                _id = str(game_doc['_id'])
                hours = str(game_doc['hours'])
                review_text = game_doc['review']

                # Instantiate a Review object
                _Review = Review(review_text,
                                 hours,
                                 game,
                                 appid,
                                 spaCy_nlp,
                                 lower=args.lowercase_text)

                # Extract features from the review text
                game_features = Counter()

                # Extract n-gram features
                ngrams_counter = generate_ngram_fdist(_Review.tokens)
                game_features.update(ngrams_counter)

                # Extract character n-gram features
                if args.lowercase_cngrams:
                    cngrams_counter = \
                        generate_cngram_fdist(_Review.orig.lower())
                else:
                    cngrams_counter = generate_cngram_fdist(_Review.orig)
                game_features.update(cngrams_counter)

                # Get the length feature
                length_feature = {'length##{}'.format(_Review.length): 1}
                game_features.update(length_feature)

                # Get the syntactic dependency features
                game_features.update(_Review.dep)

                # Append a feature dictionary for the review to feature_dicts
                feature_dicts.append({'id': _id,
                                      'y': hours,
                                      'x': game_features})

                # Update Mongo database game doc with new key "features",
                # which will be mapped to game_features
                reviewdb.update_one({'_id': _id}, {'features': game_features})

        # Write .jsonlines file
        jsonlines_filename = '{}.jsonlines'.format(combined_model_prefix)
        jsonlines_filepath = join(working_dir,
                                  jsonlines_filename)
        sys.stderr.write('Writing {} to working directory...'.format(
                                                          jsonlines_filename))
        with open(jsonlines_filepath, 'w') as jsonlines_file:
            [jsonlines_file.write('{}\n'.format(json.dumps(fd)).encode(
             'utf-8')) for fd in feature_dicts]

        # Set up SKLL job arguments
        learner_name = 'RescaledSVR'
        param_grid_list = [{'C': [10.0 ** x for x in range(-3, 4)]}]
        grid_objective = 'quadratic_weighted_kappa'

        # Create a template for the SKLL config file
        # Note that all values must be strings
        cfg_dict_base = {"General": {},
                         "Input": {"train_location": working_dir,
                                   "ids_to_floats": "False",
                                   "label_col": "y",
                                   "featuresets": json.dumps(
                                                   [[combined_model_prefix]]),
                                   "suffix": '.jsonlines',
                                   "learners": json.dumps([learner_name])
                                   },
                         "Tuning": {"feature_scaling": "none",
                                    "grid_search": "True",
                                    "min_feature_count": "1",
                                    "objective": grid_objective,
                                    "param_grids": json.dumps(
                                                           [param_grid_list]),
                                    },
                         "Output": {"probability": "False",
                                    "log": join(logs_dir,
                                                '{}.log'.format(
                                                       combined_model_prefix))
                                    }
                         }

        # Set up the job for training the model
        sys.stderr.write('Generating configuration file...')
        cfg_filename = '{}.cfg'.format(combined_model_prefix)
        cfg_filepath = join(cfg_dir,
                            cfg_filename)
        cfg_dict_base["General"]["task"] = "train"
        cfg_dict_base["General"]["experiment_name"] = combined_model_prefix
        cfg_dict_base["Output"]["models"] = models_dir
        write_config_file(cfg_dict_base,
                          cfg_filepath)

        # Run the SKLL configuration, producing a model file
        sys.stderr.write('Training combined model...\n')
        run_configuration(cfg_file)
    else:
        for game_file in game_files:
            game = game_file[:-4]
            sys.stderr.write('Training a model with training data from {}' \
                             '...\n'.format(game))

            # Initialize empty list for holding all of the feature
            # dictionaries from each review and then extract features from all
            # reviews
            feature_dicts = []

            # Get the training reviews for this game from the Mongo
            # database
            sys.stderr.write('Extracting features from the training data ' \
                             'for {}...\n'.format(game))
            appid = APPID_DICT[game]
            game_docs = list(reviewdb.find({'game': game,
                                            'partition': 'training'}))

            # Iterate over all training documents for the given game
            for game_doc in game_docs:

                # Get the game_doc ID, the hours played value, and the
                # original review text from the game_doc
                _id = str(game_doc['_id'])
                hours = str(game_doc['hours'])
                review_text = game_doc['review']

                # Instantiate a Review object
                _Review = Review(review_text,
                                 hours,
                                 game,
                                 appid,
                                 spaCy_nlp,
                                 lower=args.lowercase_text)

                # Extract features from the review text
                game_features = Counter()

                # Extract n-gram features
                ngrams_counter = generate_ngram_fdist(_Review.tokens)
                game_features.update(ngrams_counter)

                # Extract character n-gram features
                if args.lowercase_cngrams:
                    cngrams_counter = \
                        generate_cngram_fdist(_Review.orig.lower())
                else:
                    cngrams_counter = generate_cngram_fdist(_Review.orig)
                game_features.update(cngrams_counter)

                # Get the length feature
                length_feature = {'length##{}'.format(_Review.length): 1}
                game_features.update(length_feature)

                # Get the syntactic dependency features
                game_features.update(_Review.dep)

                # Append a feature dictionary for the review to feature_dicts
                feature_dicts.append({'id': _id,
                                      'y': hours,
                                      'x': game_features})

                # Update Mongo database game doc with new key "features",
                # which will be mapped to game_features
                reviewdb.update_one({'_id': _id}, {'features': game_features})

            # Write .jsonlines file
            jsonlines_filename = '{}.jsonlines'.format(game)
            jsonlines_filepath = join(working_dir,
                                      jsonlines_filename)
            sys.stderr.write('Writing {} to working directory...'.format(
                                                          jsonlines_filename))
            with open(jsonlines_filepath, 'w') as jsonlines_file:
                [jsonlines_file.write('{}\n'.format(json.dumps(fd)).encode(
                 'utf-8')) for fd in feature_dicts]

            # Set up SKLL job arguments
            learner_name = 'RescaledSVR'
            param_grid_list = [{'C': [10.0 ** x for x in range(-3, 4)]}]
            grid_objective = 'quadratic_weighted_kappa'

            # Create a template for the SKLL config file
            # Note that all values must be strings
            cfg_dict_base = {"General": {},
                             "Input": {"train_location": working_dir,
                                       "ids_to_floats": "False",
                                       "label_col": "y",
                                       "featuresets": json.dumps([[game]]),
                                       "suffix": '.jsonlines',
                                       "learners": json.dumps([learner_name])
                                       },
                             "Tuning": {"feature_scaling": "none",
                                        "grid_search": "True",
                                        "min_feature_count": "1",
                                        "objective": grid_objective,
                                        "param_grids": json.dumps(
                                                           [param_grid_list]),
                                        },
                             "Output": {"probability": "False",
                                        "log": join(logs_dir,
                                                    '{}.log'.format(game))
                                        }
                             }

            # Set up the job for training the model
            sys.stderr.write('Generating configuration file...')
            cfg_filename = '{}.cfg'.format(combined_model_prefix)
            cfg_filepath = join(cfg_dir,
                                cfg_filename)
            cfg_dict_base["General"]["task"] = "train"
            cfg_dict_base["General"]["experiment_name"] = \
                combined_model_prefix
            cfg_dict_base["Output"]["models"] = models_dir
            write_config_file(cfg_dict_base,
                              cfg_filepath)

            # Run the SKLL configuration, producing a model file
            sys.stderr.write('Training model for {}...\n'.format(game))
            run_configuration(cfg_file)

    sys.stderr.write('Model training complete.\n')