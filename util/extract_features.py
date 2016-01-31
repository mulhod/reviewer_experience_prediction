"""
:author: Matt Mulholland (mulhodm@gmail.com)
:date: September 18, 2015

Script used to extract features for review documents in the MongoDB
database.
"""
import logging
from os import makedirs
from os.path import (join,
                     dirname,
                     realpath,
                     splitext)

from argparse import (ArgumentParser,
                      ArgumentDefaultsHelpFormatter)

from src import (log_dir,
                 formatter,
                 project_dir)

logger = logging.getLogger(__name__)
logging_debug = logging.DEBUG
logger.setLevel(logging_debug)
loginfo = logger.info
logdebug = logger.debug
logerr = logger.error
logwarn = logger.warning
sh = logging.StreamHandler()
sh.setLevel(logging_debug)
sh.setFormatter(formatter)
logger.addHandler(sh)


def main():
    parser = ArgumentParser(usage='python extract_features.py --game_files '
                                  'GAME_FILE1,GAME_FILE2,...[ OPTIONS]',
        description='Extract features and add them to the Mongo database.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    _add_arg = parser.add_argument
    _add_arg('--game_files',
             help='Comma-separated list of file-names or "all" for all of the'
                  ' files (the game files should reside in the "data" '
                  'directory; the .jsonlines suffix is not necessary, but the'
                  ' file-names should be exact matches otherwise).',
             type=str,
             required=True)
    _add_arg('--do_not_binarize_features',
             help='Do not make all non-zero feature frequencies equal to 1.',
             action='store_true',
             default=False)
    _add_arg('--do_not_lowercase_text',
             help='Do not make lower-casing part of the review text '
                  'normalization step, which affects word n-gram-related '
                  'features.',
             action='store_true',
             default=False)
    _add_arg('--lowercase_cngrams',
             help='Lower-case the review text before extracting character '
                  'n-gram features.',
             action='store_true',
             default=False)
    _add_arg('--partition',
             help='Data partition, i.e., "training", "test", etc. Value must '
                  'be a valid partition set name in the Mongo database. '
                  'Alternatively, the value "all" can be used to include all '
                  'partitions.',
             type=str,
             default='all')
    _add_arg('--do_not_reuse_extracted_features',
             help="Don't make use of previously-extracted features present in"
                  " the Mongo database and instead replace them if they are.",
             action='store_true',
             default=False)
    _add_arg('-dbhost', '--mongodb_host',
             help='Host that the MongoDB server is running on.',
             type=str,
             default='localhost')
    _add_arg('-dbport', '--mongodb_port',
             help='Port that the MongoDB server is running on.',
             type=int,
             default=27017)
    _add_arg('--update_batch_size', '-batch_size',
             help='Size of each batch for the bulk updates.',
             type=int,
             default=100)
    _add_arg('-log', '--log_file_path',
             help='Path to feature extraction log file.',
             type=str,
             default=join(log_dir, 'replog_extract_features.txt'))
    args = parser.parse_args()

    # Imports
    from pymongo.errors import (BulkWriteError,
                                ConnectionFailure)
    from src import log_format_string
    from src.mongodb import (connect_to_db,
                             bulk_extract_features_and_update_db)
    from src.datasets import get_game_files

    # Make local copies of arguments
    game_files = args.game_files
    binarize = not args.do_not_binarize_features
    reuse_features = not args.do_not_reuse_extracted_features
    lowercase_text = not args.do_not_lowercase_text
    lowercase_cngrams = args.lowercase_cngrams
    partition = args.partition
    mongodb_host = args.mongodb_host
    mongodb_port = args.mongodb_port
    update_batch_size = args.update_batch_size
    if update_batch_size < 1:
        raise ValueError('--update_batch_size/-batch_size should be greater '
                         'than 0.')

    # Make sure log file directory exists
    log_file_path = realpath(args.log_file_path)
    log_file_dir = dirname(log_file_path)
    if not exists(log_file_dir):
        makedirs(log_file_dir, exist_ok=True)

    # Setup file handler
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging_debug)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Print out some logging information about the upcoming tasks
    logdebug('Project directory: {0}'.format(project_dir))
    logdebug('Binarize features? {0}'.format(binarize))
    logdebug('Try to reuse previously-extracted features in the database? {0}'
             .format(reuse_features))
    logdebug('Lower-case text as part of the normalization step? {0}'
             .format(lowercase_text))
    logdebug('Lower-case character n-grams during feature extraction? {0}'
             .format(lowercase_cngrams))
    logdebug('Batch size for database updates: {0}'.format(update_batch_size))

    # Establish connection to MongoDB database collection
    loginfo('Connecting to MongoDB database on mongodb://{0}:{1}...'
            .format(mongodb_host, mongodb_port))
    try:
        db = connect_to_db(host=mongodb_host, port=mongodb_port)
    except ConnectionFailure as e:
        logerr('Unable to connect to MongoDB reviews collection.')
        logerr(e)
        raise e
    db.write_concern['w'] = 0

    # Get list of games
    game_files = get_game_files(game_files)

    # Iterate over the game files, extracting and adding/replacing
    # features to the database
    for game_file in game_files:
        game = splitext(game_file)[0]
        if partition == 'all':
            partition_string = ' from the "training" and "test" data partitions'
        else:
            partition_string = ' from the "{0}" data partition'.format(partition)
        loginfo('Extracting features{0} for {1}...'.format(partition_string, game))
        try:
            updates = \
                bulk_extract_features_and_update_db(db,
                                                    game,
                                                    partition,
                                                    reuse_nlp_feats=reuse_features,
                                                    use_binarized_nlp_feats=binarize,
                                                    lowercase_text=lowercase_text,
                                                    lowercase_cngrams=lowercase_cngrams,
                                                    update_batch_size=update_batch_size)
        except BulkWriteError as bwe:
            logerr('Encountered a BulkWriteError while executing the call to '
                   '`bulk_extract_features_and_update_db`.')
            raise bwe
    if updates:
        loginfo('{0} updates were made to the reviews collection.'
                .format(updates))
    else:
        raise ValueError('No updates were made.')


if __name__ == '__main__':
    main()
