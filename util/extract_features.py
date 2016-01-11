"""
:author: Matt Mulholland (mulhodm@gmail.com)
:date: September 18, 2015

Script used to extract features for review documents in the MongoDB
database.
"""
from os.path import (join,
                     dirname,
                     realpath,
                     splitext)

from argparse import (ArgumentParser,
                      ArgumentDefaultsHelpFormatter)

project_dir = dirname(dirname(realpath(__file__)))

def generate_update_query(update_dict: dict, binarized_features: bool = True) -> dict:
    """
    Generate an update query in the form needed for the MongoDB
    updates.

    :param update_dict: dictionary containing an `_id` field and a
                        `features` field
    :type update_dict: dict
    :param binarized_features: value representing whether or not the
                               features were binarized
    :type binarized_features: bool

    :returns: update query dictionary
    :rtype: dict
    """

    return {'$set': {'nlp_features': bson_encode(update_dict['features']),
                     'binarized': binarized_features,
                     'id_string': str(update_dict['_id'])}}


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
             default=join(project_dir, 'logs', 'replog_extract_features.txt'))
    args = parser.parse_args()

    # Imports
    import logging

    from pymongo.errors import (BulkWriteError,
                                ConnectionFailure)

    from src import log_format_string
    from src.mongodb import (connect_to_db,
                             generate_update_query)
    from src.datasets import get_game_files
    from src.features import bulk_extract_features

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

    # Setup logger and create logging handlers
    logger = logging.getLogger('extract_features')
    logging_debug = logging.DEBUG
    logger.setLevel(logging_debug)
    loginfo = logger.info
    logdebug = logger.debug
    logerr = logger.error
    logwarn = logger.warning
    formatter = logging.Formatter(log_format_string)
    sh = logging.StreamHandler()
    sh.setLevel(logging_debug)
    fh = logging.FileHandler(realpath(args.log_file_path))
    fh.setLevel(logging_debug)
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
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
        reviewdb = connect_to_db(host=mongodb_host, port=mongodb_port)
    except ConnectionFailure as e:
        logerr('Unable to connect to MongoDB reviews collection.')
        logerr(e)
        raise e
    reviewdb.write_concern['w'] = 0

    # Get list of games
    game_files = get_game_files(game_files,
                                join(dirname(dirname(__file__)), 'data'))

    # Iterate over the game files, extracting and adding/replacing
    # features to the database
    for game_file in game_files:
        game = splitext(game_file)[0]
        if partition == 'all':
            partition_string = (' from the "training" and "test" data partitions')
        else:
            partition_string = ' from the "{0}" data partition'.format(partition)
        loginfo('Extracting features{0} for {1}...'
                .format(partition_string, game))
        bulk = db.initialize_unordered_bulk_op()
        batch_size = 100
        updates = bulk_extract_features(reviewdb,
                                        partition,
                                        game,
                                        reuse_nlp_feats=reuse_features,
                                        use_binarized_nlp_feats=binarize,
                                        lowercase_text=lowercase_text,
                                        lowercase_cngrams=lowercase_cngrams)
        NO_MORE_UPDATES = False
        TOTAL_UPDATES = 0
        while not NO_MORE_UPDATES:

            # Add updates to the bulk update builder up until reaching
            # the batch size limit (or until we run out of data)
            i = 0
            while i < update_batch_size:
                try:
                    update = next(updates)
                except StopIteration:
                    NO_MORE_UPDATES = True
                    break
                (bulk
                 .find({'_id': update['_id']})
                 .updateOne(generate_update_query(update,
                                                  binarized_features=binarize)))
                i += 1
            TOTAL_UPDATES += i

            # Execute bulk update operations
            try:
                result = bulk.execute()
            except BulkWriteError as bwe:
                logerr(bwe.details)
                raise bwe
            logdebug(repr(result))

    if TOTAL_UPDATES:
        loginfo('{0} updates were made to the reviews collection.'
                .format(TOTAL_UPDATES))
    else:
        raise ValueError()


if __name__ == '__main__':
    main()
