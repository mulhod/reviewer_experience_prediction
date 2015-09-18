'''
:author: Matt Mulholland
:date: September 18, 2015

Script used to extract features for review documents in the Mongo database.
'''
from os.path import (dirname,
                     realpath,
                     splitext)
from argparse import (ArgumentParser,
                      ArgumentDefaultsHelpFormatter)

project_dir = dirname(dirname(realpath(__file__)))

def main():
    parser = ArgumentParser(usage='python extract_features.py --game_files '
                                  'GAME_FILE1,GAME_FILE2,...[ OPTIONS]',
        description='Extract features and add them to the Mongo database.',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser_add_argument = parser.add_argument
    parser_add_argument('--game_files',
        help='Comma-separated list of file-names or "all" for all of the '
             'files (the game files should reside in the "data" directory).',
        type=str,
        required=True)
    parser_add_argument('--do_not_binarize_features',
        help='Do not make all non-zero feature frequencies equal to 1.',
        action='store_true',
        default=False)
    parser_add_argument('--do_not_lowercase_text',
        help='Do not make lower-casing part of the review text '
             'normalization step, which affects word n-gram-related '
             'features.',
        action='store_true',
        default=False)
    parser_add_argument('--lowercase_cngrams',
        help='Lower-case the review text before extracting character n-gram '
             'features.',
        action='store_true',
        default=False)
    parser_add_argument('--partition',
        help='Data partition, i.e., "training", "test", etc. Value must be a '
             'valid "partition" value in the Mongo database.',
        type=str,
        default='training')
    parser_add_argument('--mongodb_port', '-dbport',
        help='Port that the MongoDB server is running.',
        type=int,
        default=27017)
    parser_add_argument('--log_file_path', '-log',
        help='Path to log file.',
        type=str,
        default=join(project_dir,
                     'logs',
                     'replog_train.txt'))
    args = parser.parse_args()

    imports
    from util.mongodb import connect_to_db

    # Make local copies of arguments
    game_files = args.game_files
    do_not_lowercase_text = args.do_not_lowercase_text
    lowercase_cngrams = args.lowercase_cngrams
    do_not_binarize_features = args.do_not_binarize_features
    partition = args.partition
    mongodb_port = args.mongodb_port

    # Setup logger and create logging handlers
    import logging
    logger = logging.getLogger('train')
    logging_debug = logging.DEBUG
    logger.setLevel(logging_debug)
    loginfo = logger.info
    logdebug = logger.debug
    logerr = logger.error
    logwarn = logger.warning
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -'
                                  ' %(message)s')
    sh = logging.StreamHandler()
    sh.setLevel(logging_debug)
    fh = logging.FileHandler(realpath(args.log_file_path))
    fh.setLevel(logging_debug)
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)

    # Print out some logging information about the upcoming tasks
    logdebug('Project directory: {}'.format(project_dir))
    binarize = not do_not_binarize_features
    logdebug('Binarize features? {}'.format(binarize))
    lowercase_text = not do_not_lowercase_text
    logdebug('Lower-case text as part of the normalization step? {}'
             .format(lowercase_text))
    logdebug('Just extract features? {}'.format(just_extract_features))

    # Establish connection to MongoDB database collection
    reviewdb = connect_to_db(mongodb_port)
    reviewdb.write_concern['w'] = 0

    # Get list of games
    game_files = get_game_files(game_files,
                                data_dir_path)

    # Iterate over the game files, extracting features and adding the features
    # to the database
    for game_file in game_files:
        game = splitext(game_file)[0]
        loginfo('Extracting features from the training data for {}...'
                .format(game))
        

if __name__ == '__main__':
    main()
