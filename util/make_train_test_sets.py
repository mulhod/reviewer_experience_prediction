"""
:author: Matt Mulholland
:date: April 15, 2015

Script used to create training/test sets in a MongoDB database from
review data extracted from flat files.
"""
from os.path import (join,
                     exists,
                     abspath,
                     dirname,
                     realpath,
                     basename,
                     splitext)
from argparse import (ArgumentParser,
                      ArgumentDefaultsHelpFormatter)
from pymongo.errors import ConnectionFailure

project_dir = dirname(dirname(realpath(__file__)))

def main():
    parser = \
        ArgumentParser(usage='python make_train_test_sets.py --game_files '
                             'GAME_FILE1,GAME_FILE2,...[ OPTIONS]',
                       description='Build train/test sets for each game. Take'
                                   ' up to 21k reviews and split it 80/20 '
                                   'training/test, respectively, by default. '
                                   'Both the maximum size and the percentage '
                                   'split can be altered via command-line '
                                   'flags. All selected reviews will be put '
                                   'into the "reviews_project" database\'s '
                                   '"reviews" collection (which is being '
                                   ' hosted on lemur.montclair.edu on port '
                                   '27017).',
        formatter_class=ArgumentDefaultsHelpFormatter)
    _add_arg = parser.add_argument
    _add_arg('--game_files',
             help='Comma-separated list of file-names or "all" for all of the'
                  ' files (the game files should reside in the "data" '
                  'directory).',
             type=str,
             required=True)
    _add_arg('--max_size', '-m',
             help='Maximum number of reviews to get for training/testing (if '
                  'possible).',
             type=int,
             default=4000)
    _add_arg('--percent_train', '-%',
             help='Percent of selected reviews for which to use for the '
                  'training set, the rest going to the test set.',
             type=float,
             default=80.0)
    _add_arg('--convert_to_bins', '-bins',
             help='Number of sub-divisions of the hours-played values, e.g. '
                  'if 10 and the hours values range from 0 up to 1000, then '
                  'hours values 0-99 will become 1, 100-199 will become 2, '
                  'etc. (will probably be necessay to train a model that '
                  'actually is predictive to an acceptable degree); note that'
                  ' both hours values will be retained, the original under '
                  'the name "hours" and the converted value under the name '
                  '"hours_bin".',
             type=int,
             required=False)
    _add_arg('--bin_factor',
             help='If the --convert_to_bins/-bins argument is specified, '
                  'increase the sizes of the bins by the given factor so that'
                  ' bins in which there will be lots of instances will be '
                  'smaller in terms of range than bins that are more '
                  'sparsely-populated.',
             type=float,
             required=False)
    _add_arg('--make_reports', '-describe',
             help='Generate reports and histograms describing the data '
                  'filtering procedure.',
             action='store_true',
             default=False)
    _add_arg('--just_describe',
             help='Generate reports and histograms describing the data '
                  'filtering procedure, but then do NOT insert the reviews '
                  'into the MongoDB database.',
             action='store_true',
             default=False)
    _add_arg('--reports_dir',
             help='If -describe/--make_reports is used, put generated reports'
                  ' in the given directory.',
             type=str,
             required=False)
    _add_arg('-dbhost', '--mongodb_host',
             help='Host that the MongoDB server is running on.',
             type=str,
             default='localhost')
    _add_arg('--mongodb_port', '-dbport',
             help='Port that the MongoDB server is running on.',
             type=int,
             default=27017)
    _add_arg('--log_file_path', '-log',
             help='Path for log file.',
             type=str,
             default=join(project_dir, 'logs', 'replog_make_train_test_sets.txt'))
    args = parser.parse_args()

    # Imports
    import logging
    from sys import exit
    from os import listdir

    from pymongo import MongoClient

    from src.datasets import get_game_files
    from src.mongodb import (connect_to_db,
                             insert_train_test_reviews)

    # Make local copies of arguments
    game_files = args.game_files
    max_size = args.max_size
    percent_train = args.percent_train
    convert_to_bins = args.convert_to_bins
    bin_factor = args.bin_factor
    make_reports = args.make_reports
    just_describe = args.just_describe
    reports_dir = args.reports_dir
    mongodb_host = args.mongodb_host
    mongodb_port = args.mongodb_port

    # Initialize logging system
    logging_info = logging.INFO
    logger = logging.getLogger('make_train_test_sets')
    logger.setLevel(logging_info)

    # Create file handler
    fh = logging.FileHandler(abspath(args.log_file_path))
    fh.setLevel(logging_info)

    # Create console handler
    sh = logging.StreamHandler()
    sh.setLevel(logging_info)

    # Add nicer formatting
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -'
                                  ' %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    loginfo = logger.info
    logerror = logger.error
    logwarn = logger.warning

    # Make sure value passed in via the --convert_to_bins/-bins option
    # flag makes sense and, if so, assign value to variable bins (if
    # not, set bins equal to 0)
    if (convert_to_bins
        and convert_to_bins < 2):
        logerror('The value passed in via --convert_to_bins/-bins must be '
                 'greater than one since there must be multiple bins to '
                 'divide the hours played values. Exiting.')
        exit(1)
    elif convert_to_bins:
        bins = convert_to_bins
    else:
        bins = 0

    # Make sure that, if the --bin_factor argument is specified, the
    # --convert_to_bins/-bins argument was also specified
    if (bin_factor
        and not convert_to_bins):
        logerror('The --bin_factor argument was specified despite the fact '
                 'that the --convert_to_bins/-bins argument was not used. '
                 'Exiting.')
        exit(1)

    # Establish connection to MongoDB database
    loginfo('Connecting to MongoDB database on mongodb://{0}:{1}...'
            .format(mongodb_host, mongodb_port))
    reviewdb = connect_to_db(host=mongodb_host,
                             port=mongodb_port)
    reviewdb.write_concern['w'] = 0

    # Get path to the directories
    data_dir = join(project_dir, 'data')
    if reports_dir:
        reports_dir = realpath(reports_dir)

    # Make sure args make sense
    if max_size < 50:
        logerror('You can\'t be serious, right? You passed in a value of 50 '
                 'for the MAXIMUM size of the combination of training/test '
                 'sets? Exiting.')
        exit(1)
    if percent_train < 1.0:
        logerror('You can\'t be serious, right? You passed in a value of 1.0%'
                 ' for the percentage of the selected reviews that will be '
                 'devoted to the training set? That is not going to be enough'
                 ' training samples. Exiting.')
        exit(1)

    # Make sense of arguments
    if (make_reports
        and just_describe):
        logwarn('If the --just_describe and -describe/--make_reports option '
                'flags are used, --just_describe wins out, i.e., reports will'
                ' be generated, but no reviews will be inserted into the '
                'database.')
    elif (reports_dir
          and (make_reports
               or just_describe)):
        if not exists(reports_dir):
            logerror('The given --reports_dir path was invalid. Exiting.')
            exit(1)

    # Get list of games
    game_files = get_game_files(game_files, join(dirname(dirname(__file__)),
                                                 'data'))

    loginfo('Adding training/test partitions to Mongo DB for the following '
            'games: {0}'
            .format(', '.join([splitext(game)[0] for game in game_files])))
    loginfo('Maximum size for the combined training/test sets: {0}'
            .format(max_size))
    loginfo('Percentage split between training and test sets: {0:.2f}/{1:.2f}'
            .format(percent_train, 100.0 - percent_train))
    if make_reports:
        loginfo('Generating reports in {0}.'
            .format(reports_dir if reports_dir else join(data_dir, 'reports')))
    if just_describe:
        loginfo('Exiting after generating reports.')
    if bins:
        loginfo('Converting hours played values to {0} bins with a bin factor'
                ' of {1}.'.format(bins, bin_factor))

    # For each game in our list of games, we will read in the reviews
    # from the data file and then put entries in our MongoDB collection
    # with a key that identifies each review as either training or test
    for game_file in game_files:
        loginfo('Getting/inserting reviews for {}...'
                .format(splitext(basename(game_file))[0]))
        insert_train_test_reviews(reviewdb, abspath(join(data_dir, game_file)),
                                  max_size, percent_train, bins=bins,
                                  bin_factor=bin_factor, describe=make_reports,
                                  just_describe=just_describe,
                                  reports_dir=reports_dir if reports_dir
                                              else join(data_dir, 'reports'))

    loginfo('Complete.')

if __name__ == '__main__':
    main()
