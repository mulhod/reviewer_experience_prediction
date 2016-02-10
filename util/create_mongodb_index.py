"""
Create an index in the MongoDB reviews collection on the
'steam_id_number' key if one does not already exist.

:author: Matt Mulholland
:date: November, 2015
"""
import logging

from argparse import (ArgumentParser,
                      ArgumentDefaultsHelpFormatter)

from src import formatter

# Set up logger
logger = logging.getLogger('util.create_mongodb_index')
logging_info = logging.INFO
logger.setLevel(logging_info)
sh = logging.StreamHandler()
sh.setLevel(logging_info)
sh.setFormatter(formatter)
logger.addHandler(sh)


def main(argv=None):
    parser = ArgumentParser(description='Run incremental learning '
                                        'experiments.',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    _add_arg = parser.add_argument
    _add_arg('-dbhost', '--mongodb_host',
             help='Host that the MongoDB server is running on.',
             type=str,
             default='localhost')
    _add_arg('--mongodb_port', '-dbport',
             help='Port that the MongoDB server is running on.',
             type=int,
             default=37017)
    args = parser.parse_args()

    # Imports
    import sys

    from pymongo import ASCENDING
    from pymongo.errors import ConnectionFailure

    from src.mongodb import connect_to_db

    # Connect to MongoDB database
    logger.info('Connecting to MongoDB database at {0}:{1}...'
                .format(args.mongodb_host, args.mongodb_port))
    try:
        db = connect_to_db(args.mongodb_host, args.mongodb_port)
    except ConnectionFailure as e:
        logger.error('Failed to connect to the MongoDB database collection.')
        raise e

    # Create index on 'steam_id_number' so that cursors can be sorted
    # on that particular key
    logger.info('Creating index on the "steam_id_number" key.')
    db.create_index('steam_id_number', ASCENDING)
    logger.info('Created new index named "steam_id_number_1" in the "reviews" '
                'collection.')


if __name__ == '__main__':
    main()
