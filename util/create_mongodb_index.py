#!/usr/env python3.4
import sys
from pymongo import ASCENDING
from util.mongodb import connect_to_db
from argparse import (ArgumentParser,
                      ArgumentDefaultsHelpFormatter)

def main(argv=None):
    parser = ArgumentParser(description='Run incremental learning '
                                        'experiments.',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('-dbhost', '--mongodb_host',
        help='Host that the MongoDB server is running on.',
        type=str,
        default='localhost')
    parser.add_argument('--mongodb_port', '-dbport',
        help='Port that the MongoDB server is running on.',
        type=int,
        default=37017)
    args = parser.parse_args()

    # Connect to MongoDB database
    print('Connecting to MongoDB database at {0}:{1}...'
          .format(args.mongodb_host, args.mongodb_port), file=sys.stderr)
    db = connect_to_db(args.mongodb_host, args.mongodb_port)

    # Create index on 'steam_id_number' so that cursors can be sorted
    # on that particular key
    print('Creating index on the "steam_id_number" key...', file=sys.stderr)
    db.create_index('steam_id_number', ASCENDING)
    print('Created new index "steam_id_number_1" in reviews collection.',
          file=sys.stderr)
