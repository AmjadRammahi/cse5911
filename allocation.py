import xlrd
import logging
import argparse
from pprint import pprint

from apportionment import apportionment
from src.settings import Settings
from src.util import set_logging_level
from src.fetch_location_data import fetch_location_data

parser = argparse.ArgumentParser()
parser.add_argument(
    'input_xlsx',
    type=str,
    help='first positional argument, input xlsx filepath'
)
parser.add_argument(
    '--log',
    type=str,
    default='info',
    help='log level, ex: --log debug'
)


if __name__ == '__main__':
    args = parser.parse_args()

    set_logging_level(args.log)

    # =========================================================================
    # Setup

    logging.info(f'reading {args.input_xlsx}')
    voting_config = xlrd.open_workbook(args.input_xlsx)

    # get voting location data from input xlsx file
    location_data = fetch_location_data(voting_config)

    # =========================================================================
    # Main

    # NOTE/TODO: use a binary search on the service_req to keep wait times uniform
    # NOTE: rerun final service_req 2 or more times for guarantee

    total_machines_available = 50

    for location in location_data.values():
        location['NUM_MACHINES'] = Settings.MAX_MACHINES

    apportionment_results = apportionment(location_data)
    current_total_machines = sum(x['Resource'] for x in apportionment_results.values())

    if current_total_machines > total_machines_available:
        # subtract machines proportionally such that the new sum is the total_machines_available
        for i, location in enumerate(location_data.values()):
            location['NUM_MACHINES'] = int(
                apportionment_results[i + 1]['Resource'] /
                current_total_machines *
                total_machines_available
            )

        # rerun apportionment to get new times
        apportionment_results = apportionment(location_data)
        current_total_machines = sum(x['Resource'] for x in apportionment_results.values())

        # round robin increment resources for those with highest wait times
        ordered = sorted(
            [(i, location['Exp. Max. Wait Time'])
             for i, location in apportionment_results.items()
             ],
            key=lambda x: x[1]
        )
        curr = 0

        while current_total_machines < total_machines_available:
            if curr == len(apportionment_results):
                curr = 0

            apportionment_results[ordered[curr][0]]['Resource'] += 1
            current_total_machines += 1
            curr += 1

    pprint(apportionment_results)
