import xlrd
import logging
import argparse
from pprint import pprint

import src.global_var
from src.settings import Settings
from apportionment import apportionment
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

    Settings

    logging.info(f'reading {args.input_xlsx}')
    voting_config = xlrd.open_workbook(args.input_xlsx)

    # get voting location data from input xlsx file
    location_data = fetch_location_data(voting_config)

    # =========================================================================
    # Main

    total_machines_available = 120
    acceptable_resource_miss = 10

    upper_service_req = 500
    lower_service_req = 1
    current_total = 0
    num_iterations = 0

    while num_iterations < 20 and \
            abs(total_machines_available - current_total) > acceptable_resource_miss:
        # next service req to try
        current_service_req = (upper_service_req + lower_service_req) * 0.5

        # running apportionment on all locations
        logging.critical(f'allocation - running apportionment with service req: {current_service_req:.2f}')
        results = apportionment(location_data, current_service_req)

        # collecting new total
        current_total = sum(res['Resource'] for res in results.values())

        # updating upper or lower bound
        if total_machines_available > current_total:
            upper_service_req = current_service_req
        else:
            lower_service_req = current_service_req

    # NOTE: could rerun final service_req 2 or more times here for guarantee
    pprint(results)
