

import os
import sys
import xlrd
import stat
import time
import logging
import argparse
import editpyxl
import multiprocessing
from pprint import pprint

from src.settings import load_settings_from_sheet
from apportionment import apportionment
from src.util import set_logging_level
from src.fetch_location_data import fetch_location_data

ALLOCATION_RESULT = 6
parser = argparse.ArgumentParser()
parser.add_argument(
    'dir',
    type=str,
    default=os.getcwd(),
    help='first positional argument, input working dir',
    nargs='?'
)
parser.add_argument(
    'input_xlsx',
    type=str,
    default='voting_excel.xlsm',
    help='first positional argument, input xlsx filepath',
    nargs='?'
)
parser.add_argument(
    '--log',
    type=str,
    default='info',
    help='log level, ex: --log debug'
)


def allocation(location_data: dict, settings: dict) -> dict:
    '''
        Main function for Allocation.
        Makes use of apportionment to keep service reqs close.

        Params:
            location_data (dict) : location data from xlsx,
            settings (dict) : sheet settings.

        Returns:
            (dict) : allocation results by location with expected wait times.
    '''
    print(f'allocation - machines available: {settings["NUM_MACHINES"]}')

    upper_service_req = 500
    lower_service_req = 1
    current_total = 0
    num_iterations = 0

    while num_iterations < settings['MAX_ITERATIONS'] and \
            abs(settings['NUM_MACHINES'] - current_total) > settings['ACCEPTABLE_RESOURCE_MISS']:
        # next service req to try
        current_service_req = (upper_service_req + lower_service_req) * 0.5

        # running apportionment on all locations
        logging.critical(f'allocation - running apportionment with service req: {current_service_req:.2f}')
        results = apportionment(location_data, settings)

        # collecting new total
        current_total = sum(res['Resource'] for res in results.values())
        logging.critical(f'allocation - used {current_total} machines at service req: {current_service_req:.2f}')

        # updating upper or lower bound
        if settings['NUM_MACHINES'] > current_total:
            upper_service_req = current_service_req
        else:
            lower_service_req = current_service_req

        num_iterations += 1

    # NOTE: could rerun final service_req 2 or more times here for guarantee
    return results


if __name__ == '__main__':
    multiprocessing.freeze_support()

    args = parser.parse_args()

    set_logging_level(args.log)

    # =========================================================================
    # Setup

    logging.info(f'reading {args.input_xlsx}')
    voting_config = xlrd.open_workbook(args.input_xlsx)

    # get settigns from input xlsx file
    settings = load_settings_from_sheet(voting_config.sheet_by_name(u'options'))

    # get voting location data from input xlsx file
    location_data = fetch_location_data(voting_config)

    # =========================================================================
    # Main

    start_time = time.perf_counter()

    try:
        results = allocation(location_data)
    except Exception:
        logging.info(f'fatal error')
        input()

    pprint(results)

    try:
        voting_config = editpyxl.Workbook()
        voting_config.open(args.input_xlsx)
        result_sheet = voting_config.active

        for index in results:
            cell = result_sheet.cell(row=index + 1, column=ALLOCATION_RESULT)
            cell.value = results[index]['Resource']

        os.chmod(args.input_xlsx, stat.S_IRWXU)
        voting_config.save(args.input_xlsx)
        os.system('start excel.exe ' + args.input_xlsx)
    except Exception as ex:
        print('err: ', ex)
        input("Press enter to exit.")
        sys.exit()

    logging.info(f'runtime: {time.perf_counter()-start_time}')
    logging.info('Done.')
    input("Press enter to exit.")
