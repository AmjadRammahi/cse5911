from editpyxl import workbook
import xlrd
import stat
import editpyxl
import time
import logging
import argparse
import multiprocessing
import os
import sys
from tqdm import tqdm
from pprint import pprint
from multiprocessing import Pool

from src.settings import Settings
from src.settings import load_settings_from_sheet
from src.util import set_logging_level
from src.fetch_location_data import fetch_location_data
from src.evaluate_location import evaluate_location

APPORTIONMENT_RESULT = 5
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
    help='second positional argument, input xlsx filepath',
    nargs='?'
)
parser.add_argument(
    '--log',
    type=str,
    default='info',
    help='log level, ex: --log debug'
)


def apportionment(location_data: dict, service_req: float) -> dict:
    '''
        Runs apportionment against the given locations.

        Params:
            location_data (dict) :
                contains the amt of voters and the ballot length for each location,
            service_req (float) : max service requirement.

        Returns:
            (dict) : locations with the min feasible
                resource number and BatchAvg/BatchMaxAvg wait time.
    '''
    # NOTE: locations start at 1, not 0
    location_params = [
        (location_data[i + 1], service_req)
        for i in range(Settings.NUM_LOCATIONS)
    ]

    pool = Pool()

    return {
        i + 1: result
        for i, result in enumerate(tqdm(
            pool.imap(evaluate_location, location_params),
            total=len(location_params)
        ))
    }


if __name__ == '__main__':
    multiprocessing.freeze_support()
    args = parser.parse_args()

    set_logging_level(args.log)

    # =========================================================================
    # Setup
    os.chdir(args.dir)
    logging.info(f'Program Initializing...')
    logging.info(f'reading {args.input_xlsx}')
    print("Current Path: ", os.getcwd())
    print("open_workbook target: ", args.input_xlsx)
    voting_config = xlrd.open_workbook(args.input_xlsx, on_demand=True)
    # get voting location data from input xlsx file
    location_data = fetch_location_data(voting_config)
    # get settings from input xlsx file
    load_settings_from_sheet(voting_config.sheet_by_name(u'options'))

    # =========================================================================
    # Main

    start_time = time.perf_counter()

    try:
        results = apportionment(location_data, Settings.SERVICE_REQ)
    except Exception:
        logging.info(f'fatal error')
        input()

    pprint(results)

    try:
        voting_config = editpyxl.Workbook()
        voting_config.open(args.input_xlsx)
        result_sheet = voting_config.active

        for index in results:
            cell = result_sheet.cell(row=index + 1, column=APPORTIONMENT_RESULT)
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
