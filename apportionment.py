import xlrd
import math
import time
import logging
import argparse
import warnings
from tqdm import tqdm
from pprint import pprint
from multiprocessing import Pool
from typing import List, Union, Optional
from src.settings import Settings
from src.util import set_logging_level
from src.fetch_location_data import fetch_location_data
from src.evaluate_location import evaluate_location
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning

warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

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

def apportionment(location_data: dict) -> dict:
    '''
        Runs apportionment against the given locations.

        Params:
            location_data (dict) :
                contains the amt of voters and the ballot length for each location.

        Returns:
            (dict) : locations with the min feasible
                resource number and BatchAvg/BatchMaxAvg wait time.
    '''
    # NOTE: locations start at 1, not 0
    location_params = [
        location_data[i + 1]
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

    start_time = time.perf_counter()

    for location in location_data.values():
        location['NUM_MACHINES'] = Settings.MAX_MACHINES

    results = apportionment(location_data)

    pprint(results)

    logging.critical(f'runtime: {time.perf_counter()-start_time}')
    logging.critical('Done.')
