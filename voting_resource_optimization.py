from pandas.core.frame import DataFrame

import sys
import xlrd
import math
import time
import logging
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from multiprocessing import Pool
from typing import List, Union, Optional

from src.settings import Settings
from src.util import set_logging_level
from src.fetch_location_data import fetch_location_data
from src.evaluate_location import evaluate_location

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

# ===============================================================================
# Utility Functions


def create_loc_df(vote_locs):
    '''
        This function creates a dataframe of IZGBS results to be outputted.

        Params:
            vote_locs () : TODO.

        Returns:
            TODO
    '''
    res_cols = ['Resource', 'Exp. Avg. Wait Time', 'Exp. Max. Wait Time']
    # Create an empty dataframe the same size as the locations dataframe
    voter_cols = np.zeros((vote_locs, len(res_cols)))
    loc_results = pd.DataFrame(voter_cols, columns=res_cols)
    # Populates the location ID field
    loc_results['Locations'] = (loc_results.index + 1).astype('str')

    return loc_results


# ===============================================================================
# Main IZGBS Function


def populate_result_df(results: list, result_df: DataFrame) -> None:
    '''
        Store IZGBS run results in loc_df_results.

        Params:
            results (list) : lists of result from izgbs,
            result_df (DataFrame) : an empty dataframe intended to host results.

        Returns:
            None.
    '''
    for result in results:
        result_df.loc[
            result_df.Locations == str(result['i']),
            'Resource'
        ] = result['Resource']

        result_df.loc[
            result_df.Locations == str(result['i']),
            'Exp. Avg. Wait Time'
        ] = result['Exp. Avg. Wait Time']

        result_df.loc[
            result_df.Locations == str(result['i']),
            'Exp. Max. Wait Time'
        ] = result['Exp. Max. Wait Time']


if __name__ == '__main__':
    args = parser.parse_args()

    set_logging_level(args.log)

    # =========================================================================
    # Setup

    logging.info(f'reading {args.input_xlsx}')
    voting_config = xlrd.open_workbook(args.input_xlsx)

    # get voting location data from input xlsx file
    location_data = fetch_location_data(voting_config)

    loc_df_results = create_loc_df(Settings.NUM_LOCATIONS + 1)

    # =========================================================================
    # Main

    start_time = time.perf_counter()

    location_params = [
        [location_data, i]
        for i in range(1, Settings.NUM_LOCATIONS)
    ]

    pool = Pool()

    results = [
        result
        for result in tqdm(
            pool.imap(evaluate_location, location_params),
            total=len(location_params)
        )
    ]

    populate_result_df(results, loc_df_results)

    print(loc_df_results)

    logging.critical(f'runtime: {time.perf_counter()-start_time}')
    logging.critical('Done.')
