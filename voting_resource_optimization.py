from pandas.core.frame import DataFrame

import sys
import xlrd
import math
import time
import logging
import argparse
import numpy as np
import pandas as pd
import scipy.stats as st

from tqdm import tqdm
from statistics import mean
from multiprocessing import Pool
from typing import List, Union, Optional

from src.settings import Settings
from src.util import set_logging_level
from src.voter_sim import voter_sim
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

# ===============================================================================
# Utility Functions


def voting_time_calcs(ballot_length: int):
    '''
        Calculates the min/mode/max/avg for a given ballot.

        Params:
            ballot_length (int) : ballot length for the location.

        Returns:
            (float) : vote time min,
            (float) : vote time mode,
            (float) : vote time max,
            (float) : vote time avg.
    '''
    min_voting_min = Settings.MIN_VOTING_MIN
    min_voting_mode = Settings.MIN_VOTING_MODE
    min_voting_max = Settings.MIN_VOTING_MAX
    min_ballot = Settings.MIN_BALLOT

    max_voting_min = Settings.MAX_VOTING_MIN
    max_voting_mode = Settings.MAX_VOTING_MODE
    max_voting_max = Settings.MAX_VOTING_MAX
    max_ballot = Settings.MAX_BALLOT

    vote_min = \
        min_voting_min + \
        (max_voting_min - min_voting_min) / \
        (max_ballot - min_ballot) * \
        (ballot_length - min_ballot)

    vote_mode = \
        min_voting_mode + \
        (max_voting_mode - min_voting_mode) / \
        (max_ballot - min_ballot) * \
        (ballot_length - min_ballot)

    vote_max = \
        min_voting_max + \
        (max_voting_max - min_voting_max) / \
        (max_ballot - min_ballot) * \
        (ballot_length - min_ballot)

    vote_avg = (vote_min + vote_mode + vote_max) / 3

    return vote_min, vote_mode, vote_max, vote_avg


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


def create_hypotheses_df(num_h):
    '''
        This function creates a dataframe to store hypotheses testing results.

        Params:
            num_h () : TODO.

        Returns:
            TODO
    '''
    # This dataframe will contain 1 record per machine count
    res_cols = ['Machines', 'Feasible', 'BatchAvg', 'BatchAvgMax']
    # Create an empty dataframe the same size as the locations dataframe
    voter_cols = np.zeros((num_h, len(res_cols)))
    hyp_results = pd.DataFrame(voter_cols, columns=res_cols)
    # Populates the Machine count field
    hyp_results['Machines'] = (hyp_results.index + 1).astype('int')

    return hyp_results


# ===============================================================================
# Main IZGBS Function


# NOTE: this appears to be the main izgbs function
def izgbs(
    voting_location_num,
    total_num,
    start,
    min_val,
    alpha,
    location_data: dict
):
    '''
        Main IZGBS function.

        Params:
            voting_location_num (int) : voting location number,
            total_num () : TODO,
            start () : TODO,
            min_val () : TODO,
            alpha () : TODO,
            location_data (dict) : location tab of input xlsx.

        Returns:
            () : TODO.
    '''
    # read in parameters from locations dataframe
    max_voters = location_data[voting_location_num]['Eligible Voters']
    ballot_length = location_data[voting_location_num]['Ballot Length Measure']
    arrival_rt = location_data[voting_location_num]['Arrival Mean']

    # calculate voting times
    vote_min, vote_mode, vote_max, vote_avg = voting_time_calcs(ballot_length)

    # create a dataframe for total number of machines
    feasible_df = create_hypotheses_df(total_num)

    # start with the start value specified
    hypotheses_remain = True
    num_test = start
    cur_upper = total_num
    cur_lower = min_val

    while hypotheses_remain:
        logging.info(f'Current upper bound: {cur_upper}')
        logging.info(f'Current lower bound: {cur_lower}')
        logging.info(f'\tTesting with: {num_test}')

        mean_wait_times = []
        max_wait_times = []

        # =====================================

        for _ in range(Settings.NUM_REPLICATIONS):
            results_dict = {
                f'Voter {i}': {
                    'Used': False
                }
                for i in range(max_voters)
            }

            # calculate voting times
            voter_sim(
                results_dict,
                max_voters,
                vote_min,
                vote_mode,
                vote_max,
                arrival_rt,
                num_test
            )

            # only keep results records that were actually used
            results_dict = {
                name: info for name, info in results_dict.items()
                if info['Used']
            }

            wait_times = [
                info['Voting_Start_Time'] - info['Arrival_Time']
                for info in results_dict.values()
            ]
            mean_wait_times.append(mean(wait_times))
            max_wait_times.append(max(wait_times))

        # =====================================

        avg_wait_time_avg = mean(mean_wait_times)
        max_wait_time_avg = mean(max_wait_times)
        max_wait_time_std = np.std(max_wait_times)

        # populate results
        feasible_df.loc[feasible_df.Machines == num_test, 'BatchAvg'] = avg_wait_time_avg
        feasible_df.loc[feasible_df.Machines == num_test, 'BatchAvgMax'] = max_wait_time_avg

        # calculate test statistic
        if max_wait_time_std > 0:
            z = (max_wait_time_avg - Settings.SERVICE_REQ + Settings.DELTA_INDIFFERENCE_ZONE) / \
                (max_wait_time_std / math.sqrt(Settings.NUM_BATCHES))
            p = st.norm.cdf(z)

            # feasible
            if p < alpha:
                # move to lower half
                feasible_df.loc[feasible_df.Machines >= num_test, 'Feasible'] = 1
                cur_upper = num_test
                num_test = math.floor((cur_upper - cur_lower) / 2) + cur_lower
            else:
                # move to upper half
                feasible_df.loc[feasible_df.Machines == num_test, 'Feasible'] = 0
                cur_lower = num_test
                num_test = math.floor((cur_upper - num_test) / 2) + cur_lower
        else:
            # move to lower half
            feasible_df.loc[feasible_df.Machines >= num_test, 'Feasible'] = 1
            cur_upper = num_test
            num_test = math.floor((cur_upper - cur_lower) / 2) + cur_lower

        # check if there are hypotheses left to test
        hypotheses_remain = cur_lower < cur_upper and cur_lower < num_test < cur_upper

    logging.info(feasible_df)

    return feasible_df


def evaluate_location(params: list) -> None:
    '''
        Runs IZGBS on a specified location, return results in dict best_results.

        Params:
            params (list) : location data dict, location id.

        Returns:
            best_result: a dict contain all field that going to fill in result_df.
    '''
    best_result = {}
    location_data, location_id = params

    logging.info(f'Starting Location: {location_id}')

    # Placeholder, use a different start value for later machines?
    # NOTE/TODO: apportionment vs appointment - Collin
    if Settings.MIN_ALLOC_FLG:
        start_val = math.ceil(
            (Settings.MAX_MACHINES - Settings.MIN_ALLOC) / 2) + Settings.MIN_ALLOC
        sas_alpha_value = Settings.ALPHA_VALUE / math.log2(Settings.MAX_MACHINES - Settings.MIN_ALLOC)

        loc_res = izgbs(
            location_id,
            Settings.MAX_MACHINES,
            start_val,
            Settings.MIN_ALLOC,
            sas_alpha_value,
            location_data
        )
    else:
        start_val = math.ceil((Settings.MAX_MACHINES - 1) / 2)
        sas_alpha_value = Settings.ALPHA_VALUE / math.log2(Settings.MAX_MACHINES - Settings.MIN_ALLOC)

        loc_res = izgbs(
            location_id,
            Settings.NUM_MACHINES,
            start_val,
            2,
            sas_alpha_value,
            location_data
        )

    loc_feas = loc_res[loc_res['Feasible'] == 1].copy()

    if not loc_feas.empty:
        # calculate fewest feasible machines
        mach_min = loc_feas['Machines'].min()

        # keep the feasible setup with the fewest number of machines
        loc_feas_min = loc_feas[loc_feas['Machines'] == mach_min].copy()

        # populate overall results with info for this location
        best_result['i'] = location_id
        best_result['Resource'] = mach_min
        best_result['Exp. Avg. Wait Time'] = loc_feas_min.iloc[0]['BatchAvg']
        best_result['Exp. Max. Wait Time'] = loc_feas_min.iloc[0]['BatchAvgMax']

        return best_result


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
