import math
import logging
import numpy as np
from numba import njit
import pandas as pd
import scipy.stats as st
from statistics import mean
import time

from src.settings import Settings
from src.voter_sim import voter_sim


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
    # hyp_results = pd.DataFrame(voter_cols, columns=res_cols)
    # hyp_results['Machines'] = (hyp_results.index + 1).astype('int')
    # Populates the Machine count field

    for i in range(num_h):
        voter_cols[i][0] = i+1
    

    # return hyp_results
    return voter_cols


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
        # feasible_df.loc[feasible_df.Machines == num_test, 'BatchAvg'] = avg_wait_time_avg
        # feasible_df.loc[feasible_df.Machines == num_test, 'BatchAvgMax'] = max_wait_time_avg

        
        
        
        
        feasible_df[:,2][feasible_df[:,0] == num_test] = avg_wait_time_avg
        feasible_df[:,3][feasible_df[:,0] == num_test] = max_wait_time_avg
        
        




        # calculate test statistic
        if max_wait_time_std > 0:
            z = (max_wait_time_avg - Settings.SERVICE_REQ + Settings.DELTA_INDIFFERENCE_ZONE) / \
                (max_wait_time_std / math.sqrt(Settings.NUM_BATCHES))
            p = st.norm.cdf(z)

            # feasible
            if p < alpha:
                # move to lower half
                # feasible_df.loc[feasible_df.Machines >= num_test, 'Feasible'] = 1
                feasible_df[:,1][feasible_df[:,0] >= num_test] = 1
                cur_upper = num_test
                num_test = math.floor((cur_upper - cur_lower) / 2) + cur_lower
            else:
                # move to upper half
                # feasible_df.loc[feasible_df.Machines == num_test, 'Feasible'] = 0
                feasible_df[:,1][feasible_df[:,0] == num_test] = 0
                cur_lower = num_test
                num_test = math.floor((cur_upper - num_test) / 2) + cur_lower
        else:
            # move to lower half
            # feasible_df.loc[feasible_df.Machines >= num_test, 'Feasible'] = 1
            feasible_df[:,1][feasible_df[:,0] >= num_test] = 1
            cur_upper = num_test
            num_test = math.floor((cur_upper - cur_lower) / 2) + cur_lower

        # check if there are hypotheses left to test
        hypotheses_remain = cur_lower < cur_upper and cur_lower < num_test < cur_upper

    logging.info(feasible_df)
    exit()
    return feasible_df
