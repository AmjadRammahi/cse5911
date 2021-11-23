import math
import logging
import numpy as np
import scipy.stats as st
from statistics import mean
from src.voter_sim import voter_sim


def voting_time_calcs(ballot_length: int, settings: dict) -> tuple:
    '''
        Calculates the min/mode/max/avg for a given ballot.
        Params:
            ballot_length (int) : ballot length for the location,
            settings (dict) : sheet settings.

        Returns:
            (float) : vote time min,
            (float) : vote time mode,
            (float) : vote time max.
    '''
    vote_min = \
        settings['MIN_VOTING_MIN'] + \
        (settings['MAX_VOTING_MIN'] - settings['MIN_VOTING_MIN']) / \
        (settings['MAX_BALLOT'] - settings['MIN_BALLOT']) * \
        (ballot_length - settings['MIN_BALLOT'])

    vote_mode = \
        settings['MIN_VOTING_MODE'] + \
        (settings['MAX_VOTING_MODE'] - settings['MIN_VOTING_MODE']) / \
        (settings['MAX_BALLOT'] - settings['MIN_BALLOT']) * \
        (ballot_length - settings['MIN_BALLOT'])

    vote_max = \
        settings['MIN_VOTING_MAX'] + \
        (settings['MAX_VOTING_MAX'] - settings['MIN_VOTING_MAX']) / \
        (settings['MAX_BALLOT'] - settings['MIN_BALLOT']) * \
        (ballot_length - settings['MIN_BALLOT'])

    return vote_min, vote_mode, vote_max


def izgbs(
    max_machines: int,
    start_machines: int,
    min_machines: int,
    location_data: dict,
    settings: dict,
    memo: dict = {}
) -> dict:
    '''
        Main IZGBS function.
        Params:
            max_machines (int) : maximum allowed number of machines,
            start_machines (int) : starting number of machines to test,
            min_machines (int) : minimum allowed number of machines,
            location_data (list) : location data,
            settings (dict) : sheet settings,
            memo (dict) : memoization dict.

        Returns:
            (dict) : feasability of each resource amount.
    '''
    # read in parameters from locations dataframe
    max_voters = location_data['Eligible Voters']
    expected_voters = location_data['Likely or Exp. Voters']
    ballot_length = location_data['Ballot Length Measure']

    # calculate voting times
    vote_min, vote_mode, vote_max = voting_time_calcs(ballot_length, settings)
    sas_alpha_value = settings['ALPHA_VALUE'] / math.log2(settings['MAX_MACHINES'] - 1)

    # create a dataframe for total number of machines
    feasible_dict = {
        num_m + 1: {
            'Machines': num_m + 1,
            'Feasible': 0,
            'BatchAvg': 0,
            'BatchMaxAvg': 0
        }
        for num_m in range(min_machines, max_machines)
    }

    # start with the start value specified
    hypotheses_remain = True
    num_machines = start_machines
    cur_upper = max_machines
    cur_lower = min_machines

    while hypotheses_remain:
        logging.info(f'Current upper bound: {cur_upper}')
        logging.info(f'Current lower bound: {cur_lower}')
        logging.info(f'\tTesting with: {num_machines}')

        # NOTE: this is the inputs to voter_sim(), comma delimited (ballot_length is effectively vote_min/mode/max)
        # NOTE: these all `should` be ints, so '10.0' vs '10' should not be a problem
        key = f'{max_voters},{expected_voters},{ballot_length},{num_machines}'

        if key in memo:
            avg_wait_time_avg, max_wait_time_avg, max_wait_time_std = memo[key]
        else:
            batch_avg_wait_times = [[] for _ in range(settings['NUM_BATCHES'])]
            batch_max_wait_times = [[] for _ in range(settings['NUM_BATCHES'])]

            # =====================================

            # TODO: use AKPI

            for i in range(settings['NUM_REPLICATIONS']):
                # calculate voting times
                wait_times = voter_sim(
                    max_voters=max_voters,
                    expected_voters=expected_voters,
                    vote_time_min=vote_min,
                    vote_time_mode=vote_mode,
                    vote_time_max=vote_max,
                    num_machines=num_machines,
                    settings=settings
                )

                batch_avg_wait_times[i % settings['NUM_BATCHES']].append(mean(wait_times))
                batch_max_wait_times[i % settings['NUM_BATCHES']].append(max(wait_times))

            # =====================================

            # reduce individual batches to their mean's
            avg_wait_times = [
                mean(batch)
                for batch in batch_avg_wait_times
            ]
            max_wait_times = [
                mean(batch)
                for batch in batch_max_wait_times
            ]

            # collect statistics
            avg_wait_time_avg = mean(avg_wait_times)
            max_wait_time_avg = mean(max_wait_times)
            max_wait_time_std = np.std(max_wait_times)

            memo[key] = (avg_wait_time_avg, max_wait_time_avg, max_wait_time_std)

        # populate results
        feasible_dict[num_machines]['BatchAvg'] = avg_wait_time_avg
        feasible_dict[num_machines]['BatchMaxAvg'] = max_wait_time_avg

        # calculate test statistic (p)
        if max_wait_time_std > 0:  # NOTE: > 0, avoiding divide by 0 error
            z = (max_wait_time_avg - settings['SERVICE_REQ'] + settings['DELTA_INDIFFERENCE_ZONE']) / (max_wait_time_std / math.sqrt(settings['NUM_BATCHES']))
            p = st.norm.cdf(z)

            if p < sas_alpha_value:
                # move to lower half

                for key in feasible_dict:
                    if key >= num_machines:
                        feasible_dict[key]['Feasible'] = 1

                cur_upper = num_machines
                num_machines = math.floor((cur_upper - cur_lower) / 2) + cur_lower
            else:
                # move to upper half
                feasible_dict[num_machines]['Feasible'] = 0
                cur_lower = num_machines
                num_machines = math.floor((cur_upper - num_machines) / 2) + cur_lower
        else:
            # move to lower half
            feasible_dict[num_machines]['Feasible'] = 0
            cur_upper = num_machines
            num_machines = math.floor((cur_upper - cur_lower) / 2) + cur_lower

        # check if there are hypotheses left to test
        hypotheses_remain = cur_lower < cur_upper and cur_lower < num_machines < cur_upper

    logging.info(feasible_dict)

    return feasible_dict
