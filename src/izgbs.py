import math
import logging
import scipy.stats as st
from src.AKPIp1 import AKPIp1


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
    location_data: dict,
    settings: dict,
    memo: dict = {}
) -> dict:
    '''
        Main IZGBS function.

        Params:
            max_machines (int) : maximum allowed number of machines,
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

    if settings['MAX_MACHINES'] - settings['MIN_MACHINES'] > 2:
        sas_alpha_value = settings['ALPHA_VALUE'] / math.log2(settings['MAX_MACHINES'] - settings['MIN_MACHINES'])
    else:
        sas_alpha_value = settings['ALPHA_VALUE']

    cur_upper = settings['MAX_MACHINES']
    cur_lower = settings['MIN_MACHINES']
    final_avg_wait_time = None
    final_max_wait_time = None
    final_quantile_wait_time = None

    while cur_upper > cur_lower + 1:
        num_machines = math.ceil((cur_upper + cur_lower) / 2)

        logging.info(f'Current upper bound: {cur_upper}')
        logging.info(f'Current lower bound: {cur_lower}')
        logging.info(f'\tTesting with: {num_machines}')

        # NOTE: this is the inputs to voter_sim(), comma delimited (ballot_length is effectively vote_min/mode/max)
        # NOTE: these all `should` be ints, so '10.0' vs '10' should not be a problem
        key = f'{max_voters},{expected_voters},{ballot_length},{num_machines}'

        if key in memo:
            # mean_is_higher, avg_wait, max_wait, quantile_wait = memo[key]
            mean_is_higher, avg_wait, max_wait = memo[key]
        else:
            # mean_is_higher, avg_wait, max_wait, quantile_wait = AKPIp1(
            mean_is_higher, avg_wait, max_wait = AKPIp1(
                sas_alpha_value=sas_alpha_value,
                max_voters=max_voters,
                expected_voters=expected_voters,
                vote_min=vote_min,
                vote_mode=vote_mode,
                vote_max=vote_max,
                num_machines=num_machines,
                settings=settings
            )

            memo[key] = (mean_is_higher, avg_wait, max_wait)  # , quantile_wait)

        if mean_is_higher:
            cur_lower = num_machines
        else:
            cur_upper = num_machines

            final_avg_wait_time = avg_wait
            final_max_wait_time = max_wait
            # final_quantile_wait_time = quantile_wait

    return num_machines, final_avg_wait_time, final_max_wait_time  # , final_quantile_wait_time
