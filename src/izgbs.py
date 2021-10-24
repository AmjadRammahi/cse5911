import math
import logging
import numpy as np
import scipy.stats as st
from statistics import mean
from src.settings import Settings
from src.voter_sim import voter_sim
import src.global_var


def voting_time_calcs(ballot_length: int) -> tuple:
    '''
        Calculates the min/mode/max/avg for a given ballot.

        Params:
            ballot_length (int) : ballot length for the location.

        Returns:
            (float) : vote time min,
            (float) : vote time mode,
            (float) : vote time max.
    '''
    min_voting_min = src.global_var.MIN_VOTING_MIN
    min_voting_mode = src.global_var.MIN_VOTING_MODE
    min_voting_max = src.global_var.MIN_VOTING_MAX
    min_ballot = src.global_var.MIN_BALLOT

    max_voting_min = src.global_var.MAX_VOTING_MIN
    max_voting_mode = src.global_var.MAX_VOTING_MODE
    max_voting_max = src.global_var.MAX_VOTING_MAX
    max_ballot = src.global_var.MAX_BALLOT

    # VotingProcessingMin = MinVotingProcessingMin + (MaxVotingProcessingMin - MinVotingProcessingMin) / (MaxBallot - MinBallot) * (Ballot - MinBallot)
    # VotingProcessingMode = MinVotingProcessingMode + (MaxVotingProcessingMode - MinVotingProcessingMode) / (MaxBallot - MinBallot) * (Ballot - MinBallot)
    # VotingProcessingMax = MinVotingProcessingMax + (MaxVotingProcessingMax - MinVotingProcessingMax) / (MaxBallot - MinBallot) * (Ballot - MinBallot)
    # AverageVotingProcessing = (VotingProcessingMin + VotingProcessingMode + VotingProcessingMax) / 3

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

    return vote_min, vote_mode, vote_max


def izgbs(
    max_machines: int,
    start_machines: int,
    min_machines: int,
    sas_alpha_value: float,
    location_data: dict
):
    '''
        Main IZGBS function.

        Params:
            voting_location_num (int) : voting location number,
            max_machines (int) : maximum allowed number of machines,
            start_machines (int) : starting number of machines to test,
            min_machines (int) : minimum allowed number of machines,
            sas_alpha_value (float) : TODO,
            location_data (list) : location data.

        Returns:
            (pd.DataFrame) : feasability of each resource amt.
    '''
    # read in parameters from locations dataframe
    max_voters = location_data['Eligible Voters']
    expected_voters = location_data['Likely or Exp. Voters']
    ballot_length = location_data['Ballot Length Measure']
    # NOTE: arrival rate is currently recalculated within 'voter_sim.py' based
    # on the number of minutes the poll is open divided by the number of
    # expected voters.
    arrival_rt = location_data['Arrival Mean']

    # calculate voting times
    vote_min, vote_mode, vote_max = voting_time_calcs(ballot_length)

    # create a dataframe for total number of machines
    feasible_dict = {
        num_m + 1: {
            'Machines': num_m + 1,
            'Feasible': 0,
            'BatchAvg': 0,
            'BatchMaxAvg': 0
        }
        for num_m in range(max_machines)
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

        batch_avg_wait_times = [[] for _ in range(src.global_var.NUM_BATCHES)]
        batch_max_wait_times = [[] for _ in range(src.global_var.NUM_BATCHES)]

        # =====================================

        # TODO: use AKPI

        for i in range(src.global_var.NUM_REPLICATIONS):
            # calculate voting times
            wait_times = voter_sim(
                max_voters=max_voters,
                expected_voters=expected_voters,
                vote_time_min=vote_min,
                vote_time_mode=vote_mode,
                vote_time_max=vote_max,
                arrival_rt=arrival_rt,
                num_machines=num_machines
            )

            batch_avg_wait_times[i % src.global_var.NUM_BATCHES].append(mean(wait_times))
            batch_max_wait_times[i % src.global_var.NUM_BATCHES].append(max(wait_times))

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

        # populate results
        feasible_dict[num_machines]['BatchAvg'] = avg_wait_time_avg
        feasible_dict[num_machines]['BatchMaxAvg'] = max_wait_time_avg

        # calculate test statistic (p)
        if max_wait_time_std > 0:  # NOTE: > 0, avoiding divide by 0 error
            z = (max_wait_time_avg - src.global_var.SERVICE_REQ + src.global_var.DELTA_INDIFFERENCE_ZONE) / (max_wait_time_std / math.sqrt(src.global_var.NUM_BATCHES))
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
